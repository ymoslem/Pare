"""Tests for pare.calibration.data and the calibration input accepted by quantize()."""

import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from pare import QuantConfig, quantize
from pare.calibration import data
from pare.calibration.data import chunk_tokens, load_wikitext2_calibration
from pare.layers.linear import QuantizedLinear

# ---------------------------------------------------------------------------
# chunk_tokens
# ---------------------------------------------------------------------------

class TestChunkTokens:
    def test_shapes_and_order(self):
        tokens = torch.arange(20).unsqueeze(0)          # [1, 20]
        chunks = chunk_tokens(tokens, n_samples=3, seq_len=5)
        assert len(chunks) == 3
        assert all(c.shape == (1, 5) for c in chunks)
        assert torch.equal(torch.cat(chunks, dim=1), tokens[:, :15])

    def test_accepts_1d_stream(self):
        chunks = chunk_tokens(torch.arange(12), n_samples=2, seq_len=6)
        assert chunks[0].shape == (1, 6)
        assert torch.equal(chunks[1][0], torch.arange(6, 12))

    def test_too_short_raises(self):
        with pytest.raises(ValueError, match="Need 30 tokens"):
            chunk_tokens(torch.arange(20), n_samples=3, seq_len=10)


# ---------------------------------------------------------------------------
# load_wikitext2_calibration (without downloading anything)
# ---------------------------------------------------------------------------

class _StubTokenizer:
    """Maps every character to one token, so token counts are predictable."""

    def __call__(self, text, return_tensors=None):
        ids = torch.tensor([[ord(c) for c in text]])
        return SimpleNamespace(input_ids=ids)


class TestLoadWikitext2Calibration:
    def test_returns_requested_windows(self, monkeypatch):
        seen = {}

        def fake_text(split):
            seen["split"] = split
            return "x" * 1000

        monkeypatch.setattr(data, "_load_wikitext2_text", fake_text)
        calib = load_wikitext2_calibration(_StubTokenizer(), n_samples=4, seq_len=100)

        assert seen["split"] == "train"
        assert len(calib) == 4
        assert all(c.shape == (1, 100) for c in calib)
        assert all(c.dtype == torch.int64 for c in calib)

    def test_split_is_passed_through(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(
            data, "_load_wikitext2_text", lambda split: seen.setdefault("split", split) and "x" * 50
        )
        load_wikitext2_calibration(_StubTokenizer(), n_samples=1, seq_len=10, split="test")
        assert seen["split"] == "test"

    def test_missing_datasets_package_gives_install_hint(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "datasets", None)
        with pytest.raises(ImportError, match="pip install datasets"):
            load_wikitext2_calibration(_StubTokenizer(), n_samples=1, seq_len=10)


# ---------------------------------------------------------------------------
# quantize() accepts [seq_len] and [1, seq_len] calibration tensors alike
# ---------------------------------------------------------------------------

class _TokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 32)
        self.proj = nn.Linear(32, 16)

    def forward(self, input_ids):
        return self.proj(self.embed(input_ids))


class TestCalibrationInputShapes:
    def _quantized_weight(self, calib):
        torch.manual_seed(0)
        model = _TokenModel()
        cfg = QuantConfig(bits=4, scheme="gptq", granularity="per_channel")
        quantize(model, cfg, calibration_data=calib)
        assert isinstance(model.proj, QuantizedLinear)
        return model.proj.dequantize()

    def test_1d_and_2d_sequences_give_the_same_result(self):
        torch.manual_seed(1)
        seqs_2d = [torch.randint(0, 64, (1, 8)) for _ in range(4)]
        seqs_1d = [s[0] for s in seqs_2d]
        assert torch.equal(self._quantized_weight(seqs_1d), self._quantized_weight(seqs_2d))

    def test_non_tensor_raises(self):
        with pytest.raises(TypeError, match=r"calibration_data\[0\]"):
            quantize(
                _TokenModel(), QuantConfig(bits=4, scheme="gptq"), calibration_data=[[1, 2, 3]]
            )

    def test_batched_inputs_pass_through_unchanged(self):
        # Models that take features directly are calibrated with [batch, seq, dim]
        # activations, which must reach the model exactly as given.
        from pare import _as_batches

        x = torch.randn(2, 4, 32)
        assert _as_batches([x])[0] is x
