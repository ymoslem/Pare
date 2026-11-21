"""Tests for pare.eval.perplexity helpers that need no model download."""

from types import SimpleNamespace

from pare.eval.perplexity import _infer_seq_len


class TestInferSeqLen:
    def _model(self, **config):
        return SimpleNamespace(config=SimpleNamespace(**config))

    def test_capped_at_a_short_context_window(self):
        # GPT-2 style config: n_positions=1024 is below the 2048 default.
        assert _infer_seq_len(self._model(n_positions=1024)) == 1024

    def test_default_for_long_contexts(self):
        assert _infer_seq_len(self._model(max_position_embeddings=131072)) == 2048
        assert _infer_seq_len(self._model(max_position_embeddings=4096)) == 2048

    def test_window_equal_to_default_is_kept(self):
        # Llama-1 and OPT have 2048 positions, the length their published
        # perplexities use.
        assert _infer_seq_len(self._model(max_position_embeddings=2048)) == 2048

    def test_max_position_embeddings_takes_precedence(self):
        model = self._model(max_position_embeddings=1536, n_positions=1024)
        assert _infer_seq_len(model) == 1536

    def test_default_without_config(self):
        assert _infer_seq_len(SimpleNamespace()) == 2048
        assert _infer_seq_len(SimpleNamespace(), default=512) == 512
