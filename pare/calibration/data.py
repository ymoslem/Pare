"""Calibration data helpers.

GPTQ, AWQ and SmoothQuant estimate activation statistics from a small set of
token sequences. ``load_wikitext2_calibration`` builds the set that Pare's
benchmarks use: the WikiText-2 training split, joined with blank lines,
tokenized once, and cut into the first ``n_samples`` consecutive windows of
``seq_len`` tokens.

Any list of token tensors works as calibration data. Each tensor is one
sequence, shaped ``[seq_len]`` or ``[1, seq_len]``.

Usage::

    from pare.calibration.data import load_wikitext2_calibration
    calib = load_wikitext2_calibration(tokenizer, n_samples=128, seq_len=2048)
"""

from __future__ import annotations

from typing import Any

from torch import Tensor

_WIKITEXT2 = ("Salesforce/wikitext", "wikitext-2-raw-v1")


def load_wikitext2_calibration(
    tokenizer: Any,
    n_samples: int = 128,
    seq_len: int = 2048,
    split: str = "train",
) -> list[Tensor]:
    """Return ``n_samples`` WikiText-2 windows of ``seq_len`` tokens each.

    Args:
        tokenizer: A HuggingFace tokenizer matching the model to calibrate.
        n_samples: Number of sequences. 128 is the GPTQ and AWQ paper default.
        seq_len:   Tokens per sequence. 2048 is the usual choice for 7B models.
        split:     WikiText-2 split to draw from. Use ``"train"`` for real
                   runs; the test split is reserved for perplexity evaluation.

    Returns:
        A list of ``n_samples`` int64 tensors, each shaped ``[1, seq_len]``.

    Raises:
        ImportError: If the ``datasets`` package is not installed.
        ValueError:  If the split has fewer than ``n_samples * seq_len`` tokens.
    """
    text = _load_wikitext2_text(split)
    tokens = tokenizer(text, return_tensors="pt").input_ids
    return chunk_tokens(tokens, n_samples, seq_len)


def chunk_tokens(tokens: Tensor, n_samples: int, seq_len: int) -> list[Tensor]:
    """Cut a token stream into the first ``n_samples`` windows of ``seq_len``.

    Args:
        tokens:    Token ids shaped ``[n_tokens]`` or ``[1, n_tokens]``.
        n_samples: Number of windows to return.
        seq_len:   Tokens per window.

    Returns:
        A list of ``n_samples`` tensors, each shaped ``[1, seq_len]``.
    """
    tokens = tokens.reshape(1, -1)
    needed = n_samples * seq_len
    if tokens.shape[1] < needed:
        raise ValueError(
            f"Need {needed} tokens for {n_samples} sequences of {seq_len}, "
            f"but the text has only {tokens.shape[1]}. Reduce n_samples or seq_len."
        )
    return [tokens[:, i * seq_len : (i + 1) * seq_len] for i in range(n_samples)]


def _load_wikitext2_text(split: str) -> str:
    """Download one WikiText-2 split and join its lines with blank lines."""
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Loading WikiText-2 needs the 'datasets' package: pip install datasets"
        ) from None

    data = load_dataset(*_WIKITEXT2, split=split)
    return "\n\n".join(data["text"])
