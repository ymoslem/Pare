"""Perplexity evaluation on WikiText-2 and C4.

Perplexity is the standard metric for LLM quantization papers:
    PPL = exp( -1/N * Σ log p(token_i | context) )

Lower is better. FP16 baseline is the reference; a good quantization
method should add < 0.5 PPL for INT8 and < 1.0 PPL for INT4 per-group.

Usage::

    from pare.eval.perplexity import evaluate_perplexity
    ppl = evaluate_perplexity(model, tokenizer, dataset="wikitext2")
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


def evaluate_perplexity(
    model: nn.Module,
    tokenizer: Any,
    dataset: str = "wikitext2",
    seq_len: int | None = None,
    n_samples: int | None = None,
    device: str | torch.device = "cpu",
    verbose: bool = False,
) -> float:
    """Compute perplexity on WikiText-2 or C4 test split.

    Args:
        model:     Any causal LM with a standard HuggingFace-compatible
                   forward signature (input_ids → logits).
        tokenizer: Matching tokenizer.
        dataset:   ``"wikitext2"`` or ``"c4"``.
        seq_len:   Context window length (stride = seq_len, no overlap).
                   None (default) means 2048, or the model's own context
                   window when that is shorter (see ``_infer_seq_len``).
                   2048 is the length used in the GPTQ and AWQ papers.
        n_samples: Cap the number of sequences evaluated (None = all).
                   Useful for quick CPU smoke tests (e.g. n_samples=10).
        device:    Where to run evaluation.
        verbose:   Print per-batch NLL if True.

    Returns:
        Perplexity as a float.
    """
    model = model.to(device)
    model.eval()

    if seq_len is None:
        seq_len = _infer_seq_len(model)

    encodings = _load_and_encode(tokenizer, dataset)
    total_nll = 0.0
    total_tokens = 0
    seq_count = 0

    with torch.no_grad():
        n_seq = encodings.size(1) // seq_len
        if n_samples is not None:
            n_seq = min(n_seq, n_samples)

        for i in range(n_seq):
            chunk = encodings[:, i * seq_len : (i + 1) * seq_len].to(device)
            labels = chunk.clone()

            outputs = model(input_ids=chunk, labels=labels)
            nll = outputs.loss.item()   # mean NLL over this chunk

            total_nll += nll * seq_len
            total_tokens += seq_len
            seq_count += 1

            if verbose:
                print(f"  seq {i+1}/{n_seq}  NLL={nll:.4f}  PPL={math.exp(nll):.2f}")

    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def _infer_seq_len(model: nn.Module, default: int = 2048) -> int:
    """Return ``default``, capped at the model's context window.

    The window is read from ``max_position_embeddings``, or ``n_positions``
    for GPT-2 style configs. Models with no config keep ``default``.
    Evaluating GPT-2 (1024 positions) at 2048 would index past its position
    embeddings; capping gives the length its published perplexities use.
    """
    config = getattr(model, "config", None)
    max_len = getattr(config, "max_position_embeddings", None) or getattr(
        config, "n_positions", None
    )
    if not max_len:
        return default
    return int(min(default, max_len))


# Cache of tokenized corpora, keyed by (tokenizer identity, dataset name).
# _load_and_encode is called once per evaluate_perplexity() call; without this,
# comparing several quantized variants of the same model re-tokenizes the full
# test corpus from scratch every time, even though it's identical each call.
_ENCODING_CACHE: dict[tuple[str, str], Tensor] = {}


def _load_and_encode(tokenizer: Any, dataset: str) -> Tensor:
    """Download and tokenize the test split, return as a flat token tensor."""
    cache_key = (getattr(tokenizer, "name_or_path", repr(tokenizer)), dataset)
    if cache_key in _ENCODING_CACHE:
        return _ENCODING_CACHE[cache_key]

    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Install the 'datasets' package: pip install datasets"
        )

    if dataset == "wikitext2":
        data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        text = "\n\n".join(data["text"])
    elif dataset == "c4":
        data = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        # Take first 1000 documents for a manageable eval
        text = "\n\n".join(d["text"] for d, _ in zip(data, range(1000)))
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}. Use 'wikitext2' or 'c4'.")

    tokens: Tensor = tokenizer.encode(text, return_tensors="pt")
    _ENCODING_CACHE[cache_key] = tokens
    return tokens
