"""Validate AWQ INT4 on Llama-2-7B-hf.


FP16 baseline for Llama-2-7B: 5.47 PPL.
Expected AWQ INT4: ~5.55–5.65 delta (similar to GPTQ, sometimes slightly better).

    
    python scripts/validate_awq_llama2.py
"""

import os
import sys
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pare import QuantConfig, quantize
from pare.calibration.data import load_wikitext2_calibration
from pare.eval.perplexity import evaluate_perplexity

MODEL_ID  = "meta-llama/Llama-2-7b-hf"
DEVICE    = "cuda"
SEQ_LEN   = 2048
N_CALIB   = 128

def main():
    token = os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        raise RuntimeError("Set HUGGING_FACE_HUB_TOKEN")

    print(f"[validate] Loading {MODEL_ID} ...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=token)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, token=token, device_map="auto",
    )
    model.eval()

    print("[validate] Preparing calibration data ...", flush=True)
    calib = load_wikitext2_calibration(tokenizer, n_samples=N_CALIB, seq_len=SEQ_LEN)

    config = QuantConfig(bits=4, scheme="awq", granularity="per_group", group_size=128)

    print("[validate] Running AWQ INT4 (g=128) ...", flush=True)
    t0 = time.time()
    quantize(model, config, calibration_data=calib, device=DEVICE)
    print(f"[validate] AWQ done in {time.time() - t0:.0f}s", flush=True)

    print("[validate] Evaluating PPL on WikiText-2 test ...", flush=True)
    ppl = evaluate_perplexity(
        model, tokenizer, dataset="wikitext2",
        seq_len=SEQ_LEN, n_samples=None, device=DEVICE,
    )

    print("\n" + "="*60)
    print(f"  Llama-2-7B  AWQ INT4  WikiText-2 PPL: {ppl:.2f}")
    print(f"  FP16 baseline: 5.47  delta: +{ppl - 5.47:.2f}")
    print("="*60)
    sys.exit(0 if ppl <= 5.70 else 1)

if __name__ == "__main__":
    main()
