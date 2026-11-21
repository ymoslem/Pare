"""Re-derive the numbers and comparisons in README.md from results/*.json.

Every table cell and every comparative statement in the Results section of the
README is recomputed here from the results files. The script exits non-zero
if the README disagrees with them.

Run from the repository root:

    python scripts/check_readme_numbers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()
RESULTS = ROOT / "results"

ev = json.load(open(RESULTS / "eval_results.json"))
lm = json.load(open(RESULTS / "lm_eval_results.json"))
p6 = json.load(open(RESULTS / "phase6_results.json"))
sqc = json.load(open(RESULTS / "smoothquant_calibration.json"))
kv7 = json.load(open(RESULTS / "kv_cache_7b.json"))
kv05 = json.load(open(RESULTS / "kv_cache_05b.json"))

MODELS = ["llama31", "qwen25", "olmo3"]
INT_METHODS = ["rtn-int8", "gptq-int4", "awq-int4"]
P6_MODELS = ["llama2", "llama3", "qwen"]
P6_METHODS = ["rtn-int8", "gptq-int4", "awq-int4", "smoothquant"]
ZERO_SHOT = ["lambada_openai", "piqa", "winogrande", "openbookqa", "rte", "copa"]

failures: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        failures.append(msg)


def has(text: str) -> None:
    check(text in README, f"README is missing: {text!r}")


def delta(x: float) -> str:
    """Format a change as +0.19, −0.22 or 0.00, as the README prints it."""
    r = round(x, 2)
    if r == 0:
        return "0.00"
    return ("+" if r > 0 else "−") + f"{abs(r):.2f}"


def cell(value: float, base: float) -> str:
    """A table cell: the value, then its change from the base.

    The change is the difference of the two rounded values, so a reader can
    verify it from the table itself.
    """
    return f"{value:.2f} ({delta(round(value, 2) - round(base, 2))})"


def ppl(m: str, k: str) -> float:
    return ev[f"{m}/{k}"]["wikitext2_ppl"]


def p6ppl(m: str, k: str) -> float:
    return p6[f"{m}/{k}"]["wikitext2_ppl"]


def acc(m: str, k: str) -> float:
    scores = lm[f"{m}/{k}"]
    assert set(scores) == set(ZERO_SHOT), f"{m}/{k} has tasks {sorted(scores)}"
    return sum(scores.values()) / len(scores)


def tps(m: str, k: str) -> float:
    thr = ev[f"{m}/{k}"]["throughput"]
    return next(t for t in thr if t["batch_size"] == 1)["tokens_per_sec"]


# --- Perplexity table 1 -------------------------------------------------------
for m in MODELS:
    base = ppl(m, "fp16")
    has(f"| {base:.2f} |" if m != "olmo3" else f"| {base:.2f} |")
    for k in INT_METHODS:
        has(cell(ppl(m, k), base))
ao = ppl("llama31", "gptq-int4-act_order")
has(cell(ao, ppl("llama31", "fp16")))
has(f"lowers perplexity from {ppl('llama31', 'gptq-int4'):.2f} to {ao:.2f}")

# --- Perplexity table 2 (phase 6 sweep) --------------------------------------
for m in P6_MODELS:
    base = p6ppl(m, "fp16")
    has(f"| {base:.2f} |")
    for k in P6_METHODS:
        has(cell(p6ppl(m, k), base))
check(
    p6ppl("qwen", "fp16") == ppl("qwen25", "fp16")
    and all(p6ppl("qwen", k) == ppl("qwen25", k) for k in INT_METHODS),
    "claim: Qwen2.5-7B rows in the two perplexity tables are the same runs",
)

# --- Perplexity prose ---------------------------------------------------------
rtn_deltas = [abs(ppl(m, "rtn-int8") - ppl(m, "fp16")) for m in MODELS] + [
    abs(p6ppl(m, "rtn-int8") - p6ppl(m, "fp16")) for m in P6_MODELS
]
check(max(rtn_deltas) <= 0.01, "claim: RTN INT8 within 0.01 perplexity of FP16 on all models")

sq_deltas = sorted(p6ppl(m, "smoothquant") - p6ppl(m, "fp16") for m in P6_MODELS)
check(0.095 <= sq_deltas[0] and sq_deltas[-1] < 0.115, "claim: SmoothQuant adds 0.10 to 0.11")
has("SmoothQuant W8A8 adds 0.10 to 0.11")

check(ppl("qwen25", "gptq-int4") < ppl("qwen25", "awq-int4"), "claim: GPTQ lower PPL on Qwen2.5-7B")
check(ppl("olmo3", "gptq-int4") < ppl("olmo3", "awq-int4"), "claim: GPTQ lower PPL on OLMo-3-7B")
check(p6ppl("llama2", "awq-int4") < p6ppl("llama2", "gptq-int4"), "claim: AWQ lower PPL on Llama-2")
check(p6ppl("llama3", "awq-int4") < p6ppl("llama3", "gptq-int4"),
      "claim: AWQ lower PPL on Llama-3-8B",
)
check(ppl("llama31", "awq-int4") < ppl("llama31", "gptq-int4"),
      "claim: AWQ lower PPL on Llama-3.1-8B",
)

d_l3 = p6ppl("llama3", "gptq-int4") - p6ppl("llama3", "fp16")
d_l31 = ppl("llama31", "gptq-int4") - ppl("llama31", "fp16")
has(f"GPTQ adds {d_l3:.2f} perplexity on Llama-3-8B and {d_l31:.2f} on Llama-3.1-8B")
other = [ppl(m, "gptq-int4") - ppl(m, "fp16") for m in ["qwen25", "olmo3"]] + [
    p6ppl("llama2", "gptq-int4") - p6ppl("llama2", "fp16")
]
has(f"against {min(other):.2f} to {max(other):.2f} on the other models")
has(f"the Llama-3.1-8B increase falls to {ao - ppl('llama31', 'fp16'):.2f}")

# --- Accuracy table and prose -------------------------------------------------
for m in MODELS:
    base = acc(m, "fp16")
    has(f"| {base:.2f} |")
    for k in INT_METHODS:
        has(cell(acc(m, k), base))
check(acc("qwen25", "awq-int4") > acc("qwen25", "gptq-int4"), "claim: AWQ higher acc on Qwen2.5-7B")
check(acc("olmo3", "awq-int4") > acc("olmo3", "gptq-int4"),
      "claim: AWQ higher accuracy on OLMo-3-7B",
)
check(acc("llama31", "gptq-int4") > acc("llama31", "awq-int4"),
      "claim: GPTQ higher acc on Llama-3.1-8B",
)
# One COPA example is 1 point on that task, so 1/6 of a point on the mean.
has(f"up to {1 / 6:.2f} points")

# --- Throughput table and prose -----------------------------------------------
for m in MODELS:
    for k in ["fp16"] + INT_METHODS:
        has(f"| {tps(m, k):.1f} |" if k == "fp16" else f" {tps(m, k):.1f} |")
ratios = [tps(m, "fp16") / tps(m, k) for m in MODELS for k in INT_METHODS]
check(all(r > 1 for r in ratios), "claim: reference path slower than FP16 for every quantized row")
has(f"{min(ratios):.0f} to {max(ratios):.0f} times slower than FP16")

# --- SmoothQuant calibration modes --------------------------------------------
has(
    f"absmax gave {sqc['smoothquant_absmax']:.2f} perplexity, percentile "
    f"{sqc['smoothquant_percentile']:.2f} and MSE {sqc['smoothquant_mse']:.2f}, "
    f"against {sqc['fp16']:.2f} for FP16"
)

# --- KV cache -----------------------------------------------------------------
for kv in (kv7, kv05):
    fp, ki = kv["results"]["fp16"]["ppl"], kv["results"]["kivi"]["ppl"]
    has(f"| {fp:.2f} | {cell(ki, fp)} | {kv['results']['kivi']['attn_rel_err'] * 100:.1f}% |")
    cfg = kv["config"]
    check(cfg["bits"] == 4 and cfg["group_size"] == 32, "KV config is 4-bit, G=32")

# --- Style ---------------------------------------------------------------------
check("—" not in README, "README contains an em dash")

if failures:
    print("FAIL")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("OK: every number and comparison in README.md matches results/")
