# Pare

Pare is a Python library for post-training quantization of Hugging Face causal language models. It implements RTN, GPTQ, AWQ and SmoothQuant, plus NF4 and FP8 weight formats, behind one `quantize()` call, and the scheme is a field of `QuantConfig`. The same code quantizes Llama, Qwen and OLMo models, and the results below were produced with the benchmark scripts in this repository.

Pare provides the following.

- One API and one configuration object for RTN, GPTQ, AWQ and SmoothQuant, with INT4, INT8, NF4 and FP8 weights.
- A layerwise calibration path for GPTQ, AWQ and SmoothQuant that processes one transformer block at a time.
- Save and load of quantized models in safetensors, and export of INT4 models to the AWQ checkpoint format that vLLM loads.
- Sensitivity-based mixed precision, a KIVI-style quantized KV cache, and an optional Triton INT4 kernel.
- Evaluation helpers for WikiText-2 and C4 perplexity, LAMBADA accuracy and decode throughput, and a command-line interface.

The tables in this README compare Pare's schemes with each other on three models. They do not compare Pare with other quantization libraries.

## Installation

```bash
pip install pare-quant                 # core: torch, safetensors, numpy
pip install "pare-quant[all]"          # adds transformers, datasets and triton
```

Pare needs Python 3.11 or later and PyTorch 2.1 or later. The test suite runs on PyTorch 2.1.2 and on the current release.

## Quickstart

Quantize Qwen2.5-7B to 4-bit AWQ with 128 calibration sequences from WikiText-2, then save it.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from pare import QuantConfig, quantize, save_quantized
from pare.calibration.data import load_wikitext2_calibration

model_id = "Qwen/Qwen2.5-7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16)

calib = load_wikitext2_calibration(tokenizer, n_samples=128, seq_len=2048)
# Any list of token tensors works, each shaped [seq_len] or [1, seq_len].

config = QuantConfig(bits=4, scheme="awq", group_size=128)
quantize(model, config, calibration_data=calib, device="cuda")
save_quantized(model, "qwen25-awq-int4")
```

Load it again into a model built from the architecture alone.

```python
from transformers import AutoConfig, AutoModelForCausalLM
from pare import load_quantized

model = AutoModelForCausalLM.from_config(
    AutoConfig.from_pretrained(model_id), torch_dtype=torch.float16
)
load_quantized(model, "qwen25-awq-int4")
```

The same two steps are available from the command line.

```bash
python -m pare quantize --model Qwen/Qwen2.5-7B --scheme awq --bits 4 --output qwen25-awq-int4
python -m pare eval --model Qwen/Qwen2.5-7B --quantized qwen25-awq-int4 --dataset wikitext2
```

RTN needs no calibration data. GPTQ, AWQ and SmoothQuant need it and raise a `ValueError` without it.

## Methods

| `scheme=` | Calibration | Bits | Weights | Activations |
|-----------|-------------|------|---------|-------------|
| `"rtn"` | No | 4 or 8 | Round to nearest, per group or per channel | FP16 |
| `"gptq"` | Yes | 4 or 8 | Hessian-based error compensation, column by column | FP16 |
| `"awq"` (default) | Yes | 4 or 8 | Per-channel scales searched on activation statistics, then RTN | FP16 |
| `"smoothquant"` | Yes | 8 | Per-channel INT8 after migrating activation outliers into the weights | INT8, simulated |

All INT4 results below use `group_size=128`, the value used in the GPTQ and AWQ papers. `QuantConfig` also accepts 2 and 3 bits; these are stored unpacked in int8 and are not benchmarked.

AWQ and SmoothQuant fuse their per-channel scales into the layer that feeds each linear layer. This needs a Llama-style block with `input_layernorm` before the attention projections. On blocks without one, such as OLMo-3, both schemes skip the fusion with a warning; AWQ then quantizes the block with RTN at the same bit-width, and SmoothQuant quantizes it to INT8 without smoothing. The OLMo-3 "AWQ INT4" rows below are therefore RTN INT4 with `group_size=128`.

SmoothQuant quantizes activations per token to INT8 and dequantizes them again before the matmul. This measures the accuracy of INT8 weights and activations; it does not run an INT8 matmul and gives no speed gain over the FP16 path.

### Options

`act_order=True` makes GPTQ quantize columns in order of decreasing Hessian diagonal. On Llama-3.1-8B it lowers perplexity from 11.10 to 6.54 (see the results).

```python
QuantConfig(bits=4, scheme="gptq", group_size=128, act_order=True)
```

`calibration=` selects how SmoothQuant estimates activation ranges. The options are `"absmax"` (default), `"percentile"` with `calibration_percentile`, and `"mse"`. On Qwen2.5-7B, evaluated on 64 sequences of 2048 tokens, absmax gave 6.72 perplexity, percentile 6.74 and MSE 6.76, against 6.63 for FP16 (`results/smoothquant_calibration.json`).

`sensitive_bits=` promotes layers whose RTN reconstruction error, weighted by calibration activations, exceeds `sensitivity_threshold`.

```python
QuantConfig(bits=4, scheme="awq", sensitive_bits=8, sensitivity_threshold=0.05)
```

NF4 uses the 4-bit normal-float codebook from QLoRA with one absmax scale per output row. FP8 stores weights as `torch.float8_e4m3fn` or `torch.float8_e5m2` with one scale per row and dequantizes them to FP16 for the matmul, so it runs on any device PyTorch supports. Neither format is benchmarked below.

```python
from pare import QuantDtype
QuantConfig(bits=4, dtype=QuantDtype.NF4, scheme="rtn")
QuantConfig(bits=8, dtype=QuantDtype.FP8_E4M3, scheme="rtn")
```

`exclude=` lists module-name substrings to leave in FP16; the default excludes `lm_head`.

## Results

All numbers come from `results/*.json`, and `python scripts/check_readme_numbers.py` re-derives every number and comparison in this section from those files. Runs used one NVIDIA A40 (46 GB) with the environment pinned in `requirements-gpu.txt`. Calibration uses the first 128 windows of 2048 tokens from the WikiText-2 training split. Perplexity is measured on the full WikiText-2 test split in windows of 2048 tokens with no overlap (`benchmarks/run_all.py`). Zero-shot accuracy is the mean over LAMBADA, PIQA, WinoGrande, OpenBookQA, RTE and COPA with lm-evaluation-harness 0.4.12, using `acc_norm` where a task reports it and `acc` otherwise (`benchmarks/run_lm_eval.py`). INT4 rows use `group_size=128`; INT8 rows are per channel. Changes in parentheses are differences of the rounded values shown.

### Perplexity

WikiText-2 perplexity (lower is better), with the change against FP16 in parentheses.

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|--------------|------------|-----------|
| FP16 | 6.24 | 6.85 | 9.92 |
| RTN INT8 | 6.25 (+0.01) | 6.85 (0.00) | 9.92 (0.00) |
| GPTQ INT4 | 11.10 (+4.86) | 7.04 (+0.19) | 10.21 (+0.29) |
| GPTQ INT4, `act_order=True` | 6.54 (+0.30) | | |
| AWQ INT4 | 6.77 (+0.53) | 7.13 (+0.28) | 10.36 (+0.44), RTN fallback |

Models are `meta-llama/Llama-3.1-8B`, `Qwen/Qwen2.5-7B` and `allenai/Olmo-3-1025-7B`. The `act_order` row was run for Llama-3.1-8B only.

An earlier sweep (`results/phase6_results.json`) used the same protocol on Llama-2-7B, Llama-3-8B and Qwen2.5-7B and included SmoothQuant, with INT8 per-channel weights and activations and `smooth_alpha=0.5`.

| Method | Llama-2-7B | Llama-3-8B | Qwen2.5-7B |
|--------|------------|------------|------------|
| FP16 | 5.47 | 6.14 | 6.85 |
| RTN INT8 | 5.48 (+0.01) | 6.14 (0.00) | 6.85 (0.00) |
| GPTQ INT4 | 5.74 (+0.27) | 8.75 (+2.61) | 7.04 (+0.19) |
| AWQ INT4 | 5.66 (+0.19) | 6.67 (+0.53) | 7.13 (+0.28) |
| SmoothQuant INT8 W8A8 | 5.58 (+0.11) | 6.25 (+0.11) | 6.96 (+0.11) |

Models are `meta-llama/Llama-2-7b-hf`, `meta-llama/Meta-Llama-3-8B` and `Qwen/Qwen2.5-7B`. The Qwen2.5-7B rows are the same runs as in the first table.

RTN INT8 stays within 0.01 perplexity of FP16 on all five models, and SmoothQuant W8A8 adds 0.10 to 0.11 on the three it was run on. Between the INT4 methods, GPTQ has the lower perplexity on Qwen2.5-7B and OLMo-3-7B, while AWQ has the lower perplexity on Llama-2-7B, Llama-3-8B and Llama-3.1-8B. Without `act_order`, GPTQ adds 2.61 perplexity on Llama-3-8B and 4.86 on Llama-3.1-8B, against 0.19 to 0.29 on the other models; with `act_order=True` the Llama-3.1-8B increase falls to 0.30. We have not established why the Llama-3 models depend on column ordering this much, and a comparison with the reference GPTQ implementation on these models is still to be done.

### Zero-shot accuracy

Mean accuracy over the six tasks (higher is better), with the change against FP16 in parentheses.

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|--------------|------------|-----------|
| FP16 | 73.22 | 74.13 | 69.57 |
| RTN INT8 | 73.00 (−0.22) | 74.09 (−0.04) | 69.57 (0.00) |
| GPTQ INT4 | 71.65 (−1.57) | 73.39 (−0.74) | 69.42 (−0.15) |
| AWQ INT4 | 70.69 (−2.53) | 73.93 (−0.20) | 69.57 (0.00), RTN fallback |

The ranking of the INT4 methods by accuracy differs from the ranking by perplexity. AWQ has the higher accuracy on Qwen2.5-7B and OLMo-3-7B, where GPTQ had the lower perplexity, and GPTQ has the higher accuracy on Llama-3.1-8B. COPA has 100 test examples and RTE 277, so a single example moves the six-task mean by up to 0.17 points, and differences of that size should be read as ties. Per-task scores are in `results/lm_eval_results.json`.

### Throughput of the reference path

Decode throughput at batch size 1 in tokens per second, generating 128 tokens after a 256-token prompt with `model.generate`.

| Method | Llama-3.1-8B | Qwen2.5-7B | OLMo-3-7B |
|--------|--------------|------------|-----------|
| FP16 | 25.8 | 32.4 | 25.1 |
| RTN INT8 | 2.1 | 2.3 | 2.3 |
| GPTQ INT4 | 1.1 | 1.2 | 1.2 |
| AWQ INT4 | 1.1 | 1.2 | 1.2 |

These rows use `QuantizedLinear`'s default forward pass, which dequantizes the weights to FP16 on every call. It is a reference for checking accuracy, and it is 11 to 27 times slower than FP16 here. Quantized inference at speed needs a fused kernel. Pare offers two routes.

- `export_awq` writes a 4-bit model as an AWQ checkpoint (`qweight`, `qzeros`, `scales` and an `awq` `quantization_config`) that vLLM loads and serves with its own INT4 kernels. The repack is lossless, since Pare stores INT4 weights in the same affine form as AWQ. Serving measurements will be added once they are rerun with the current code.
- `QuantizedLinear.use_kernel = True` dispatches INT4 layers to the Triton kernel in `pare.kernels.matmul_int4`, which dequantizes inside the matmul. `benchmarks/bench_triton_kernel.py` measures it against the reference path; its results are not yet saved to `results/` and are therefore not reported here.

```python
from pare.model.export_awq import export_awq
export_awq(model, "qwen25-awq-int4-vllm", tokenizer=tokenizer)   # 4-bit models only
```

### KV cache

`pare.kv_cache.QuantizedKVCache` implements the KIVI scheme. Keys are quantized per channel and values per token, in groups of 32 tokens, and the most recent 128 tokens are kept in FP16. It is a standalone cache object that returns full-precision keys and values for attention, not a drop-in cache for `generate`. On WikiText-2 test text with 4-bit storage (`results/kv_cache_7b.json`, `results/kv_cache_05b.json`):

| Model | Evaluation | FP16 PPL | KIVI 4-bit PPL | Attention output relative error |
|-------|------------|----------|----------------|---------------------------------|
| Qwen2.5-7B | 64 sequences of 2048 tokens | 6.63 | 6.66 (+0.03) | 1.4% |
| Qwen2.5-0.5B | 64 sequences of 512 tokens | 17.02 | 17.20 (+0.18) | 1.3% |

The cache stores 4-bit values in uint8 without packing, so its memory saving over FP16 is 2 times rather than 4.

```python
from pare import KVCacheConfig, QuantizedKVCache

cache = QuantizedKVCache(num_layers=28, config=KVCacheConfig(bits=4, group_size=32, residual_length=128))
k, v = cache.update(layer_idx, new_k, new_v)   # full-precision tensors for attention
```

## Hardware

The layerwise calibration path loads the FP16 model on the GPU, then quantizes one transformer block at a time while holding that block's calibration activations. Quantizing a 7B model therefore needs the FP16 model (about 15 GB) plus the activations of 128 sequences of 2048 tokens for one block. All runs reported here used a 46 GB A40. RTN, NF4 and FP8 run on CPU as well; the Triton kernel needs a CUDA GPU and `triton>=3.0`.

## Development

```bash
pip install -e ".[dev,all]"
pytest                                # CPU unit tests, no downloads
pytest --smoke                        # GPT-2 and Qwen2.5-0.5B end to end, needs internet
pytest --gpu                          # Triton kernel tests, needs CUDA
python scripts/check_readme_numbers.py   # re-derive the numbers in this README
```

GitHub Actions runs the CPU unit tests on Python 3.11 and 3.12 for every push and pull request.

## References

- **GPTQ**, Frantar, Ashkboos, Hoefler, Alistarh. *GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers.* ICLR 2023. [arXiv:2210.17323](https://arxiv.org/abs/2210.17323)
- **AWQ**, Lin et al. *AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration.* MLSys 2024. [arXiv:2306.00978](https://arxiv.org/abs/2306.00978)
- **SmoothQuant**, Xiao, Lin, Seznec, Wu, Demouth, Han. *SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.* ICML 2023. [arXiv:2211.10438](https://arxiv.org/abs/2211.10438)
- **KIVI**, Liu, Yuan, Jin, Zhong, Xu, Braverman, Chen, Hu. *KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache.* ICML 2024. [arXiv:2402.02750](https://arxiv.org/abs/2402.02750)
- **NF4 / QLoRA**, Dettmers, Pagnoni, Holtzman, Zettlemoyer. *QLoRA: Efficient Finetuning of Quantized LLMs.* NeurIPS 2023. [arXiv:2305.14314](https://arxiv.org/abs/2305.14314)

The GPTQ loop is adapted from IST-DASLab/gptq (Apache-2.0) and the AWQ scale search and fusion from mit-han-lab/llm-awq (MIT); see `LICENSE`.

Other resources:

- Vivek Kalyanarangan. *Quantization and Fast Inference: A Practitioner's Guide to Efficient AI.* Manning, May 2026.

## Citation

```bibtex
@misc{moslem2026pare,
  author = {Moslem, Yasmin},
  title  = {Pare: A {Python} library for post-training quantization of large language models},
  year   = {2026},
  url    = {https://github.com/ymoslem/Pare},
}
```
