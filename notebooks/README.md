# Pare notebooks

Eight notebooks build quantization methods from their primitives. The first seven cover the schemes in Pare: each derives a method, implements it in a few dozen lines, runs an experiment that isolates the mechanism, checks the implementation against the corresponding function in Pare, and ends with the method applied to a small model and exercises. The eighth is a study of TurboQuant, a KV-cache quantization method that Pare does not implement, checked against the numbers stated in its paper. The code in the notebooks uses the same conventions as the library, so a reader who finishes them can read `pare/core` and `pare/schemes` without translating.

| Notebook | Covers | Checked against | Model in the last section |
|---|---|---|---|
| `01_quantization_basics.ipynb` | Scale, zero-point, the round trip, granularity, the 6 dB per bit rule | `pare.core.scale`, `pare.core.functional` | none |
| `02_rtn.ipynb` | Round-to-nearest, weight against output error, INT4 packing, memory accounting | `pare.core.pack`, `QuantizedLinear.from_linear` | GPT-2 |
| `03_gptq.ipynb` | The reconstruction objective, error compensation through the Cholesky factor, damping, column order, calibration size | `pare.schemes.gptq._gptq_one_layer` | GPT-2 |
| `04_awq.ipynb` | Salient channels, what one scale does to a group, the scale search, where the inverse scale goes, RTN against GPTQ against AWQ on one model | `pare.schemes.awq._search_scale` | Qwen2.5-0.5B |
| `05_smoothquant.ipynb` | Activation outliers, the smoothing factor, the migration strength, a W8A8 layer | `pare.schemes.smoothquant._smooth_factor`, `QuantizedLinear` with `quantize_inputs` | Qwen2.5-0.5B |
| `06_nf4.ipynb` | Normality of trained weights, the NF4 codebook, grid against granularity | `pare.core.functional.quantize_nf4` | GPT-2 |
| `07_fp8.ipynb` | The FP8 grids, scaling before the cast, E4M3 against E5M2, weight-only FP8 | `pare.core.functional.quantize_fp8` | GPT-2 |
| `08_turboquant.ipynb` | Random rotation, Lloyd-Max levels, unbiased inner products from a one-bit residual (TurboQuant, arXiv:2504.19874) | the paper's stated distortions and constants | none |

## Running them

```bash
pip install -e ".[dev,all]"       # from the repository root; installs jupyter, transformers and datasets
jupyter lab notebooks/
```

Every notebook runs on a CPU. The first code cell adds the repository root to `sys.path`, so the notebooks use the checkout they live in; with `pip install pare-quant` that line is harmless. Approximate run times on a laptop CPU, after the downloads:

| Notebook | Downloads | Run time |
|---|---|---|
| 01, 08 | none | seconds |
| 02, 03 | GPT-2 (about 500 MB) and WikiText-2 | 1 to 2 minutes each |
| 04, 05 | Qwen2.5-0.5B (about 1 GB) and WikiText-2 | 2 to 3 minutes for 04, 1 to 2 for 05 |
| 06, 07 | GPT-2 and WikiText-2 | about a minute together |

Notebooks 04 and 05 use Qwen2.5-0.5B because AWQ and SmoothQuant fold their scales into a Llama-style block (`input_layernorm`, `q_proj`, `gate_proj`, ...); on GPT-2, which has no such block, Pare falls back to RTN for both schemes. The perplexities in the notebooks use 20 windows of 512 tokens, enough to rank the methods on these models; the README of the repository reports the full WikiText-2 test set on 7B models.

The notebooks are committed without outputs. GitHub Actions executes notebooks 01, 02, 03 and 08 on every push and pull request, so their cells are checked against the current code.

## Which schemes need which hardware

The arithmetic in every scheme is plain PyTorch and runs on a CPU for small models. What differs is what makes each scheme practical at 7B parameters and above.

| Scheme or feature | Runs on CPU | Hardware for 7B models | Notes |
|---|---|---|---|
| RTN, NF4, FP8 weights | yes | any CUDA GPU | No calibration pass |
| GPTQ, AWQ, SmoothQuant | yes, slowly | any CUDA GPU | The layerwise calibration loads the FP16 model on the device |
| Triton INT4 kernel (`use_kernel=True`) | no | CUDA GPU with `triton>=3.0` | Raises if the input is not on CUDA |
| SmoothQuant INT8 matmul | not implemented | | Pare simulates INT8 activations; the matmul runs in floating point |
| FP8 matmul | not implemented | | Pare's FP8 is weight-only; FP8 tensor cores (Hopper, Ada and later) are not used |
