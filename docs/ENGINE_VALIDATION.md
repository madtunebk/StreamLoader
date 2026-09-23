# ENGINE_VALIDATION

Companion to `VALIDATION.md` (which covers the CPU-only loader). This
covers only the GPU engine (`engine/`) and its integration into a real
diffusers pipeline (`inference/generate_flux.py`).

## Environment

- Same machine/toolchain as `VALIDATION.md`: rustc/cargo 1.97.1, Linux
  6.8.0-137-generic.
- GPU: 2x NVIDIA GeForce RTX 3060, 12288 MiB each, driver 580.173.02.
  Only GPU 0 used (single-GPU scope, see `ENGINE.md`).
- Python 3.12.3, `torch 2.14.0+cu130`, `diffusers 0.40.0`,
  `transformers 5.16.1`, `accelerate 1.14.0`, `cudarc 0.19.9`,
  `pyo3 0.27.2`, `maturin 1.15.0`.
- Model: `black-forest-labs/FLUX.2-klein-9B`, already present in the
  local Hugging Face cache — not downloaded by this work. Full pipeline:
  transformer (18.16GB, the Rust-managed component), Qwen3-based text
  encoder (~15.4GB), AutoencoderKLFlux2 VAE.

## What is proven, and by what

| Claim | Evidence |
|---|---|
| Cross-stream CUDA event handoff (Rust's transfer stream → PyTorch's compute stream) is correct with zero CPU-side synchronization | `inference/tests/test_dlpack.py`: 6 trials, different fill values, real DLPack tensor read back via `torch.Tensor.sum()` with no `torch.cuda.synchronize()` call anywhere in the test. All 6 correct. |
| The engine reads the real checkpoint correctly through the whole pipeline (mmap → pinned → VRAM → DLPack → `torch.Tensor`) | `inference/tests/test_engine_real_model.py` against the real 18.16GB model: all 16 tensors of `transformer_blocks.7` (872,416,256 bytes) byte-identical to an independent, from-scratch Python parse of the actual shard file (not reusing any Rust or `safetensors`-package code). |
| Buffer reuse (2 VRAM slots, 32 blocks) does not corrupt data | Same test: forced a genuine eviction (fetched 2 more distinct blocks to cycle both ring slots), asserted `transfer_count` actually incremented (i.e. the eviction was real, not accidentally still a cache hit), then re-verified all 16 tensors byte-identical to the first fetch. |
| The full pipeline, with the transformer entirely served by the Rust engine, produces correct output | Pixel diff against the plain-diffusers baseline (`inference/generate.py`, same prompt/seed/resolution/steps/dtype): **0 pixels differ, out of 1,048,576** (1024×1024, RGB). See below. |
| No silent fallback on engine failure | Observed directly during development: pointing the engine at the real model *without* `trust_root` raises `RuntimeError` (the loader's symlink-escape rejection, propagated through `EngineError`/`PyErr`) and the script exits nonzero — never fell through to any other loading path, because there isn't one in `generate_flux.py`. |
| A real bug this testing caught (not a hypothetical) | `RustEngine::new()` originally let the pinned host allocation's owner (`PinnedHostSlice`, a local variable) drop at the end of the function, freeing the buffer immediately. The shared-block transfer (which happens *during* `new()`) worked; the first real per-block fetch afterward read through the now-dangling pointer and segfaulted. Caught by `test_engine_real_model.py`, fixed by storing the `PinnedHostSlice` in the engine struct. |

## Benchmark: baseline vs. Rust-engine, identical inputs

Both runs: same checkpoint (`FLUX.2-klein-9B`), same prompt
(`"a bottle on a table with the milky way galaxy swirling inside it"`),
same seed (`0`), same resolution (`1024x1024`), same dtype (`bfloat16`),
same **50** inference steps (the pipeline's own default — used here
specifically so the comparison is apples-to-apples against the baseline
run recorded earlier; see the note on step count below), same attention
backend (whatever diffusers/torch selects by default — not overridden in
either run).

| | Baseline (`generate.py`, `enable_sequential_cpu_offload()`) | Rust engine (`generate_flux.py`) |
|---|---|---|
| Load time | not separately measured in the baseline run | 3.82s (pipeline components + engine init + hook attachment) |
| Generation time (50 steps + VAE decode) | 278.1s | **195.43s** |
| Output | `inference/output.png` | `inference/output_rust.png` |
| Pixel diff | — | **0 / 1,048,576 pixels differ** (max abs diff 0, mean abs diff 0.0) |

**195.43s vs 278.1s is a real, measured ~30% reduction** — but the honest
explanation, from the engine's own counters on that run, is not "the
weights were cached":

```
bytes transferred H2D:   872.416 GB
transfer count:          1600        (= 50 steps x 32 blocks)
cache hits:              1551        (prefetch-then-immediate-get_block pairs)
pinned host bytes:       18.157 GB
VRAM bytes (engine):     2.454 GB    (2 ring slots + 1 shared slot)
```

`transfer_count == 1600` means **every non-shared block was re-transferred
on every single step** — only the ~0.7GB of `shared/unassigned` tensors
stayed VRAM-resident across the whole run, because 2 buffers sized to the
largest block (~2.45GB total VRAM used by the engine) cannot hold a
32-block, 18GB transformer at once on a 12GB card. `872.416GB / 50 steps
≈ 17.45GB/step`, matching `18.16GB total - ~0.7GB shared` almost exactly.
This workload is **PCIe-bandwidth-bound**: `872.416GB / 195.43s ≈
4.5GB/s` sustained effective H2D throughput for the whole run. The
speedup over the baseline comes from **overlapping** that unavoidable
transfer with compute via `prefetch()` (the transfer for block N+1 starts
while block N is still computing), not from avoiding the transfer —
because at this VRAM budget, the transfer cannot be avoided. This is
exactly the "don't assume Rust is automatically faster" case: the
language wrote fewer instructions per transfer than Python's own
offload path, but the win came from a specific overlap mechanism, and
the underlying bottleneck (moving ~872GB over PCIe for 50 steps of an
18GB model that doesn't fit in 12GB VRAM) is identical physics for any
implementation using this VRAM budget. A GPU with enough VRAM to hold
the whole transformer resident would not pay this cost at all, in either
language.

### Note on step count

The comparison above intentionally used 50 steps to match the
already-recorded baseline run exactly. Separately, `black-forest-labs/FLUX.2-klein-9B`'s
own `config.json` (`is_distilled: true`, `guidance_embeds: false`) is a
step-distilled model; ~18-20 steps is the useful range for it, and
`generate_flux.py` now defaults to `--steps 20` rather than the
pipeline's 50. A second demo run at `--steps 50` (a different, "crazy"
prompt, not used for the correctness/speed comparison above) measured
**186.22s** generation time with the same per-block-transfer pattern
(`transfer_count=1600`, `bytes_h2d=872.416GB`) — consistent with the
first run, and not directly comparable to the 20-step default since it
used a different prompt and full 50 steps. No 20-step baseline run
(plain `generate.py`, no Rust) exists to compare against; the 50-step
comparison above is the only apples-to-apples measurement taken.

## What is fixture-tested / real-model-tested / not tested (engine)

**Real-hardware-tested**: everything in the table above. There is no
mock GPU path and no simulated CUDA in this engine — every claim here
required the actual RTX 3060 and the actual downloaded model.

**Not tested**:
- Dual-GPU operation. Out of scope by explicit design (`ENGINE.md`).
- Behavior with a VRAM budget large enough to hold the whole transformer
  resident (would require a bigger GPU than what's available here) — the
  "kept resident across forward passes when budget allows" path for
  *non-shared* blocks was therefore never actually exercised in a
  favorable-budget scenario, only the current, budget-constrained one.
- `mark_block_done` correctness under a *misused* calling discipline
  (e.g. prefetching more than one block ahead with only 2 ring slots,
  which the current protocol does not defend against — see the
  "sync contract" note in `engine/src/engine.rs`'s `issue_transfer`).
  The demonstrated usage (`generate_flux.py`) always prefetches exactly
  one block ahead and marks each block done immediately after its
  forward returns, which is the pattern the 2-slot ring is correct for.
- Any dtype other than BF16 (the only dtype this real model actually
  uses) going through the DLPack path — `F32`/`F16`/`I64`/etc. dtype
  mapping in `dlpack.rs` is implemented and each one's code/bits values
  were checked against the DLPack spec by hand, but none of them were
  exercised against a real tensor of that dtype.
- Cargo-level unit tests for the engine crate: none exist (`cargo test`
  in `engine/` runs 0 tests). This is a deliberate, stated limitation,
  not an oversight — the engine's correctness properties (real CUDA
  context/stream/event behavior, real PyTorch interop) aren't
  meaningfully testable without a GPU and a Python/torch environment, so
  they're tested via the Python scripts above instead of `cargo test`.

## Int8 quantization experiment — real findings, not delivered

At the user's request, tried streaming an int8-quantized version of the
transformer through the same engine, to isolate whether halved transfer
volume (not just BF16) moves generation time. `inference/experimental/quantize_transformer.py`
(kept in the repo; its ~9GB output directory was deleted, not delivered)
row-wise-quantizes every 2D weight tensor via
`bitsandbytes.functional.int8_vectorwise_quant` — exactly what
`Linear8bitLt` does internally — producing a real, standard safetensors
checkpoint (int8 `CB` weight + fp32 `SCB` per-channel scale, 9.09GB, 50.0%
of the original 18.16GB, confirmed via file size after quantizing).

**Confirmed correct, unchanged code required:** the loader/engine indexed
and streamed the quantized files with **zero changes** to `streamloader`
or `engine` — `I8`/`F32` dtype handling was already in place. Spot-checked
byte-exact against the quantized file (`torch.equal` true) via the engine,
same rigor as the BF16 correctness tests above.

**Confirmed real speedup when it ran:** 20 steps, same prompt/seed,
through the *same* streaming engine: 2.17s/it (int8) vs 3.52s/it (BF16,
freshly re-measured for a fair same-step-count comparison:
`generation_time=73.91s`, `bytes_h2d=348.967GB` for 20-step BF16, vs
`generation_time=50.12s`, `bytes_h2d=174.620GB` for 20-step int8 — H2D
traffic exactly halved as expected, wall time dropped too, both before
the correctness bug below was found).

**Not correct — a real bug, not shipped:** the resulting images were
pure noise. Block-by-block tracing (`transformer_blocks.0` through `.5`,
one real forward pass) showed hidden-state magnitudes blowing up from
the first block onward (`max` values in the 10,000+ range, where a
correctly-normalized block output should be O(1)-O(10)), plus a dtype
warning (`input dtype = float, weight dtype = bfloat16` reaching an
`RMSNorm`) and an eventual crash in VAE decode on a float/bfloat16
mismatch. Isolated unit tests of the exact same `Int8Params`/`Linear8bitLt`
wiring pattern — 2D input, 3D input (matching real `[batch, seq, hidden]`
activations), same-module weight reassignment across multiple calls,
same-VRAM-address buffer reuse (matching the ring buffer exactly) — all
passed cleanly (quantization-level error only, no blowup). Raising the
LLM.int8() outlier threshold from `0.0` to the paper's standard `6.0`
did not fix it. Working hypothesis: `bnb.matmul`'s internal
bf16→fp16 cast and/or its output dtype handling doesn't compose cleanly
with this specific model's bf16-throughout attention/RoPE path — an edge
case outside bitsandbytes' primary fp16-LLM design target, not pinned
down further given time constraints and the user's explicit call to stop
here rather than chase it further.

**Net honest conclusion:** quantization-driven transfer/size reduction is
real and correctly delivered through this engine unmodified; a *working*
int8 compute path for this specific model was not achieved and is not
part of this delivery. `quantize_transformer.py` and
`inference/experimental/generate_rust_int8.py` are kept as a documented, reproducible
starting point for anyone who wants to pick this up, not as a working
feature.
