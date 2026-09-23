# `inference/`

Real diffusers inference where the transformer's weights are served by
the Rust `streamloader_engine` (see `../docs/ENGINE.md`) instead of
diffusers' own loader. Requires an NVIDIA GPU (tested on a single RTX
3060 12GB) — there is no CPU fallback anywhere in this directory.

## Layout

- `generate_flux.py` — FLUX.2-klein-9B
- `generate_qwen.py` — Qwen-Image-2.1
- `tests/` — byte-exact correctness tests against the real checkpoint
- `benchmarks/` — CPU/GPU-side timing breakdowns (diagnostic, not a benchmark suite to run routinely)
- `experimental/` — int8 quantization work; **not a working feature**, see `../docs/ENGINE_VALIDATION.md`'s "Int8 quantization experiment" section before touching it

Both `generate_*.py` scripts share the same shape: `build_meta_transformer()`
builds the transformer with 0-byte weights (`accelerate.init_empty_weights()`),
`build_pipeline()` loads everything else (scheduler/tokenizer-or-processor/
text_encoder/vae) as ordinary real weights, then `main()` wires the Rust
engine in via `streamloader_engine.attach_engine(...)` and calls the
pipeline like any other diffusers script.

## Prerequisites

- An NVIDIA GPU + driver with CUDA 13.0 support (`nvidia-smi` should work).
- A Rust toolchain (`rustc`/`cargo` 1.97+, 2024 edition) — needed to build
  `../engine/`, the PyO3 extension these scripts import as
  `streamloader_engine`. It is **not** a PyPI package; `pip`/`uv` cannot
  install it, it must be compiled with `maturin` (below).
- Python **3.10** specifically — the compiled `streamloader_engine`
  extension is built for CPython 3.10's ABI (`cp310` in its filename).
  A different Python minor version needs the engine rebuilt against it
  (`maturin develop --release` again, from that interpreter's venv) —
  nothing stops that, it just isn't what's been tested here.
- ~35GB free disk for model weights (FLUX.2-klein-9B ~18GB downloaded,
  Qwen-Image-2.1 ~31GB) in your Hugging Face cache
  (`~/.cache/huggingface` by default) — both are downloaded automatically
  on first run via `huggingface_hub.snapshot_download`, no manual step.

## Setup with `uv` (what this repo was built/tested with)

See `../docs/BUILD.md` for the full walkthrough. Short version:

```bash
cd /path/to/StreamLoader
uv sync                       # installs torch/diffusers/transformers/etc from pyproject.toml
cd engine
VIRTUAL_ENV=/path/to/StreamLoader/.venv \
  /path/to/StreamLoader/.venv/bin/maturin develop --release
cd ..
uv run --no-sync inference/generate_flux.py --prompt "a fancy speech bubble"
```

`--no-sync` (or `UV_NO_SYNC=1`) is required on every `uv run` in this
repo — see `../docs/BUILD.md` for why.

## Setup without `uv` (plain `venv` + `pip`)

Nothing here actually requires `uv` — it's just what building/testing
used. A plain venv works the same way:

```bash
cd /path/to/StreamLoader
python3.10 -m venv .venv
source .venv/bin/activate        # .venv/Scripts/activate on Windows (untested there)

pip install \
  torch==2.14.0 \
  torchvision==0.29.0 \
  "diffusers @ git+https://github.com/huggingface/diffusers.git" \
  transformers==5.17.0 \
  accelerate==1.15.0 \
  bitsandbytes==0.50.2 \
  pillow==12.3.0 \
  sentencepiece==0.2.2 \
  protobuf==7.36.1 \
  safetensors==0.8.0 \
  numpy==2.5.3 \
  maturin

# diffusers is pinned to git main deliberately, not a typo -- see the
# comment in inference/pyproject.toml. The PyPI release (0.40.0 as of
# 2026-09-23) doesn't have Qwen-Image-2.1's classes yet.
```

Then build the engine the same way as with `uv` (this part is identical
either way -- `maturin` doesn't care how the venv was created):

```bash
cd engine
VIRTUAL_ENV=/path/to/StreamLoader/.venv \
  /path/to/StreamLoader/.venv/bin/maturin develop --release
cd ..
```

Verify, then run:

```bash
.venv/bin/python -c "import streamloader_engine as se; print(se.__file__)"
.venv/bin/python inference/generate_flux.py --prompt "a fancy speech bubble"
```

(No `--no-sync` needed here — that flag only matters for `uv run`, which
you're not using.)

## Usage

```bash
# FLUX.2-klein-9B, step-distilled (defaults to 20 steps)
uv run --no-sync inference/generate_flux.py --prompt "..." --steps 20

# Qwen-Image-2.1, NOT distilled -- needs more steps for clean small text.
# 30 = functional minimum, 45-50 = full quality (see ../docs/RESIDENT_POOL_TODO.md
# for the measured step-count sweep this is based on).
uv run --no-sync inference/generate_qwen.py --prompt "..." --steps 45 --aspect 16:9
```

Common flags on both scripts:

| Flag | Default | What it does |
| --- | --- | --- |
| `--prompt` | *(required)* | Text prompt |
| `--out` | random UUID under `inference/output/` | Output path (batch>1 appends `_0`, `_1`, ...) |
| `--seed` | `0` | Generator seed |
| `--width` / `--height` | `1024`/`1024` | Output resolution |
| `--steps` | `20` (flux) / `40` (qwen) | Denoising steps |
| `--resident-gb` | `0.0` | Static ResidentPool VRAM budget (see `../docs/RESIDENT_POOL_TODO.md`); `0` = pure streaming, identical to before ResidentPool existed |
| `--batch` | `1` | Images per prompt (`num_images_per_prompt`) — no throughput win on a single GPU here, see the batching section of `../docs/RESIDENT_POOL_TODO.md`, but useful for quick variety |
| `--image` | *(none)* | Path to a reference image, enables image-conditioning mode |

`generate_qwen.py` only: `--aspect` picks a size preset from
`ASPECT_RATIOS` in the script (`1:1`, `16:9`, `9:16`, ... and lighter
`-lo` variants) instead of setting `--width`/`--height` directly. Full
2048px+ presets need VAE tiling to avoid an OOM at decode — already
enabled in the script; see `../docs/RESIDENT_POOL_TODO.md` if you're
curious why.

On a desktop (not headless), the first 1-2 seconds of loading a model
can make the machine feel briefly frozen — this is expected, explained,
and has a one-line mitigation in `../docs/BUILD.md`.
