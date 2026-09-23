# Building the Rust engine addon

## Always run `uv run --no-sync` (or set `UV_NO_SYNC=1`) in this repo

**Never run a bare `uv run ...` here, from the repo root OR from
`inference/`.** Both directories have their own `pyproject.toml`, and a
bare `uv run` reconciles (`uv sync`s) the venv against whichever one is
in scope for the path you ran from -- it doesn't just add missing
packages, it can downgrade or remove what's already installed to match
that file exactly. This already happened once for real: `diffusers` was
installed from git main (needed for `inference/generate_qwen.py` --
Qwen-Image-2.1's classes aren't in any PyPI release yet), then a bare
`uv run inference/generate_qwen.py --help` from the repo root silently
reinstalled plain `diffusers==0.40.0` from PyPI over it (matching the
root `pyproject.toml`'s pin at the time), breaking every Qwen-Image-2.1
import until it was reinstalled from git again. Both `pyproject.toml`
(root) and `inference/pyproject.toml` now pin `diffusers` to the git
source specifically so a sync wouldn't silently regress this again --
but `--no-sync` is still the actual safety net; don't rely on the pins
alone. It's also required so `uv` doesn't try to manage
`streamloader_engine` and `torchvision`, neither of which is a normal
PyPI dependency it fully understands here.

`streamloader_engine` (imported by `inference/generate_rust.py`) is a PyO3
extension crate at `engine/`. It is not a PyPI package and `uv sync` will
never install it — it has to be built and installed into the venv manually
with `maturin`.

`engine/` intentionally has no `pyproject.toml` of its own. If you run
`maturin develop` without `VIRTUAL_ENV` set, maturin walks up the directory
tree, finds the repo-root `pyproject.toml` (torch/diffusers/etc.), and fails
because that file has no `[build-system]` table. Setting `VIRTUAL_ENV`
explicitly skips that lookup.

## One-time setup

```bash
cd ~/Raid0/RustStream
uv add --dev maturin
```

## Build + install (run this any time engine/src changes)

```bash
cd ~/Raid0/RustStream/engine
VIRTUAL_ENV=/home/nobus/Raid0/RustStream/.venv \
  /home/nobus/Raid0/RustStream/.venv/bin/maturin develop --release
```

This compiles the crate and installs it editable into
`.venv/lib/python3.10/site-packages/streamloader_engine/`. It does **not**
rebuild automatically — re-run it after every change to `engine/src/*.rs`.

## Verify

```bash
cd ~/Raid0/RustStream
.venv/bin/python -c "import streamloader_engine as se; print(se.__file__)"
```

## Run inference

```bash
cd ~/Raid0/RustStream
uv run --no-sync inference/generate_rust.py --prompt "..."
```

`--no-sync` (or `UV_NO_SYNC=1`) is required — otherwise `uv run` tries to
reconcile the venv against `pyproject.toml`'s dependency list and won't know
about the locally-built `streamloader-engine` package.
