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

`streamloader_engine` (imported by `inference/generate_flux.py`) is a PyO3
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
cd /path/to/RustStream
uv add --dev maturin
```

## Build + install (run this any time engine/src changes)

```bash
cd /path/to/RustStream/engine
VIRTUAL_ENV=/path/to/RustStream/.venv \
  /path/to/RustStream/.venv/bin/maturin develop --release
```

This compiles the crate and installs it editable into
`.venv/lib/python3.10/site-packages/streamloader_engine/`. It does **not**
rebuild automatically — re-run it after every change to `engine/src/*.rs`.

## Verify

```bash
cd /path/to/RustStream
.venv/bin/python -c "import streamloader_engine as se; print(se.__file__)"
```

## Run inference

```bash
cd /path/to/RustStream
uv run --no-sync inference/generate_flux.py --prompt "..."
```

`--no-sync` (or `UV_NO_SYNC=1`) is required — otherwise `uv run` tries to
reconcile the venv against `pyproject.toml`'s dependency list and won't know
about the locally-built `streamloader-engine` package.

## Desktop freeze/mouse-lag during model load (harmless, explained)

On a desktop system (not a headless server), the first 1-2 seconds of
`RustEngine::new()` can make the whole machine feel briefly frozen or
laggy — observed directly: dropped mouse input events
(`SYN_DROPPED` in the X server log) right at load time, not during
generation itself. This is not a bug or a hardware fault — it's
`cudaHostAlloc`'ing and filling the *entire* pinned host buffer (the
whole transformer, ~18GB for FLUX.2-klein-9B, ~14GB for Qwen-Image-2.1)
in one uninterrupted burst: the kernel page-locking that much RAM in one
call, plus the disk reads that fill it, both run at normal process
priority and can starve the desktop's input/GUI threads for those couple
of seconds. (The engine deliberately has no partial/LRU pinned cache —
see `docs/ENGINE.md` — so this whole-buffer burst is inherent to the
current design, not something a config flag shrinks.)

If this is disruptive on your machine, lower the process's CPU/IO
priority so the desktop stays responsive through the load burst — the
engine finishes just as fast once the system is otherwise idle, it just
yields immediately if something else needs the CPU or disk:

```bash
ionice -c3 nice -n 10 uv run --no-sync inference/generate_flux.py --prompt "..."
```

`ionice -c3` = idle I/O class (yields to any other disk request),
`nice -n 10` = lower CPU scheduling priority. Verified this doesn't
change behavior or timing under normal (uncontended) conditions — only
matters when something else actually wants the CPU/disk at the same
moment.
