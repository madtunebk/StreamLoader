# Building the Rust engine addon

## Always run `uv run --no-sync` (or set `UV_NO_SYNC=1`) in this repo

**Never run a bare `uv run ...` anywhere in this repo.** There used to
be two `pyproject.toml` files (root + `inference/`), which caused a real
incident: a bare `uv run inference/generate_qwen.py --help` from the
repo root synced against the *root* file's looser `diffusers` pin,
silently downgrading a deliberately git-installed `diffusers` back to a
plain PyPI release and breaking every Qwen-Image-2.1 import. They've
since been merged into one root `pyproject.toml` specifically to remove
that "which file is in scope" ambiguity.

That fixed *one* cause, not the underlying risk: `uv sync` (which any
bare `uv run` can trigger) reconciles the venv strictly against
`pyproject.toml` — including *removing* anything installed that isn't
listed there. `streamloader_engine` (built locally by `maturin`, not a
PyPI package) is exactly such a thing: a real `uv sync` run while
merging the two files removed it from the venv outright, requiring
`maturin develop --release` to be re-run before anything using it would
import again. `--no-sync` is what actually prevents this — don't rely on
the dependency pins alone, and after any deliberate `uv sync` (e.g. after
editing `pyproject.toml`), always re-run the engine build step below
before assuming things still work.

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
cd /path/to/StreamLoader
uv add --dev maturin
```

## Build + install (run this any time engine/src changes)

```bash
cd /path/to/StreamLoader/engine
VIRTUAL_ENV=/path/to/StreamLoader/.venv \
  /path/to/StreamLoader/.venv/bin/maturin develop --release
```

This compiles the crate and installs it editable into
`.venv/lib/python3.10/site-packages/streamloader_engine/`. It does **not**
rebuild automatically — re-run it after every change to `engine/src/*.rs`.

## Verify

```bash
cd /path/to/StreamLoader
.venv/bin/python -c "import streamloader_engine as se; print(se.__file__)"
```

## Run inference

```bash
cd /path/to/StreamLoader
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
