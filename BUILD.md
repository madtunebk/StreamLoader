# Building the Rust engine addon

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
