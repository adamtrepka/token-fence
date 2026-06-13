# Casifier Tools

This directory contains the Python tooling for the Casifier pipeline.

## What this does
- Collects OpenCode CLI/tool calls from the local SQLite database.
- Builds a shell-only training dataset.
- Trains an ONNX-exportable model for the first runtime blocker.

## Requirements
- Python 3.12+
- `uv`

## Project layout
- `collect_opencode_cli.py` - exports raw OpenCode tool calls to JSONL.
- `build_shell_dataset.py` - turns raw shell calls into train/val/test JSONL files.
- `train_shell_model.py` - trains the shell blocker and exports ONNX.
- `pyproject.toml` - project metadata and console scripts.

## Install
From this directory:

```bash
uv sync
```

## Step 1: Collect raw data
Collect all shell calls only:

```bash
uv run casifier-collect --tool bash --output shell-only.jsonl
```

Useful options:
- `--db <path>` to point to a custom OpenCode database file or directory.
- `--search-root <path>` to add extra directories to scan.
- `--limit <n>` to sample only a few rows.
- `--stdout` to stream JSONL to stdout.

## Step 2: Build the dataset
Turn raw shell calls into training splits:

```bash
uv run casifier-build-dataset --input shell-only.jsonl --output-dir dataset-shell
```

Default behavior:
- deduplicates by `tool + workdir + command`
- labels a row as `blocked` if `total_chars >= 2000` or `total_lines >= 80`
- writes:
  - `dataset-shell/train.jsonl`
  - `dataset-shell/val.jsonl`
  - `dataset-shell/test.jsonl`
  - `dataset-shell/manifest.json`

## Step 3: Train the model
Train the first ONNX-safe shell blocker:

```bash
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
```

This produces:
- `model-shell/model.onnx`
- `model-shell/model.joblib`
- `model-shell/threshold.json`
- `model-shell/manifest.json`

## Model input
The trainer builds a token stream from:
- `cwd`
- the shell `command`
- simple shell features like argument count, pipes, redirects, globs, recursion, and cwd depth

## Notes
- v1 is shell-only (`bash`).
- Generated datasets and models are ignored by git.
- The ONNX model is the artifact intended for the Node.js plugin runtime.
