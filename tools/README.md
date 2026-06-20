# Casifier Tools

This directory contains the Python tooling for the Casifier pipeline.

## What this does
- Collects OpenCode CLI/tool calls from the local SQLite database.
- Builds a unified output-risk training dataset for shell and native tool calls.
- Trains an ONNX-exportable model for runtime output-risk scoring.

## Requirements
- Python 3.12+
- `uv`

## Project layout
- `collect_opencode_cli.py` - exports raw OpenCode tool calls to JSONL.
- `build_shell_dataset.py` - turns raw tool calls into unified train/val/test JSONL files.
- `shell_features.py` - shared generic shell tokenization and feature extraction.
- `train_shell_model.py` - trains the output-risk model and exports ONNX.
- `pyproject.toml` - project metadata and console scripts.

## Install
From this directory:

```bash
uv sync
```

## Step 1: Collect raw data
Collect all completed tool calls:

```bash
uv run casifier-collect --output tool-calls.jsonl
```

For shell-only experiments, you can still pass `--tool bash`, but the default model should use the full local tool-call history.

Useful options:
- `--db <path>` to point to a custom OpenCode database file or directory.
- `--search-root <path>` to add extra directories to scan.
- `--limit <n>` to sample only a few rows.
- `--stdout` to stream JSONL to stdout.

## Step 2: Build the dataset
Turn raw tool calls into training splits:

```bash
uv run casifier-build-dataset --input tool-calls.jsonl --output-dir dataset-shell
```

Default behavior:
- deduplicates repeated tool inputs
- labels a row as `blocked` if `total_chars >= 2000` or `total_lines >= 80`; semantically this means `large_output`
- keeps shell command features for `bash`, including generic flag-name / short-flag / numeric buckets and output-control tokens
- adds structured-input features for native tools, including capped field-name tokens, field classes, and limit-value buckets
- uses `--tool-identity hash` by default for native tools
- writes:
  - `dataset-shell/train.jsonl`
  - `dataset-shell/val.jsonl`
  - `dataset-shell/test.jsonl`
  - `dataset-shell/manifest.json`

Tool identity modes:
- `hash` - default; stable tool-name hash token for local personalization without raw tool names in feature tokens.
- `raw` - normalized raw tool-name token for maximum local personalization and easier audit interpretation.
- `none` - no tool identity feature; relies only on input shape.

## Step 3: Train the model
Train the ONNX-safe output-risk model:

```bash
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
```

This produces:
- `model-shell/model.onnx`
- `model-shell/model.joblib`
- `model-shell/threshold.json`
- `model-shell/manifest.json`

Default threshold selection targets validation precision `0.60` for binary blocking and falls back to F-beta only if no validation threshold meets that precision. The manifest records the selection method, target precision, and any fallback details.

The manifest also records the unified feature contract, including native tool identity mode, and reports metrics for all tools plus `shell` and `native` families separately.

## Model input

The model input is a single token stream derived from shell commands or native tool JSON input. Shell features include command tokens, executable/subcommand markers, operators, flags, paths, URLs, globs, output-limiting / quieting cues (`--quiet`, `--silent`, `head -n`, `tail -n`, `Select-Object -First`, `git log -n`, `--tail`, `-m/--max-count`), and bucketed aggregate counts. Native-tool features include JSON shape, value-type counts, string/path/URL-like markers, and optional tool identity (`hash`, `raw`, or `none`).

## Artifacts

Training writes:

- `model-shell/model.onnx` - runtime model for the Node.js plugin
- `model-shell/model.joblib` - Python/scikit-learn model
- `model-shell/threshold.json` - selected block/warn thresholds
- `model-shell/manifest.json` - feature contract and metrics

## Notes
- Generated datasets and models are ignored by git.
- The ONNX model is the artifact intended for the Node.js plugin runtime.
- The command names still contain `shell` for compatibility, but the dataset and trainer now support unified shell/native output-risk modeling.
