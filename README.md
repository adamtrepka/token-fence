# Token Fence

[![CI](https://github.com/adamtrepka/token-fence/actions/workflows/ci.yml/badge.svg)](https://github.com/adamtrepka/token-fence/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Token Fence is an OpenCode output-risk guard backed by a small machine-learning model. It watches tool calls before execution, scores the expected output size with an ONNX model, and blocks or warns according to runtime policy.

The model is trained from local OpenCode tool-call history, exported to ONNX, and loaded by the Node.js plugin at runtime. It supports shell commands and unified native-tool input features, including setup-agnostic handling for MCP-like tools.

## What It Contains

- `.opencode/plugins/shell-guard.js` - OpenCode plugin that intercepts tool execution.
- `tools/shell_features.py` - shared shell feature extraction used by training.
- `tools/train_shell_model.py` - trains and exports the ONNX model.
- `tools/build_shell_dataset.py` - builds unified train/validation/test splits from collected OpenCode tool calls.
- `tools/collect_opencode_cli.py` - collects local OpenCode tool-call history into JSONL.

Generated datasets and model artifacts are intentionally ignored by git. A working plugin install needs `tools/model-shell/model.onnx`, `tools/model-shell/threshold.json`, and `tools/model-shell/manifest.json` to exist locally.

## Requirements

- OpenCode with plugin support.
- Node.js 20+.
- Python 3.12+.
- `uv` for the Python tooling.

## Install The Plugin Globally

Clone the repository into a stable location. The plugin resolves the model path relative to this repository layout, so do not copy only `shell-guard.js` unless you also update `MODEL_DIR` inside the plugin.

```bash
git clone https://github.com/adamtrepka/token-fence.git
cd token-fence/.opencode
npm install
```

Build or provide the model artifacts:

```bash
cd ../tools
uv sync
uv run casifier-collect --output tool-calls.jsonl
uv run casifier-build-dataset --input tool-calls.jsonl --output-dir dataset-shell
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
```

This must produce:

- `tools/model-shell/model.onnx`
- `tools/model-shell/threshold.json`
- `tools/model-shell/manifest.json`

Add the plugin to your global OpenCode config at `~/.config/opencode/opencode.json`. Use an absolute path to this repository's plugin file and replace the placeholder with your local clone path.

Windows example:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "file:///C:/path/to/token-fence/.opencode/plugins/shell-guard.js"
  ]
}
```

Linux/macOS example:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "plugin": [
    "file:///absolute/path/to/token-fence/.opencode/plugins/shell-guard.js"
  ]
}
```

If your `opencode.json` already has a `plugin` array, append the `file:///.../shell-guard.js` entry instead of replacing the existing list.

Restart OpenCode after changing `opencode.json`; config and plugins are loaded at startup.

You can verify that OpenCode accepts the config with:

```bash
opencode debug config
```

## How It Works

The plugin handles OpenCode's `tool.execute.before` hook. For every supported tool call it can resolve, it:

1. Builds the same token stream shape used during model training.
2. Runs `tools/model-shell/model.onnx` with `onnxruntime-node`.
3. Reads thresholds from `tools/model-shell/threshold.json`.
4. Uses `tools/model-shell/manifest.json` to verify whether the loaded model supports native-tool scoring.
5. Applies tool-specific runtime policy.

It also blocks nested `opencode -s/--session` shell commands for the main agent even before model scoring.

Runtime policy:

- `bash` can block or warn for the main agent.
- `webfetch` and external native/MCP-like tools can block when predicted output risk is high.
- Built-in edit/read/control tools such as `apply_patch`, `read`, `write`, `edit`, `glob`, `grep`, `task`, `delegate`, `question`, `todowrite`, and `skill` never block from this model; they can only warn/log.
- Subagents are warning-only.

## Retrain The Model

Use the documented pipeline in `tools/README.md`. The short version is:

```bash
cd tools
uv sync
uv run casifier-collect --output tool-calls.jsonl
uv run casifier-build-dataset --input tool-calls.jsonl --output-dir dataset-shell
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
```

`casifier-build-dataset` supports `--tool-identity hash|raw|none`. The default `hash` mode gives local personalization without putting raw MCP/native tool names into feature tokens.

Restart OpenCode after retraining so the plugin loads the new ONNX model and thresholds.

## Verify

Check Python syntax:

```bash
python -m py_compile tools/shell_features.py tools/train_shell_model.py tools/build_shell_dataset.py tools/collect_opencode_cli.py
```

Check plugin syntax:

```bash
node --check .opencode/plugins/shell-guard.js
```

## Notes

- The generated model is local by design and is not committed.
- Native/MCP handling is setup-agnostic: the plugin does not hardcode MCP server or tool names.
- If you move `shell-guard.js`, update `MODEL_DIR` in the plugin or keep the same relative layout.
