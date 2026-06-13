# Token Fence

Token Fence is an OpenCode shell-guard plugin backed by a small machine-learning model. It watches `bash` tool calls before execution, scores the command with an ONNX model, and blocks or warns on command shapes that look risky for the main agent.

The current version is intentionally narrow: it protects shell execution in OpenCode. The model is trained from local OpenCode command history, exported to ONNX, and loaded by the Node.js plugin at runtime.

## What It Contains

- `.opencode/plugins/shell-guard.js` - OpenCode plugin that intercepts shell tool execution.
- `tools/shell_features.py` - shared shell feature extraction used by training.
- `tools/train_shell_model.py` - trains and exports the ONNX model.
- `tools/build_shell_dataset.py` - builds train/validation/test splits from collected OpenCode shell calls.
- `tools/collect_opencode_cli.py` - collects local OpenCode tool-call history into JSONL.
- `tools/audit_shell_model.py` - audits model errors and feature weights.

Generated datasets and model artifacts are intentionally ignored by git. A working plugin install needs `tools/model-shell/model.onnx` and `tools/model-shell/threshold.json` to exist locally.

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
uv run casifier-collect --tool bash --output shell-only.jsonl
uv run casifier-build-dataset --input shell-only.jsonl --output-dir dataset-shell
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
```

This must produce:

- `tools/model-shell/model.onnx`
- `tools/model-shell/threshold.json`

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

The plugin handles OpenCode's `tool.execute.before` hook for `bash` commands. For every command it can resolve, it:

1. Builds the same token stream shape used during model training.
2. Runs `tools/model-shell/model.onnx` with `onnxruntime-node`.
3. Reads thresholds from `tools/model-shell/threshold.json`.
4. Blocks high-scoring commands for the main agent.
5. Logs warnings instead of blocking for subagents.

It also blocks nested `opencode -s/--session` shell commands for the main agent even before model scoring.

## Retrain The Model

Use the documented pipeline in `tools/README.md`. The short version is:

```bash
cd tools
uv sync
uv run casifier-collect --tool bash --output shell-only.jsonl
uv run casifier-build-dataset --input shell-only.jsonl --output-dir dataset-shell
uv run casifier-train-shell-model --input-dir dataset-shell --output-dir model-shell
uv run casifier-audit-shell-model --output shell-model-audit.md
```

Restart OpenCode after retraining so the plugin loads the new ONNX model and thresholds.

## Verify

Check Python syntax:

```bash
python -m py_compile tools/shell_features.py tools/train_shell_model.py tools/build_shell_dataset.py tools/audit_shell_model.py
```

Check plugin syntax:

```bash
node --check .opencode/plugins/shell-guard.js
```

## Notes

- The plugin is shell-only in this version.
- The generated model is local by design and is not committed.
- If you move `shell-guard.js`, update `MODEL_DIR` in the plugin or keep the same relative layout.
