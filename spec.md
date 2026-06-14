# Unified Output Risk Model

## Goal

Build one local ONNX model that predicts whether an OpenCode tool call is likely to return too much text for the main model context.

The model should cover both shell commands and non-shell tools, including MCP-backed tools, while staying independent from any user's concrete MCP setup.

## Non-Goals

- Do not replace the current ONNX approach with a local LLM.
- Do not train separate models per tool family.
- Do not hardcode MCP server names or MCP tool names in the plugin or training code.
- Do not make MCP handling depend on this repository owner's local server list.
- Do not block core edit/read tools purely because the model predicts large output.

## Core Decision

Use a single output-risk classifier for all tools.

The model answers one question:

```text
Will this tool call likely produce large textual output?
```

The runtime policy decides what to do with that score:

- Shell calls can keep the current warning/blocking behavior for the main agent.
- Shell calls from subagents can keep the current warning-only behavior.
- `webfetch` and external native tools, including MCP-backed tools, may block when they are likely to return excessive output.
- Core edit/read tools should use the score only for logging, routing, capping, summarization, or lightweight-model handling.

## Label

Initial label should stay compatible with the current dataset semantics:

```text
large_output = total_chars >= 2000 or total_lines >= 80
```

This keeps the first unified iteration simple and aligned with the current shell model.

Future reports may include an additional `huge_output` metric, but the first model should remain binary.

## Dataset

The collector can continue exporting all completed tool calls from the local OpenCode SQLite database.

The dataset builder should stop filtering only to `tool_family == "shell"` and `tool == "bash"`. Instead, it should emit unified rows for every supported tool call with:

- schema version
- tool family
- input payload
- shell command and workdir when present
- generic input features
- output-size label
- source metadata

The source data may contain tool names for traceability. Model feature extraction may optionally include local tool identity, but it must be dynamic and data-driven rather than hardcoded.

## Feature Extraction

### Shell Calls

For shell calls, reuse the existing shell feature extractor as much as possible:

- command token classes
- executable and subcommand tokens
- generic intent tokens
- argument, pipe, redirect, glob, path, URL, and length buckets
- cwd shape buckets

Shell command text is acceptable because `bash` is a standard local execution surface, not a private MCP setup.

### Non-Shell Calls

For non-shell calls, use structured-input features plus optional local tool identity.

Tool identity modes:

- `hash`: default; exposes a stable hash token for the tool name, giving local personalization without raw names in model features.
- `raw`: exposes a normalized raw tool-name token for maximum local personalization and easier audit interpretation.
- `none`: disables tool identity features and relies only on input shape.

The default should be `hash`, because it gives most of the predictive value of raw tool names while avoiding raw MCP identifiers in feature vocabularies and audit output.

Allowed feature categories:

- dynamic local tool identity, according to the selected tool identity mode
- JSON size buckets
- object field-count buckets
- array length buckets
- maximum JSON depth buckets
- counts of strings, numbers, booleans, nulls, objects, and arrays
- total string length buckets
- maximum string length buckets
- URL count buckets, without domains
- path-like value count buckets, without concrete paths
- code-like or markdown-like value markers, without raw content
- query-like value length and word-count buckets
- value markers such as `markdown`, `html`, `json`, `raw`, `fit`, `bm25`, or `llm` only when they occur as generic parameter values

Disallowed features:

- hardcoded MCP server names
- hardcoded MCP tool names
- raw MCP/native tool names unless `--tool-identity raw` is explicitly selected
- raw argument key names when they are tool-specific
- URL domains
- concrete file paths
- project names
- user names
- raw long text payloads

The model should learn from input shape, value classes, and optionally local tool identity. It must not contain any hand-authored knowledge about a specific user's MCP setup.

## Unknown Tools

For tools that did not appear in the training set, the model can only generalize from input shape.

This is acceptable because each user trains the model on their own OpenCode history, so most recurring local tools and usage patterns should be represented in their local dataset.

Still, the model is only an early-warning signal. Runtime output measurement remains mandatory:

- predict before execution when possible
- measure actual output after execution
- apply hard output caps or summarization when the actual output exceeds budget
- feed new observations into future retraining

## Runtime Policy

The plugin should score every tool call when model state is available.

Suggested initial policy:

- main-agent shell score >= block threshold: block
- main-agent shell score >= warn threshold: warn
- subagent shell score >= warn threshold: log warning only
- main-agent `webfetch` score >= block threshold: block
- main-agent external native/MCP-like score >= block threshold: block
- core native tool score >= threshold: never block; log or mark as large-output risk
- subagent native score >= threshold: log warning only
- any actual output over budget: cap, summarize, or route regardless of prediction

Native tools are split by policy, not by model type:

- block-capable: `webfetch` and native tools that are not known built-in edit/read/control tools; this covers MCP in a setup-agnostic way without naming MCP servers or tools
- never-block: built-in edit/read/control tools such as `apply_patch`, `read`, `write`, `edit`, `glob`, `grep`, task/delegation helpers, questions, todo updates, and skill loading

The exact post-execution handling depends on what OpenCode plugin hooks expose before and after tool execution.

## Training And Evaluation

Keep the current lightweight baseline unless evidence says otherwise:

- TF-IDF over generated token streams
- LogisticRegression with balanced class weights
- ONNX export
- threshold selection on validation data

Report metrics separately for:

- all tools
- shell calls
- non-shell calls
- high-output classes

In addition to the current random split, add a time-based evaluation report when possible:

- train on older calls
- validate/test on newer calls

This is a better proxy for future behavior than random split alone.

## Migration Plan

1. Rename the concept from shell blocker to output-risk classifier in docs and artifact metadata.
2. Add unified feature extraction in Python.
3. Mirror the same token contract in the plugin JavaScript runtime.
4. Update the dataset builder to emit shell and non-shell rows.
5. Keep trainer architecture mostly unchanged.
6. Regenerate dataset and model from local history.
7. Add audit output that redacts or avoids setup-specific identifiers.
8. Update plugin policy to score all tools but keep behavior differentiated by tool family.

## Acceptance Criteria

- One ONNX model is used for all tool calls.
- MCP/native model features contain no raw MCP server names or raw MCP tool names in the default `hash` mode.
- Shell protection behavior does not regress.
- Non-shell large-output calls are detected well enough to be useful as routing signals.
- Actual output size remains measured at runtime as a safety net.
- Audit artifacts do not expose setup-specific MCP identifiers as learned features.
