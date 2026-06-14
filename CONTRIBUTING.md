# Contributing

## Local setup

- Use Node.js 20+ for the OpenCode plugin.
- Use Python 3.12+ for tooling under `tools/`.
- Install plugin dependencies from `.opencode` with `npm ci` when `package-lock.json` is present.
- For Python tooling, use the existing `tools/pyproject.toml` setup. Generated datasets and model artifacts stay local.

## Checks

Before opening a pull request, run the narrow checks relevant to your change:

```bash
cd .opencode
npm run check
```

```bash
python -m py_compile tools/collect_opencode_cli.py tools/build_shell_dataset.py tools/shell_features.py tools/train_shell_model.py
```

## Privacy

Do not commit or publish local OpenCode histories, JSONL datasets, trained models, logs, diffs with sensitive content, local filesystem paths, or private tool outputs. Keep examples synthetic or redacted.

## Issues and pull requests

- Search existing issues first.
- Use the bug report or feature request templates.
- Keep pull requests focused and describe the behavior change, checks run, and privacy impact.
- Do not add dependencies unless there is a concrete need and the trade-off is documented.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0-beta.2/) for commit messages.

Examples:

- `feat: add model audit command`
- `fix: ignore local dataset archives`
- `docs: clarify OpenCode plugin setup`
