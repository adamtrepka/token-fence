from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_BLOCKED_CHARS = 2000
DEFAULT_BLOCKED_LINES = 80

PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")
REDIRECT_RE = re.compile(r"(?:\d?>|>>|<&|>&|<|>)")
GLOB_RE = re.compile(r"(?<!\\)[*?\[]")
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SplitSet:
    train: list[dict[str, object]]
    val: list[dict[str, object]]
    test: list[dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build shell-only training datasets from OpenCode collector JSONL."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("opencode-tool-calls.jsonl"),
        help="Input JSONL produced by casifier-collect.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset-shell"),
        help="Directory for train/val/test JSONL outputs.",
    )
    parser.add_argument(
        "--blocked-chars",
        type=int,
        default=DEFAULT_BLOCKED_CHARS,
        help="Label as blocked when total_chars reaches this threshold.",
    )
    parser.add_argument(
        "--blocked-lines",
        type=int,
        default=DEFAULT_BLOCKED_LINES,
        help="Label as blocked when total_lines reaches this threshold.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="Validation split ratio.",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.15,
        help="Test split ratio.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for stratified splitting.",
    )
    parser.add_argument(
        "--dedupe",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deduplicate repeated commands within the same cwd.",
    )
    return parser


def read_jsonl(path: Path) -> Iterable[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                yield value


def normalize_whitespace(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text).strip()


def split_command(command: str) -> list[str]:
    normalized = normalize_whitespace(command)
    if not normalized:
        return []
    return normalized.split(" ")


def extract_workdir(input_raw: object) -> str | None:
    if not isinstance(input_raw, dict):
        return None
    for key in ("workdir", "cwd", "path"):
        value = input_raw.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def extract_command(record: dict[str, object]) -> str | None:
    command = record.get("command")
    if isinstance(command, str) and command.strip():
        return command.strip()

    input_raw = record.get("input_raw")
    if isinstance(input_raw, dict):
        raw_command = input_raw.get("command")
        if isinstance(raw_command, str) and raw_command.strip():
            return raw_command.strip()
    return None


def command_features(command: str, input_raw: object) -> dict[str, object]:
    normalized = normalize_whitespace(command)
    tokens = split_command(command)
    workdir = extract_workdir(input_raw)
    workdir_depth = 0
    if workdir:
        workdir_depth = max(workdir.replace("\\", "/").count("/") - 1, 0)

    lower = normalized.lower()
    return {
        "input_length": len(command),
        "normalized_length": len(normalized),
        "arg_count": max(len(tokens) - 1, 0),
        "token_count": len(tokens),
        "pipe_count": len(PIPE_RE.findall(command)),
        "redirect_count": len(REDIRECT_RE.findall(command)),
        "glob_count": len(GLOB_RE.findall(command)),
        "has_recursive": any(flag in lower for flag in (" -r ", " --recursive ", " -R ", " /s ", " find ")),
        "has_find": "find " in lower or lower.startswith("find "),
        "has_rg": lower.startswith("rg ") or " rg " in lower,
        "has_grep": lower.startswith("grep ") or " grep " in lower,
        "has_sudo": lower.startswith("sudo ") or " sudo " in lower,
        "has_pipe": "|" in command,
        "has_redirect": bool(REDIRECT_RE.search(command)),
        "cwd_depth": workdir_depth,
        "workdir": workdir,
    }


def make_label(record: dict[str, object], blocked_chars: int, blocked_lines: int) -> dict[str, object]:
    total_chars = int(record.get("total_chars") or 0)
    total_lines = int(record.get("total_lines") or 0)
    blocked = total_chars >= blocked_chars or total_lines >= blocked_lines
    size_class = "large" if blocked else "small"
    return {
        "stdout_lines": int(record.get("output_lines") or 0),
        "stdout_chars": int(record.get("output_chars") or 0),
        "stderr_lines": int(record.get("stderr_lines") or 0),
        "stderr_chars": int(record.get("stderr_chars") or 0),
        "total_lines": total_lines,
        "total_chars": total_chars,
        "blocked": blocked,
        "size_class": size_class,
    }


def row_signature(tool: str, workdir: str | None, command: str) -> str:
    payload = json.dumps({"tool": tool, "workdir": workdir, "command": command}, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def build_rows(
    records: Iterable[dict[str, object]],
    blocked_chars: int,
    blocked_lines: int,
    dedupe: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for record in records:
        if record.get("tool_family") != "shell":
            continue
        if record.get("tool") != "bash":
            continue

        command = extract_command(record)
        if command is None:
            continue

        input_raw = record.get("input_raw")
        workdir = extract_workdir(input_raw)
        signature = row_signature("bash", workdir, command)
        if dedupe and signature in seen:
            continue
        seen.add(signature)

        row = {
            "schema_version": 1,
            "tool_family": "shell",
            "tool": "bash",
            "input": {
                "command": command,
                **({"workdir": workdir} if workdir is not None else {}),
            },
            "features": command_features(command, input_raw),
            "label": make_label(record, blocked_chars, blocked_lines),
            "source": record.get("source") or {},
        }
        rows.append(row)

    return rows


def stratified_split(rows: list[dict[str, object]], val_ratio: float, test_ratio: float, seed: int) -> SplitSet:
    if val_ratio < 0 or test_ratio < 0:
        raise ValueError("Split ratios must be non-negative.")
    if val_ratio + test_ratio >= 1:
        raise ValueError("val-ratio + test-ratio must be less than 1.")

    grouped: dict[bool, list[dict[str, object]]] = {False: [], True: []}
    for row in rows:
        grouped[bool(row["label"]["blocked"])].append(row)

    rng = random.Random(seed)
    train: list[dict[str, object]] = []
    val: list[dict[str, object]] = []
    test: list[dict[str, object]] = []

    for label, items in grouped.items():
        rng.shuffle(items)
        total = len(items)
        test_count = int(round(total * test_ratio))
        val_count = int(round(total * val_ratio))
        if test_count + val_count > total:
            overflow = test_count + val_count - total
            while overflow > 0 and val_count > 0:
                val_count -= 1
                overflow -= 1
            while overflow > 0 and test_count > 0:
                test_count -= 1
                overflow -= 1
        train_count = total - val_count - test_count

        train.extend(items[:train_count])
        val.extend(items[train_count:train_count + val_count])
        test.extend(items[train_count + val_count:train_count + val_count + test_count])

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return SplitSet(train=train, val=val, test=test)


def write_jsonl(path: Path, rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize(rows: list[dict[str, object]]) -> dict[str, object]:
    blocked = sum(1 for row in rows if row["label"]["blocked"])
    return {
        "rows": len(rows),
        "blocked": blocked,
        "allowed": len(rows) - blocked,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = build_rows(
        read_jsonl(args.input),
        blocked_chars=args.blocked_chars,
        blocked_lines=args.blocked_lines,
        dedupe=args.dedupe,
    )
    splits = stratified_split(rows, args.val_ratio, args.test_ratio, args.seed)

    write_jsonl(args.output_dir / "train.jsonl", splits.train)
    write_jsonl(args.output_dir / "val.jsonl", splits.val)
    write_jsonl(args.output_dir / "test.jsonl", splits.test)

    manifest = {
        "schema_version": 1,
        "input": str(args.input),
        "blocked_chars": args.blocked_chars,
        "blocked_lines": args.blocked_lines,
        "dedupe": args.dedupe,
        "split": {
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
        },
        "summary": {
            "all": summarize(rows),
            "train": summarize(splits.train),
            "val": summarize(splits.val),
            "test": summarize(splits.test),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
