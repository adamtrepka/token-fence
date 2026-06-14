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
MAX_BUCKET = 10**12

PIPE_RE = re.compile(r"(?<!\|)\|(?!\|)")
REDIRECT_RE = re.compile(r"(?:\d?>|>>|<&|>&|<|>)")
GLOB_RE = re.compile(r"(?<!\\)[*?\[]")
WHITESPACE_RE = re.compile(r"\s+")
URL_VALUE_RE = re.compile(r"https?://\S+", re.I)
WINDOWS_PATH_VALUE_RE = re.compile(r"(?i)^(?:[a-z]:[\\/]|\\\\)")
POSIX_PATH_VALUE_RE = re.compile(r"^(?:/|\.\.?/|~[/\\]?)")
PATH_EXTENSION_RE = re.compile(r"\.[A-Za-z0-9]{1,10}(?:$|[?#])")
MARKDOWN_VALUE_RE = re.compile(r"(^|\n)\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+\.\s|```|>)")
CODE_VALUE_RE = re.compile(r"```|=>|[{};]|\b(?:class|def|function|import|select|from|where)\b", re.I)

GENERIC_VALUE_MARKERS = {
    "bm25",
    "csv",
    "fit",
    "html",
    "json",
    "llm",
    "markdown",
    "md",
    "pdf",
    "png",
    "raw",
    "svg",
    "text",
    "xml",
    "yaml",
}

COUNT_BUCKETS = [
    (0, "0"),
    (1, "1"),
    (2, "2"),
    (5, "3_5"),
    (10, "6_10"),
    (25, "11_25"),
    (50, "26_50"),
    (100, "51_100"),
    (250, "101_250"),
    (MAX_BUCKET, "251_plus"),
]
CHAR_BUCKETS = [
    (0, "0"),
    (80, "1_80"),
    (240, "81_240"),
    (800, "241_800"),
    (2000, "801_2000"),
    (8000, "2001_8000"),
    (32000, "8001_32000"),
    (MAX_BUCKET, "32001_plus"),
]
DEPTH_BUCKETS = [(0, "0"), (1, "1"), (2, "2"), (4, "3_4"), (8, "5_8"), (MAX_BUCKET, "9_plus")]


@dataclass(frozen=True)
class SplitSet:
    train: list[dict[str, object]]
    val: list[dict[str, object]]
    test: list[dict[str, object]]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build unified output-risk training datasets from OpenCode collector JSONL."
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
        help="Deduplicate repeated tool inputs.",
    )
    parser.add_argument(
        "--tool-identity",
        choices=("hash", "raw", "none"),
        default="hash",
        help="How to expose non-shell tool identity as a feature. 'hash' personalizes without raw tool names.",
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


def count_bucket(value: int, buckets: list[tuple[int, str]]) -> str:
    for upper, label in buckets:
        if value <= upper:
            return label
    return buckets[-1][1]


def bucket_token(prefix: str, value: int, buckets: list[tuple[int, str]]) -> str:
    return f"{prefix}_{count_bucket(value, buckets)}"


def normalize_identifier(value: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    return normalized[:80] or "empty"


def tool_identity_feature(tool: str, mode: str) -> str | None:
    if mode == "none" or not tool:
        return None
    if mode == "raw":
        return f"tool_raw_{normalize_identifier(tool)}"
    digest = hashlib.sha1(tool.encode("utf-8")).hexdigest()[:12]
    return f"tool_hash_{digest}"


def canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def is_path_like(value: str) -> bool:
    text = value.strip()
    if not text or URL_VALUE_RE.search(text):
        return False
    if WINDOWS_PATH_VALUE_RE.search(text) or POSIX_PATH_VALUE_RE.search(text):
        return True
    if ("/" in text or "\\" in text) and not WHITESPACE_RE.search(text) and PATH_EXTENSION_RE.search(text):
        return True
    return False


def word_count(value: str) -> int:
    return len(re.findall(r"\w+", value))


def empty_native_stats() -> dict[str, object]:
    return {
        "input_json_chars": 0,
        "max_depth": 0,
        "object_count": 0,
        "array_count": 0,
        "field_count": 0,
        "max_object_fields": 0,
        "max_array_length": 0,
        "string_count": 0,
        "string_total_chars": 0,
        "string_max_chars": 0,
        "string_total_lines": 0,
        "string_max_lines": 0,
        "number_count": 0,
        "boolean_count": 0,
        "null_count": 0,
        "url_count": 0,
        "path_like_count": 0,
        "markdown_like_count": 0,
        "code_like_count": 0,
        "json_like_string_count": 0,
        "long_string_count": 0,
        "query_like_count": 0,
        "query_word_max": 0,
        "value_markers": set(),
    }


def inspect_native_string(value: str, stats: dict[str, object]) -> None:
    stripped = value.strip()
    line_count = len(value.splitlines()) if value else 0
    words = word_count(value)

    stats["string_count"] = int(stats["string_count"]) + 1
    stats["string_total_chars"] = int(stats["string_total_chars"]) + len(value)
    stats["string_max_chars"] = max(int(stats["string_max_chars"]), len(value))
    stats["string_total_lines"] = int(stats["string_total_lines"]) + line_count
    stats["string_max_lines"] = max(int(stats["string_max_lines"]), line_count)
    stats["url_count"] = int(stats["url_count"]) + len(URL_VALUE_RE.findall(value))

    if is_path_like(value):
        stats["path_like_count"] = int(stats["path_like_count"]) + 1
    if MARKDOWN_VALUE_RE.search(value):
        stats["markdown_like_count"] = int(stats["markdown_like_count"]) + 1
    if CODE_VALUE_RE.search(value):
        stats["code_like_count"] = int(stats["code_like_count"]) + 1
    if stripped.startswith(("{", "[")) and stripped.endswith(("}", "]")):
        stats["json_like_string_count"] = int(stats["json_like_string_count"]) + 1
    if len(value) >= 1000:
        stats["long_string_count"] = int(stats["long_string_count"]) + 1
    if 3 <= words <= 80 and not is_path_like(value):
        stats["query_like_count"] = int(stats["query_like_count"]) + 1
        stats["query_word_max"] = max(int(stats["query_word_max"]), words)

    marker = stripped.lower()
    if marker in GENERIC_VALUE_MARKERS:
        stats["value_markers"].add(marker)  # type: ignore[union-attr]


def walk_native_input(value: object, stats: dict[str, object], depth: int = 0) -> None:
    stats["max_depth"] = max(int(stats["max_depth"]), depth)

    if isinstance(value, dict):
        stats["object_count"] = int(stats["object_count"]) + 1
        stats["field_count"] = int(stats["field_count"]) + len(value)
        stats["max_object_fields"] = max(int(stats["max_object_fields"]), len(value))
        for item in value.values():
            walk_native_input(item, stats, depth + 1)
        return

    if isinstance(value, list):
        stats["array_count"] = int(stats["array_count"]) + 1
        stats["max_array_length"] = max(int(stats["max_array_length"]), len(value))
        for item in value:
            walk_native_input(item, stats, depth + 1)
        return

    if isinstance(value, str):
        inspect_native_string(value, stats)
        return

    if isinstance(value, bool):
        stats["boolean_count"] = int(stats["boolean_count"]) + 1
        return

    if isinstance(value, int | float):
        stats["number_count"] = int(stats["number_count"]) + 1
        return

    if value is None:
        stats["null_count"] = int(stats["null_count"]) + 1


def native_tokens(stats: dict[str, object], identity: str | None) -> list[str]:
    tokens = ["family_native"]
    if identity:
        tokens.append(identity)

    count_fields = (
        "object_count",
        "array_count",
        "field_count",
        "max_object_fields",
        "max_array_length",
        "string_count",
        "number_count",
        "boolean_count",
        "null_count",
        "url_count",
        "path_like_count",
        "markdown_like_count",
        "code_like_count",
        "json_like_string_count",
        "long_string_count",
        "query_like_count",
        "query_word_max",
    )
    char_fields = ("input_json_chars", "string_total_chars", "string_max_chars", "string_total_lines", "string_max_lines")

    tokens.append(bucket_token("json_char_count", int(stats["input_json_chars"]), CHAR_BUCKETS))
    tokens.append(bucket_token("json_depth", int(stats["max_depth"]), DEPTH_BUCKETS))
    for field in count_fields:
        tokens.append(bucket_token(field, int(stats[field]), COUNT_BUCKETS))
    for field in char_fields:
        tokens.append(bucket_token(field, int(stats[field]), CHAR_BUCKETS))
    for marker in sorted(stats["value_markers"]):
        tokens.append(f"value_marker_{marker}")
    return tokens


def native_input_features(input_raw: object, tool: str, tool_identity: str) -> dict[str, object]:
    stats = empty_native_stats()
    stats["input_json_chars"] = len(canonical_json(input_raw))
    walk_native_input(input_raw, stats)

    identity = tool_identity_feature(tool, tool_identity)
    tokens = native_tokens(stats, identity)
    stats["value_markers"] = sorted(stats["value_markers"])

    return {
        "kind": "structured_input",
        "tool_identity_mode": tool_identity,
        "tool_identity": identity,
        "stats": stats,
        "tokens": tokens,
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


def row_signature(payload: dict[str, object]) -> str:
    payload_json = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload_json.encode("utf-8")).hexdigest()


def build_shell_row(record: dict[str, object], blocked_chars: int, blocked_lines: int) -> dict[str, object] | None:
    command = extract_command(record)
    if command is None:
        return None

    input_raw = record.get("input_raw")
    workdir = extract_workdir(input_raw)

    features = command_features(command, input_raw)
    features["kind"] = "shell_command"

    return {
        "schema_version": 2,
        "tool_family": "shell",
        "tool": "bash",
        "input": {
            "command": command,
            **({"workdir": workdir} if workdir is not None else {}),
        },
        "features": features,
        "label": make_label(record, blocked_chars, blocked_lines),
        "source": record.get("source") or {},
    }


def build_native_row(
    record: dict[str, object],
    blocked_chars: int,
    blocked_lines: int,
    tool_identity: str,
) -> dict[str, object] | None:
    tool = record.get("tool")
    if not isinstance(tool, str) or not tool.strip():
        return None

    input_raw = record.get("input_raw")
    if input_raw is None:
        input_raw = {}

    features = native_input_features(input_raw, tool.strip(), tool_identity)
    stats = features["stats"]
    input_summary = {
        "json_chars": stats["input_json_chars"],
        "field_count": stats["field_count"],
        "max_depth": stats["max_depth"],
    }

    return {
        "schema_version": 2,
        "tool_family": "native",
        "tool": tool.strip(),
        "input": input_summary,
        "features": features,
        "label": make_label(record, blocked_chars, blocked_lines),
        "source": record.get("source") or {},
    }


def signature_for_row(row: dict[str, object], input_raw: object) -> str:
    input_data = row.get("input") if isinstance(row.get("input"), dict) else {}
    payload = {
        "tool_family": row.get("tool_family"),
        "tool": row.get("tool"),
        "input": input_data,
        "input_raw": input_raw if row.get("tool_family") != "shell" else None,
    }
    return row_signature(payload)


def build_rows(
    records: Iterable[dict[str, object]],
    blocked_chars: int,
    blocked_lines: int,
    dedupe: bool,
    tool_identity: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()

    for record in records:
        input_raw = record.get("input_raw")
        if record.get("tool_family") == "shell" or record.get("tool") == "bash":
            row = build_shell_row(record, blocked_chars, blocked_lines)
        else:
            row = build_native_row(record, blocked_chars, blocked_lines, tool_identity)
        if row is None:
            continue

        signature = signature_for_row(row, input_raw)
        if dedupe and signature in seen:
            continue
        seen.add(signature)

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
    by_family: dict[str, dict[str, int]] = {}
    for row in rows:
        family = str(row.get("tool_family") or "unknown")
        if family not in by_family:
            by_family[family] = {"rows": 0, "blocked": 0, "allowed": 0}
        by_family[family]["rows"] += 1
        if row["label"]["blocked"]:
            by_family[family]["blocked"] += 1
        else:
            by_family[family]["allowed"] += 1
    return {
        "rows": len(rows),
        "blocked": blocked,
        "allowed": len(rows) - blocked,
        "by_family": by_family,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    rows = build_rows(
        read_jsonl(args.input),
        blocked_chars=args.blocked_chars,
        blocked_lines=args.blocked_lines,
        dedupe=args.dedupe,
        tool_identity=args.tool_identity,
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
        "tool_identity": args.tool_identity,
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
