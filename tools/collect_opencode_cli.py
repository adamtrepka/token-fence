from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Iterable


DB_FILENAMES = ("opencode.db", "opencode-prod.db")
DB_ENV_VARS = (
    "OPENCODE_DB",
    "OPENCODE_DB_PATH",
    "OPENCODE_DATABASE_PATH",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Collect OpenCode CLI calls from the local SQLite database."
    )
    parser.add_argument(
        "--db",
        type=Path,
        help="Path to the OpenCode SQLite database file or its containing directory.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        type=Path,
        default=[],
        help="Additional directory to search recursively for OpenCode databases.",
    )
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help="Tool name to collect. Repeatable. Defaults to all tools.",
    )
    parser.add_argument(
        "--exclude-tool",
        action="append",
        default=[],
        help="Tool name to exclude. Repeatable.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("opencode-tool-calls.jsonl"),
        help="Write JSONL to this file. Defaults to ./opencode-tool-calls.jsonl.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Write JSONL to stdout instead of a file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after emitting this many records.",
    )
    return parser


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = str(path.resolve()) if path.exists() else str(path)
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def normalize_path(path: Path) -> Path:
    return path.expanduser()


def candidate_db_files_from_dir(directory: Path) -> list[Path]:
    candidates: list[Path] = []
    for name in DB_FILENAMES:
        direct = directory / name
        if direct.is_file():
            candidates.append(direct)
    return candidates


def recursive_db_files(root: Path) -> list[Path]:
    if not root.exists() or not root.is_dir():
        return []
    matches: list[Path] = []
    for name in DB_FILENAMES:
        matches.extend(path for path in root.rglob(name) if path.is_file())
    return matches


def default_search_roots() -> list[Path]:
    roots: list[Path] = []
    home = Path.home()
    env_roots = [os.environ.get("APPDATA"), os.environ.get("LOCALAPPDATA")]
    for raw in env_roots:
        if raw:
            roots.append(Path(raw) / "opencode")
    roots.extend(
        [
            home / ".local" / "share" / "opencode",
            home / ".config" / "opencode",
            home / "AppData" / "Roaming" / "opencode",
            home / "AppData" / "Local" / "opencode",
        ]
    )
    return unique_paths(roots)


def resolve_db_path(explicit: Path | None, search_roots: list[Path]) -> Path:
    if explicit is not None:
        explicit = normalize_path(explicit)
        if explicit.is_file():
            return explicit
        if explicit.is_dir():
            candidates = candidate_db_files_from_dir(explicit)
            if candidates:
                return pick_newest(candidates)
        raise FileNotFoundError(f"OpenCode database not found at: {explicit}")

    for env_name in DB_ENV_VARS:
        raw = os.environ.get(env_name)
        if not raw:
            continue
        candidate = normalize_path(Path(raw))
        if candidate.is_file():
            return candidate
        if candidate.is_dir():
            matches = candidate_db_files_from_dir(candidate)
            if matches:
                return pick_newest(matches)

    candidates: list[Path] = []
    for root in unique_paths(search_roots + default_search_roots()):
        candidates.extend(candidate_db_files_from_dir(root))
        candidates.extend(recursive_db_files(root))

    candidates = unique_paths(candidates)
    if candidates:
        return pick_newest(candidates)

    searched = [str(path) for path in unique_paths(search_roots + default_search_roots())]
    raise FileNotFoundError(
        "Could not find an OpenCode database. Provide --db or set one of: "
        + ", ".join(DB_ENV_VARS)
        + ". Searched roots: "
        + "; ".join(searched)
    )


def pick_newest(paths: list[Path]) -> Path:
    return max(paths, key=lambda path: path.stat().st_mtime)


def dig(mapping: object, *keys: str) -> object | None:
    current = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def as_text(value: object | None) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def count_text(value: object | None) -> tuple[int, int]:
    text = as_text(value)
    if not text:
        return 0, 0
    return len(text.splitlines()), len(text)


def pick_output_text(state: dict[str, object]) -> tuple[object | None, str]:
    metadata = state.get("metadata")
    metadata_obj = metadata if isinstance(metadata, dict) else {}

    candidates: list[tuple[str, object | None]] = [
        ("state.output", state.get("output")),
        ("metadata.output", metadata_obj.get("output")),
        ("metadata.preview", metadata_obj.get("preview")),
        ("metadata.error", metadata_obj.get("error")),
        ("metadata.stderr", metadata_obj.get("stderr")),
    ]
    for source, value in candidates:
        if value not in (None, ""):
            return value, source
    return None, ""


def extract_records(
    db_path: Path,
    tools: list[str],
    exclude_tools: list[str],
    limit: int | None,
) -> Iterable[dict[str, object]]:
    tool_filter = ""
    params: list[object] = []
    if tools:
        placeholders = ", ".join("?" for _ in tools)
        tool_filter = f"AND json_extract(data, '$.tool') IN ({placeholders})"
        params.extend(tools)

    exclude_filter = ""
    if exclude_tools:
        placeholders = ", ".join("?" for _ in exclude_tools)
        exclude_filter = f"AND json_extract(data, '$.tool') NOT IN ({placeholders})"
        params.extend(exclude_tools)

    query = f"""
        SELECT id, message_id, time_created, data
        FROM part
        WHERE json_extract(data, '$.type') = 'tool'
          AND coalesce(json_extract(data, '$.state.status'), '') = 'completed'
          {tool_filter}
          {exclude_filter}
        ORDER BY time_created ASC
    """

    with sqlite3.connect(db_path) as connection:
        connection.row_factory = sqlite3.Row
        emitted = 0
        for row in connection.execute(query, params):
            try:
                payload = json.loads(row["data"])
            except json.JSONDecodeError:
                continue

            state = payload.get("state")
            if not isinstance(state, dict):
                continue

            tool = payload.get("tool")
            if not isinstance(tool, str) or not tool.strip():
                continue

            command = dig(payload, "state", "input", "command")
            input_raw = state.get("input")
            metadata_raw = state.get("metadata") if isinstance(state.get("metadata"), dict) else {}

            output_value, output_source = pick_output_text(state)
            output_lines, output_chars = count_text(output_value)

            stderr_value = dig(state, "stderr")
            if stderr_value is None:
                stderr_value = dig(metadata_raw, "stderr")
            if stderr_value is None:
                stderr_value = dig(metadata_raw, "error")
            stderr_lines, stderr_chars = count_text(stderr_value)

            if isinstance(command, str) and command.strip():
                command = command.strip()
            else:
                command = None

            tool_family = "shell" if tool == "bash" else "native"

            yield {
                "schema_version": 2,
                "tool_family": tool_family,
                "tool": tool,
                "status": state.get("status"),
                "command": command,
                "input_raw": input_raw,
                "metadata_raw": metadata_raw,
                "output_source": output_source,
                "output_lines": output_lines,
                "output_chars": output_chars,
                "stderr_lines": stderr_lines,
                "stderr_chars": stderr_chars,
                "total_lines": output_lines + stderr_lines,
                "total_chars": output_chars + stderr_chars,
                "source": {
                    "part_id": row["id"],
                    "message_id": row["message_id"],
                    "time_created": row["time_created"],
                },
            }

            emitted += 1
            if limit is not None and emitted >= limit:
                break


def open_output_stream(path: Path | None):
    if path is None:
        return nullcontext(sys.stdout)
    path = normalize_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path.open("w", encoding="utf-8", newline="\n")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        db_path = resolve_db_path(args.db, args.search_root)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    print(f"Using OpenCode database: {db_path}", file=sys.stderr)

    records = extract_records(db_path, args.tool, args.exclude_tool, args.limit)

    emitted = 0
    output_path = None if args.stdout else args.output
    with open_output_stream(output_path) as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            emitted += 1

    if args.stdout:
        print(f"Exported {emitted} records to stdout.", file=sys.stderr)
    else:
        print(f"Exported {emitted} records to {args.output}.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
