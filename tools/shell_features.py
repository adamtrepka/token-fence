from __future__ import annotations

import re
from pathlib import Path

CountBuckets = list[tuple[int, str]]

TOKEN_RE = re.compile(
    r"&&|\|\||>>|<<|[|;&<>()]|"
    r"`[^`]*`|"
    r'"(?:[^"\\]|\\.)*"|'
    r"'(?:[^'\\]|\\.)*'|"
    r"\$\((?:[^()\\]|\\.)*\)|"
    r"\$\{[^{}]*\}|"
    r"[^\s|;&<>()]+"
)

ASSIGNMENT_RE = re.compile(
    r"^\s*(?:"
    r"(?:\$env:[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)\s*=\s*"
    r"(?:'[^']*'|\"[^\"]*\"|[^;|&\s]+)\s*[;&]?\s*)+",
    re.I,
)
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9-]{1,30}$")
IDENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]{2,}$")
NUMBER_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
VERSION_RE = re.compile(r"^\d+(?:\.\d+){1,4}$")
HEX_RE = re.compile(r"^(?:0x)?[0-9a-fA-F]{8,}$")
UUID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
URL_RE = re.compile(r"^https?://", re.I)
WINDOWS_PATH_RE = re.compile(r"(?i)^[a-z]:[\\/]|^\\\\")
POSIX_PATH_RE = re.compile(r"^(?:/|\.\.?/|~[/\\]?)")
GLOB_RE = re.compile(r"(?<!\\)[*?[]")

GENERIC_INTENTS = {
    "help",
    "version",
    "list",
    "show",
    "search",
    "find",
    "status",
    "diff",
    "test",
    "build",
    "run",
    "commit",
    "add",
    "restore",
    "inspect",
    "check",
    "query",
    "open",
    "create",
    "install",
    "update",
    "remove",
    "delete",
    "export",
    "import",
    "debug",
    "view",
    "apply",
    "generate",
}

MAX_BUCKET = 10**9

COMMAND_CHAR_BUCKETS: CountBuckets = [
    (0, "0"),
    (80, "1_80"),
    (240, "81_240"),
    (800, "241_800"),
    (2000, "801_2000"),
    (MAX_BUCKET, "2001_plus"),
]
COMMAND_LINE_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (10, "4_10"), (MAX_BUCKET, "11_plus")]
CHAIN_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (MAX_BUCKET, "4_plus")]
PIPE_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (MAX_BUCKET, "2_plus")]
REDIRECT_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (MAX_BUCKET, "4_plus")]
SUBSTITUTION_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (MAX_BUCKET, "2_plus")]
PAREN_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (MAX_BUCKET, "4_plus")]
TOKEN_BUCKETS: CountBuckets = [
    (1, "1"),
    (3, "2_3"),
    (6, "4_6"),
    (10, "7_10"),
    (25, "11_25"),
    (MAX_BUCKET, "26_plus"),
]
FLAG_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (6, "4_6"), (MAX_BUCKET, "7_plus")]
PATH_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (2, "2"), (4, "3_4"), (MAX_BUCKET, "5_plus")]
URL_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (MAX_BUCKET, "2_plus")]
NUMBER_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (3, "2_3"), (MAX_BUCKET, "4_plus")]
GLOB_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (MAX_BUCKET, "2_plus")]
QUOTED_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (2, "2"), (MAX_BUCKET, "3_plus")]
PARAM_EXPANSION_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (MAX_BUCKET, "2_plus")]
OP_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (2, "2"), (4, "3_4"), (MAX_BUCKET, "5_plus")]
IDENTIFIER_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (2, "2"), (4, "3_4"), (MAX_BUCKET, "5_plus")]
CHAIN_DENSITY_BUCKETS: CountBuckets = [(0, "0"), (1, "1"), (2, "2"), (MAX_BUCKET, "3_plus")]
CWD_DEPTH_BUCKETS: CountBuckets = [
    (0, "0"),
    (1, "1"),
    (2, "2"),
    (4, "3_4"),
    (7, "5_7"),
    (MAX_BUCKET, "8_plus"),
]
OUTPUT_LIMITER_SIZE_BUCKETS: CountBuckets = [(10, "1_10"), (100, "11_100"), (1000, "101_1000"), (MAX_BUCKET, "1001_plus")]
FLAG_NAME_MAX_LEN = 48
NUMERIC_VALUE_BUCKETS: CountBuckets = [(0, "0"), (10, "1_10"), (100, "11_100"), (1000, "101_1000"), (MAX_BUCKET, "1001_plus")]

LIMITER_FLAG_HINTS = {"limit", "max", "max_count", "max_results", "max_items", "page_size", "first", "last", "tail", "count", "top", "take"}
QUIET_FLAG_HINTS = {"quiet", "silent", "no_progress", "terse"}
EXPANDER_FLAG_HINTS = {"verbose", "debug", "trace", "all", "recursive", "recurse", "paginate", "follow"}
FORMAT_FLAG_HINTS = {"json", "yaml", "xml", "format", "output"}


COUNT_KEYS = (
    "flag",
    "path",
    "url",
    "number",
    "glob",
    "quoted",
    "command_subst",
    "param_expansion",
    "op",
    "identifier",
    "chain",
    "pipe",
    "redirect",
    "paren",
)

OUTPUT_LIMITER_SILENT_SHORT_HEADS = {"curl", "npm"}


def _int_from_token(token: str) -> int | None:
    stripped = _strip_quotes(token).strip()
    if re.fullmatch(r"\d+", stripped):
        return int(stripped)
    return None


def _flag_value(token: str, flag: str) -> int | None:
    lowered = token.lower()
    flag_lower = flag.lower()
    if lowered.startswith(f"{flag_lower}="):
        return _int_from_token(token[len(flag) + 1 :])
    if lowered.startswith(flag_lower) and len(token) > len(flag):
        return _int_from_token(token[len(flag) :])
    return None


def _append_unique(tokens: list[str], seen: set[str], token: str) -> None:
    if token and token not in seen:
        seen.add(token)
        tokens.append(token)


def _normalize_feature_name(value: str, max_len: int = FLAG_NAME_MAX_LEN) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    normalized = normalized[:max_len]
    return normalized or "empty"


def _flag_value_bucket(value: int) -> str:
    return _bucket_tokens("flag_numeric_value", abs(int(value)), NUMERIC_VALUE_BUCKETS)


def _looks_like_short_flag_group(body: str) -> bool:
    return bool(
        body.isalpha()
        and 2 <= len(body) <= 3
        and not re.search(r"[aeiouy]", body, re.I)
    )


def _looks_like_attached_short_value(body: str) -> bool:
    return bool(len(body) > 1 and body[0].isalpha() and not body[1:].isdigit())


def _flag_control_tokens(normalized: str) -> list[str]:
    tokens: list[str] = []
    parts = set(normalized.split("_"))

    def matches(hints: set[str]) -> bool:
        return bool(parts & hints or normalized in hints)

    if matches(LIMITER_FLAG_HINTS):
        tokens.append("output_limit_flag")
        tokens.append("output_limiter_present")
    if matches(QUIET_FLAG_HINTS):
        tokens.append("output_quiet_flag")
        if "silent" in parts:
            tokens.append("output_silent_flag")
        tokens.append("output_limiter_present")
    if matches(EXPANDER_FLAG_HINTS):
        tokens.append("output_expander_flag")
    if matches(FORMAT_FLAG_HINTS):
        tokens.append("output_format_flag")
        if normalized in {"json", "yaml", "xml", "format", "output"}:
            tokens.append("output_structured_output_flag")
    if normalized == "output" or "structured_output" in normalized:
        tokens.append("output_structured_output_flag")
    return tokens


def _flag_numeric_value(raw_tokens: list[str], index: int, token: str, flag_name: str) -> int | None:
    lowered = token.lower()
    flag_lower = flag_name.lower()
    if lowered.startswith(f"{flag_lower}="):
        return _int_from_token(token[len(flag_name) + 1 :])
    if lowered.startswith(f"{flag_lower}:"):
        return _int_from_token(token[len(flag_name) + 1 :])
    if lowered == flag_lower and index + 1 < len(raw_tokens):
        return _int_from_token(raw_tokens[index + 1])
    return None


def flag_features(raw_tokens: list[str]) -> list[str]:
    features: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        _append_unique(features, seen, token)

    for index, token in enumerate(raw_tokens):
        stripped = _strip_quotes(token).strip()
        if not stripped.startswith("-") or stripped in {"-", "--"}:
            continue

        lowered = stripped.lower()
        if lowered.startswith("--"):
            flag_name = re.split(r"[=:]", stripped[2:], 1)[0]
            normalized = _normalize_feature_name(flag_name)
            add(f"flag_name_{normalized}")
            add("flag_long")
            add("flag")
            for extra in _flag_control_tokens(normalized):
                add(extra)
            value = _flag_numeric_value(raw_tokens, index, stripped, f"--{flag_name}")
            if value is not None:
                add(_flag_value_bucket(value))
                if "output_limit_flag" in features or "output_limiter_present" in features:
                    add(_bucket_tokens("output_limiter_size", abs(int(value)), OUTPUT_LIMITER_SIZE_BUCKETS))
            continue

        body = stripped[1:]
        if len(body) == 1:
            short = body.lower()
            add(f"flag_short_{short}")
            add("flag")
            add("flag_short")
            if short == "q":
                add("output_quiet_flag")
            elif short == "v":
                add("output_expander_flag")
            value = _int_from_token(raw_tokens[index + 1]) if index + 1 < len(raw_tokens) else None
            if value is not None and short == "n":
                add(_flag_value_bucket(value))
            continue

        if _looks_like_short_flag_group(body):
            add("flag_short_group")
            add("flag")
            for short in body.lower():
                add(f"flag_short_{short}")
                if short == "q":
                    add("output_quiet_flag")
                elif short == "v":
                    add("output_expander_flag")
            continue

        if _looks_like_attached_short_value(body):
            short = body[0].lower()
            add(f"flag_short_{short}")
            add("flag")
            add("flag_short")
            continue

        short_match = re.fullmatch(r"([A-Za-z])(?:=)?([+-]?\d+)", body)
        if short_match:
            short = short_match.group(1).lower()
            value = _int_from_token(short_match.group(2))
            add(f"flag_short_{short}")
            add("flag")
            add("flag_short")
            if short == "q":
                add("output_quiet_flag")
            elif short == "v":
                add("output_expander_flag")
            if value is not None:
                add(_flag_value_bucket(value))
            continue

        flag_name = re.split(r"[=:]", body, 1)[0]
        normalized = _normalize_feature_name(flag_name)
        add(f"flag_name_{normalized}")
        add("flag_long")
        add("flag")
        for extra in _flag_control_tokens(normalized):
            add(extra)
        value = _flag_numeric_value(raw_tokens, index, stripped, f"-{flag_name}")
        if value is not None:
            add(_flag_value_bucket(value))
            if "output_limit_flag" in features or "output_limiter_present" in features:
                add(_bucket_tokens("output_limiter_size", abs(int(value)), OUTPUT_LIMITER_SIZE_BUCKETS))

    return features


def output_limiter_features(raw_tokens: list[str], head: str, subcmd: str) -> list[str]:
    features: list[str] = []
    seen: set[str] = set()

    def add(token: str) -> None:
        _append_unique(features, seen, token)

    if head == "head":
        add("output_limiter_present")
        add("output_limiter_head")
    elif head == "tail":
        add("output_limiter_present")
        add("output_limiter_tail")

    if head == "select-object" and any(token.lower() in {"-first", "--first"} for token in raw_tokens):
        add("output_limiter_present")
        add("output_limiter_select_first")

    if head == "git" and subcmd == "log":
        add("output_limiter_present")
        for index, token in enumerate(raw_tokens):
            value = _flag_value(token, "-n")
            if value is None and token.lower() == "-n" and index + 1 < len(raw_tokens):
                value = _int_from_token(raw_tokens[index + 1])
            if value is not None:
                add("output_limiter_max_count")
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))

    if head in {"docker", "kubectl"} and subcmd == "logs":
        add("output_limiter_present")
        for index, token in enumerate(raw_tokens):
            value = _flag_value(token, "--tail")
            if value is None and token.lower() == "--tail" and index + 1 < len(raw_tokens):
                value = _int_from_token(raw_tokens[index + 1])
            if value is not None:
                add("output_limiter_tail_flag")
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))

    if head == "rg":
        found_limiter = False
        for index, token in enumerate(raw_tokens):
            lowered = token.lower()
            value = _flag_value(token, "-m")
            if value is None and lowered in {"-m", "--max-count"} and index + 1 < len(raw_tokens):
                value = _int_from_token(raw_tokens[index + 1])
            if value is None and lowered.startswith("--max-count="):
                value = _int_from_token(token.split("=", 1)[1])
            if value is not None:
                found_limiter = True
                add("output_limiter_max_count")
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))
        if found_limiter:
            add("output_limiter_present")

    for index, token in enumerate(raw_tokens):
        lowered = token.lower()
        if lowered in {"--quiet", "-q"}:
            add("output_quiet_flag")
            add("output_limiter_present")
        elif lowered == "--silent":
            add("output_silent_flag")
            add("output_limiter_present")
        elif lowered == "-s" and head in OUTPUT_LIMITER_SILENT_SHORT_HEADS:
            add("output_silent_short_flag")
            add("output_limiter_present")

        if lowered == "-first" and head == "select-object" and index + 1 < len(raw_tokens):
            value = _int_from_token(raw_tokens[index + 1])
            if value is not None:
                add("output_limiter_present")
                add("output_limiter_select_first")
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))

        if head in {"head", "tail"} and lowered == "-n" and index + 1 < len(raw_tokens):
            value = _int_from_token(raw_tokens[index + 1])
            if value is not None:
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))

        if head in {"head", "tail"} and lowered.startswith("-n"):
            value = _flag_value(token, "-n")
            if value is not None:
                add(_bucket_tokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS))

    return features

def _strip_quotes(token: str) -> str:
    return token.strip('"\'`')


def _strip_assignment_prefix(command: str) -> str:
    current = command.strip()
    while True:
        updated = ASSIGNMENT_RE.sub("", current, count=1)
        if updated == current:
            break
        current = updated.strip()
    current = re.sub(r"^(?:export|set|env)\s+", "", current, flags=re.I).strip()
    return current


def _tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _normalize_executable(token: str) -> str:
    raw = _strip_quotes(token).strip()
    raw = raw.replace("\\", "/")
    if "/" in raw:
        raw = raw.rsplit("/", 1)[-1]
    raw = re.sub(r"(?i)\.(?:exe|cmd|bat|ps1|sh)$", "", raw)
    return raw.lower()


def _bucket_length(text: str) -> str:
    length = len(text)
    if length <= 20:
        return "short"
    if length <= 80:
        return "med"
    if length <= 240:
        return "long"
    return "xlong"


def _classify_payload(text: str, prefix: str) -> str:
    stripped = text.strip()
    if not stripped:
        return f"{prefix}_empty"
    if URL_RE.search(stripped):
        return f"{prefix}_url"
    if WINDOWS_PATH_RE.search(stripped) or POSIX_PATH_RE.search(stripped) or "/" in stripped or "\\" in stripped:
        ext = Path(_strip_quotes(stripped)).suffix.lower().lstrip(".")
        suffix = f"_ext_{ext}" if ext else ""
        return f"{prefix}_path{suffix}"
    if any(marker in stripped for marker in ("\n", "SELECT ", "FROM ", "WHERE ", "INSERT ", "UPDATE ", "DELETE ")):
        return f"{prefix}_sql"
    if any(marker in stripped for marker in ("{", "}", "[", "]", ":", "\"")):
        return f"{prefix}_structured"
    return f"{prefix}_{_bucket_length(stripped)}"


def shellish_tokens(command: str) -> list[str]:
    command = _strip_assignment_prefix(command)
    tokens: list[str] = []
    for raw in _tokenize(command):
        if not raw:
            continue
        if raw[0] in {'"', "'", "`"} and len(raw) >= 2:
            tokens.append(_classify_payload(raw[1:-1], "quoted"))
            continue
        if raw.startswith("$(") and raw.endswith(")"):
            tokens.append(_classify_payload(raw[2:-1], "command_subst"))
            continue
        if raw.startswith("${") and raw.endswith("}"):
            tokens.append("param_expansion")
            continue
        tokens.append(raw)
    return tokens


def classify_token(token: str) -> list[str]:
    lowered = token.lower()
    if lowered in {"&&", "||", ";", "|", "(", ")", ">", ">>", "<", "<<"}:
        mapping = {
            "&&": "op_chain_and",
            "||": "op_chain_or",
            ";": "op_chain_seq",
            "|": "op_pipe",
            "(": "op_lparen",
            ")": "op_rparen",
            ">": "op_redirect_out",
            ">>": "op_redirect_append",
            "<": "op_redirect_in",
            "<<": "op_heredoc",
        }
        return [mapping[lowered]]

    if token.startswith("quoted_"):
        return [token]
    if token.startswith("command_subst_"):
        return [token]
    if token == "param_expansion":
        return ["param_expansion"]
    if lowered.startswith("-"):
        if lowered.startswith("--"):
            return ["flag_long", "flag"]
        if len(lowered) > 2 and lowered[1].isalpha() and _looks_like_short_flag_group(lowered[1:]):
            return ["flag_short_group", "flag"]
        return ["flag", "flag_short"]
    if URL_RE.match(token):
        return ["url"]
    if UUID_RE.match(token):
        return ["uuid"]
    if HEX_RE.match(token):
        return ["hex"]
    if VERSION_RE.match(token):
        return ["version_like"]
    if NUMBER_RE.match(token):
        return ["number"]
    if WINDOWS_PATH_RE.match(token) or POSIX_PATH_RE.match(token) or "/" in token or "\\" in token:
        ext = Path(_strip_quotes(token)).suffix.lower().lstrip(".")
        suffix = f"_ext_{ext}" if ext else ""
        return [f"path{suffix}", "path"]
    if GLOB_RE.search(token):
        return ["glob"]
    if IDENT_RE.match(token):
        if len(token) <= 15:
            return [f"word_{lowered}"]
        return ["identifier"]
    if WORD_RE.match(token):
        return [f"word_{lowered}"]
    return ["other_token"]


def head_and_subcommand(tokens: list[str]) -> tuple[str, str]:
    words: list[str] = []
    for token in tokens:
        if token in {"quoted", "command_subst", "param_expansion"}:
            continue
        if token.startswith("quoted_") or token.startswith("command_subst_"):
            continue
        if token.startswith("op_"):
            continue
        if token.startswith("flag"):
            continue
        if token.lower().startswith(("$", "=", ":")):
            continue
        words.append(token)
    if not words:
        return "", ""
    head = _normalize_executable(words[0])
    subcmd = ""
    for token in words[1:]:
        if WORD_RE.match(token) or IDENT_RE.match(token):
            subcmd = token.lower()
            break
    return head, subcmd


def count_bucket(value: int, buckets: CountBuckets) -> str:
    for upper, label in buckets:
        if value <= upper:
            return label
    return buckets[-1][1]


def _bucket_tokens(prefix: str, value: int, buckets: CountBuckets) -> str:
    return f"{prefix}_{count_bucket(value, buckets)}"


def _new_counts() -> dict[str, int]:
    return dict.fromkeys(COUNT_KEYS, 0)


def _append_count_bucket(classified: list[str], prefix: str, value: int, buckets: CountBuckets) -> None:
    classified.append(_bucket_tokens(prefix, value, buckets))


def build_model_text(command: str, workdir: str = "") -> str:
    raw_tokens = shellish_tokens(command)
    classified: list[str] = []
    counts = _new_counts()

    for token in raw_tokens:
        labels = classify_token(token)
        classified.extend(labels)
        for label in labels:
            if label in counts:
                counts[label] += 1
            if label.startswith("op_"):
                counts["op"] += 1
            if label in {"op_chain_and", "op_chain_or", "op_chain_seq"}:
                counts["chain"] += 1
            if label == "op_pipe":
                counts["pipe"] += 1
            if label in {"op_redirect_out", "op_redirect_append", "op_redirect_in", "op_heredoc"}:
                counts["redirect"] += 1
            if label in {"op_lparen", "op_rparen"}:
                counts["paren"] += 1

    head, subcmd = head_and_subcommand(raw_tokens)
    if head:
        classified.append(f"head_{head}")
    else:
        classified.append("head_empty")
    if subcmd:
        classified.append(f"subcmd_{subcmd}")
    else:
        classified.append("subcmd_none")

    for token in {head, subcmd}:
        if token in GENERIC_INTENTS:
            classified.append(f"intent_{token}")

    classified.extend(output_limiter_features(raw_tokens, head, subcmd))
    classified.extend(flag_features(raw_tokens))

    _append_count_bucket(classified, "command_char_count", len(command), COMMAND_CHAR_BUCKETS)
    _append_count_bucket(classified, "command_line_count", command.count("\n") + (1 if command.strip() else 0), COMMAND_LINE_BUCKETS)
    _append_count_bucket(classified, "chain_count", counts["chain"], CHAIN_BUCKETS)
    _append_count_bucket(classified, "pipe_count", counts["pipe"], PIPE_BUCKETS)
    _append_count_bucket(classified, "redirect_count", counts["redirect"], REDIRECT_BUCKETS)
    _append_count_bucket(classified, "substitution_count", counts["command_subst"], SUBSTITUTION_BUCKETS)
    _append_count_bucket(classified, "paren_count", counts["paren"], PAREN_BUCKETS)

    _append_count_bucket(classified, "token_count", len(raw_tokens), TOKEN_BUCKETS)
    _append_count_bucket(classified, "flag_count", counts["flag"], FLAG_BUCKETS)
    _append_count_bucket(classified, "path_count", counts["path"], PATH_BUCKETS)
    _append_count_bucket(classified, "url_count", counts["url"], URL_BUCKETS)
    _append_count_bucket(classified, "number_count", counts["number"], NUMBER_BUCKETS)
    _append_count_bucket(classified, "glob_count", counts["glob"], GLOB_BUCKETS)
    _append_count_bucket(classified, "quoted_count", counts["quoted"], QUOTED_BUCKETS)
    _append_count_bucket(classified, "command_subst_count", counts["command_subst"], SUBSTITUTION_BUCKETS)
    _append_count_bucket(classified, "param_expansion_count", counts["param_expansion"], PARAM_EXPANSION_BUCKETS)
    _append_count_bucket(classified, "op_count", counts["op"], OP_BUCKETS)
    _append_count_bucket(classified, "identifier_count", counts["identifier"], IDENTIFIER_BUCKETS)
    _append_count_bucket(classified, "chain_density", (counts["chain"] * 10) // max(1, len(raw_tokens)), CHAIN_DENSITY_BUCKETS)

    if workdir:
        depth = len([segment for segment in re.split(r"[\\/]+", workdir) if segment])
        _append_count_bucket(classified, "cwd_depth", depth, CWD_DEPTH_BUCKETS)
        classified.append(f"cwd_has_space_{int(' ' in workdir)}")
        classified.append(f"cwd_has_drive_{int(bool(re.match(r'(?i)^[a-z]:', workdir)))}")

    return " ".join(classified)
