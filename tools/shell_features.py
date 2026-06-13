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
        if len(lowered) > 2 and lowered[1].isalpha():
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
