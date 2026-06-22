import fs from "node:fs/promises";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import os from "node:os";
import { createHash } from "node:crypto";

const require = createRequire(import.meta.url);
const ort = require("onnxruntime-node");
const __dirname = path.dirname(fileURLToPath(import.meta.url));

const TOKEN_RE = /&&|\|\||>>|<<|[|;&<>()]|`[^`]*`|"(?:[^"\\]|\\.)*"|'(?:[^'\\]|\\.)*'|\$\((?:[^()\\]|\\.)*\)|\$\{[^{}]*\}|[^\s|;&<>()]+/g;
const ASSIGNMENT_RE = /^\s*(?:(?:\$env:[A-Za-z_][A-Za-z0-9_]*|[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?:'[^']*'|"[^"]*"|[^;|&\s]+)\s*[;&]?\s*)+/i;
const WORD_RE = /^[A-Za-z][A-Za-z0-9-]{1,30}$/;
const IDENT_RE = /^[A-Za-z_][A-Za-z0-9_-]{2,}$/;
const NUMBER_RE = /^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/;
const VERSION_RE = /^\d+(?:\.\d+){1,4}$/;
const HEX_RE = /^(?:0x)?[0-9a-fA-F]{8,}$/;
const UUID_RE = /^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$/;
const URL_RE = /^https?:\/\//i;
const WINDOWS_PATH_RE = /^(?:[a-z]:[\\/]|\\\\)/i;
const POSIX_PATH_RE = /^(?:\/|\.\.?\/|~[\/\\]?)/;
const GLOB_RE = /(?<!\\)(?:[*?]|\[)/;

const GENERIC_INTENTS = new Set([
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
]);

const MODEL_DIR = path.resolve(__dirname, "../../tools/model-shell");
const MODEL_PATH = path.join(MODEL_DIR, "model.onnx");
const THRESHOLD_PATH = path.join(MODEL_DIR, "threshold.json");
const MANIFEST_PATH = path.join(MODEL_DIR, "manifest.json");
const MAX_BUCKET = Number.MAX_SAFE_INTEGER;

const COMMAND_CHAR_BUCKETS = [
  [0, "0"],
  [80, "1_80"],
  [240, "81_240"],
  [800, "241_800"],
  [2000, "801_2000"],
  [MAX_BUCKET, "2001_plus"],
];
const COMMAND_LINE_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [10, "4_10"], [MAX_BUCKET, "11_plus"]];
const CHAIN_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [MAX_BUCKET, "4_plus"]];
const PIPE_BUCKETS = [[0, "0"], [1, "1"], [MAX_BUCKET, "2_plus"]];
const REDIRECT_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [MAX_BUCKET, "4_plus"]];
const SUBSTITUTION_BUCKETS = [[0, "0"], [1, "1"], [MAX_BUCKET, "2_plus"]];
const PAREN_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [MAX_BUCKET, "4_plus"]];
const TOKEN_BUCKETS = [
  [1, "1"],
  [3, "2_3"],
  [6, "4_6"],
  [10, "7_10"],
  [25, "11_25"],
  [MAX_BUCKET, "26_plus"],
];
const FLAG_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [6, "4_6"], [MAX_BUCKET, "7_plus"]];
const PATH_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [4, "3_4"], [MAX_BUCKET, "5_plus"]];
const URL_BUCKETS = [[0, "0"], [1, "1"], [MAX_BUCKET, "2_plus"]];
const NUMBER_BUCKETS = [[0, "0"], [1, "1"], [3, "2_3"], [MAX_BUCKET, "4_plus"]];
const GLOB_BUCKETS = [[0, "0"], [1, "1"], [MAX_BUCKET, "2_plus"]];
const QUOTED_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [MAX_BUCKET, "3_plus"]];
const PARAM_EXPANSION_BUCKETS = [[0, "0"], [1, "1"], [MAX_BUCKET, "2_plus"]];
const OP_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [4, "3_4"], [MAX_BUCKET, "5_plus"]];
const IDENTIFIER_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [4, "3_4"], [MAX_BUCKET, "5_plus"]];
const CHAIN_DENSITY_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [MAX_BUCKET, "3_plus"]];
const CWD_DEPTH_BUCKETS = [
  [0, "0"],
  [1, "1"],
  [2, "2"],
  [4, "3_4"],
  [7, "5_7"],
  [MAX_BUCKET, "8_plus"],
];
const OUTPUT_LIMITER_SIZE_BUCKETS = [[10, "1_10"], [100, "11_100"], [1000, "101_1000"], [MAX_BUCKET, "1001_plus"]];
const FLAG_NAME_MAX_LEN = 48;
const NUMERIC_VALUE_BUCKETS = [[0, "0"], [10, "1_10"], [100, "11_100"], [1000, "101_1000"], [MAX_BUCKET, "1001_plus"]];
const LIMITER_FLAG_HINTS = new Set(["limit", "max", "max_count", "max_results", "max_items", "page_size", "first", "last", "tail", "count", "top", "take"]);
const QUIET_FLAG_HINTS = new Set(["quiet", "silent", "no_progress", "terse"]);
const EXPANDER_FLAG_HINTS = new Set(["verbose", "debug", "trace", "all", "recursive", "recurse", "paginate", "follow"]);
const FORMAT_FLAG_HINTS = new Set(["json", "yaml", "xml", "format", "output"]);
const FIELD_NAME_LIMIT = 48;
const FIELD_NAME_TOKEN_LIMIT = 40;
const FIELD_LIMIT_VALUE_BUCKETS = [[10, "1_10"], [100, "11_100"], [1000, "101_1000"], [MAX_BUCKET, "1001_plus"]];
const FIELD_CLASS_HINTS = {
  field_output_limit: new Set(["limit", "max", "results", "first", "last", "tail", "count", "top", "take"]),
  field_pagination: new Set(["offset", "page", "cursor", "token"]),
  field_path: new Set(["path", "file", "directory", "dir", "glob", "pattern", "include", "exclude"]),
  field_query: new Set(["query", "search", "filter"]),
  field_url: new Set(["url", "uri", "link"]),
  field_format: new Set(["format", "json", "yaml", "xml", "output", "mime", "type"]),
  field_provider: new Set(["provider", "model", "engine", "tool"]),
  field_temperature: new Set(["temperature"]),
};

const NATIVE_COUNT_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [5, "3_5"], [10, "6_10"], [25, "11_25"], [50, "26_50"], [100, "51_100"], [250, "101_250"], [MAX_BUCKET, "251_plus"]];
const NATIVE_CHAR_BUCKETS = [[0, "0"], [80, "1_80"], [240, "81_240"], [800, "241_800"], [2000, "801_2000"], [8000, "2001_8000"], [32000, "8001_32000"], [MAX_BUCKET, "32001_plus"]];
const NATIVE_DEPTH_BUCKETS = [[0, "0"], [1, "1"], [2, "2"], [4, "3_4"], [8, "5_8"], [MAX_BUCKET, "9_plus"]];
const URL_VALUE_RE = /https?:\/\/\S+/gi;
const PATH_EXTENSION_RE = /\.[A-Za-z0-9]{1,10}(?:$|[?#])/;
const MARKDOWN_VALUE_RE = /(^|\n)\s{0,3}(?:#{1,6}\s|[-*+]\s|\d+\.\s|```|>)/;
const CODE_VALUE_RE = /```|=>|[{};]|\b(?:class|def|function|import|select|from|where)\b/i;
const GENERIC_VALUE_MARKERS = new Set(["bm25", "csv", "fit", "html", "json", "llm", "markdown", "md", "pdf", "png", "raw", "svg", "text", "xml", "yaml"]);
const OUTPUT_LIMITER_SILENT_SHORT_HEADS = new Set(["curl", "npm"]);
const NEVER_BLOCK_NATIVE_TOOLS = new Set([
  "apply_patch",
  "delegate",
  "delegation_list",
  "delegation_read",
  "edit",
  "glob",
  "grep",
  "multi_tool_use.parallel",
  "question",
  "read",
  "skill",
  "task",
  "todowrite",
  "write",
]);

let statePromise = null;
let runtimeConfig = {
  defaultAgent: "",
  agentModes: new Map(),
};
let pluginClient = null;
const sessionRoles = new Map();
const sessionAgents = new Map();
const BLOCKED_CALL_GUIDANCE =
  "Token Fence blocked this parent-agent call because it is likely to produce excessive context/output or is too broad. Do not retry or work around it in the parent. Delegate the task to an appropriate subagent using a compact, bounded prompt; ask it to return only the needed result.";

function stripQuotes(token) {
  return token.replace(/^["'`]+|["'`]+$/g, "");
}

function stripAssignmentPrefix(command) {
  let current = String(command || "").trim();
  while (true) {
    const updated = current.replace(ASSIGNMENT_RE, "").trim();
    if (updated === current) {
      break;
    }
    current = updated;
  }
  return current.replace(/^(?:export|set|env)\s+/i, "").trim();
}

function tokenize(text) {
  return String(text || "").match(TOKEN_RE) || [];
}

function normalizeExecutable(token) {
  let raw = stripQuotes(String(token || "")).trim().replace(/\\/g, "/");
  if (raw.includes("/")) {
    raw = raw.slice(raw.lastIndexOf("/") + 1);
  }
  return raw.replace(/\.(?:exe|cmd|bat|ps1|sh)$/i, "").toLowerCase();
}

function bucketLength(text) {
  const length = String(text || "").length;
  if (length <= 20) return "short";
  if (length <= 80) return "med";
  if (length <= 240) return "long";
  return "xlong";
}

function classifyPayload(text, prefix) {
  const stripped = String(text || "").trim();
  if (!stripped) return `${prefix}_empty`;
  if (URL_RE.test(stripped)) return `${prefix}_url`;
  if (WINDOWS_PATH_RE.test(stripped) || POSIX_PATH_RE.test(stripped) || stripped.includes("/") || stripped.includes("\\")) {
    const ext = path.extname(stripQuotes(stripped)).toLowerCase().replace(/^\./, "");
    const suffix = ext ? `_ext_${ext}` : "";
    return `${prefix}_path${suffix}`;
  }
  if (/(\n|SELECT |FROM |WHERE |INSERT |UPDATE |DELETE )/i.test(stripped)) return `${prefix}_sql`;
  if (/[{}\[\]:"]/.test(stripped)) return `${prefix}_structured`;
  return `${prefix}_${bucketLength(stripped)}`;
}

function shellishTokens(command) {
  const tokens = [];
  for (const raw of tokenize(stripAssignmentPrefix(command))) {
    if (!raw) continue;
    if ((raw[0] === '"' || raw[0] === "'" || raw[0] === "`") && raw.length >= 2) {
      tokens.push(classifyPayload(raw.slice(1, -1), "quoted"));
      continue;
    }
    if (raw.startsWith("$(") && raw.endsWith(")")) {
      tokens.push(classifyPayload(raw.slice(2, -1), "command_subst"));
      continue;
    }
    if (raw.startsWith("${") && raw.endsWith("}")) {
      tokens.push("param_expansion");
      continue;
    }
    tokens.push(raw);
  }
  return tokens;
}

function buildBlockedCallMessage({ toolLabel, agentRole, agentName, scoreText, thresholdText, detailLabel, detailText }) {
  return `Token Fence blocked ${toolLabel} for ${agentRole} agent${agentName ? ` (${agentName})` : ""}. ${BLOCKED_CALL_GUIDANCE} score=${scoreText}, threshold=${thresholdText}${detailText ? `, ${detailLabel}=${detailText}` : ""}`;
}

function classifyToken(token) {
  const lowered = String(token || "").toLowerCase();
  if (["&&", "||", ";", "|", "(", ")", ">", ">>", "<", "<<"].includes(lowered)) {
    const mapping = {
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
    };
    return [mapping[lowered]];
  }

  if (lowered.startsWith("quoted_")) return [token];
  if (lowered.startsWith("command_subst_")) return [token];
  if (token === "param_expansion") return ["param_expansion"];
  if (lowered.startsWith("-")) {
    if (lowered.startsWith("--")) return ["flag_long", "flag"];
    if (lowered.length > 2 && /[a-z]/i.test(lowered[1]) && looksLikeShortFlagGroup(lowered.slice(1))) return ["flag_short_group", "flag"];
    return ["flag", "flag_short"];
  }
  if (URL_RE.test(token)) return ["url"];
  if (UUID_RE.test(token)) return ["uuid"];
  if (HEX_RE.test(token)) return ["hex"];
  if (VERSION_RE.test(token)) return ["version_like"];
  if (NUMBER_RE.test(token)) return ["number"];
  if (WINDOWS_PATH_RE.test(token) || POSIX_PATH_RE.test(token) || token.includes("/") || token.includes("\\")) {
    const ext = path.extname(stripQuotes(token)).toLowerCase().replace(/^\./, "");
    const suffix = ext ? `_ext_${ext}` : "";
    return [`path${suffix}`, "path"];
  }
  if (GLOB_RE.test(token)) return ["glob"];
  if (IDENT_RE.test(token)) {
    if (token.length <= 15) return [`word_${lowered}`];
    return ["identifier"];
  }
  if (WORD_RE.test(token)) return [`word_${lowered}`];
  return ["other_token"];
}

function headAndSubcommand(tokens) {
  const words = [];
  for (const token of tokens) {
    if (token === "param_expansion") continue;
    if (token.startsWith("quoted_") || token.startsWith("command_subst_")) continue;
    if (token.startsWith("op_")) continue;
    if (token.startsWith("flag")) continue;
    if (/^[\$=:]/.test(String(token))) continue;
    words.push(token);
  }
  if (!words.length) return ["", ""];
  const head = normalizeExecutable(words[0]);
  let subcmd = "";
  for (const token of words.slice(1)) {
    if (WORD_RE.test(token) || IDENT_RE.test(token)) {
      subcmd = String(token).toLowerCase();
      break;
    }
  }
  return [head, subcmd];
}

function countBucket(value, buckets) {
  for (const [upper, label] of buckets) {
    if (value <= upper) {
      return label;
    }
  }
  return buckets[buckets.length - 1][1];
}

function bucketTokens(prefix, value, buckets) {
  return `${prefix}_${countBucket(value, buckets)}`;
}

function appendBucketToken(classified, prefix, value, buckets) {
  classified.push(bucketTokens(prefix, value, buckets));
}

function integerFromToken(token) {
  const stripped = stripQuotes(String(token || "").trim());
  return /^\d+$/.test(stripped) ? Number(stripped) : null;
}

function flagValue(token, flag) {
  const lowered = String(token || "").toLowerCase();
  const flagLower = String(flag || "").toLowerCase();
  if (lowered.startsWith(`${flagLower}=`)) {
    return integerFromToken(String(token).slice(flag.length + 1));
  }
  if (lowered.startsWith(flagLower) && String(token).length > flag.length) {
    return integerFromToken(String(token).slice(flag.length));
  }
  return null;
}

function appendUnique(classified, seen, token) {
  if (token && !seen.has(token)) {
    seen.add(token);
    classified.push(token);
  }
}

function outputLimiterFeatures(rawTokens, head, subcmd) {
  const classified = [];
  const seen = new Set();
  const add = (token) => appendUnique(classified, seen, token);

  if (head === "head") {
    add("output_limiter_present");
    add("output_limiter_head");
  } else if (head === "tail") {
    add("output_limiter_present");
    add("output_limiter_tail");
  }

  if (head === "select-object" && rawTokens.some((token) => ["-first", "--first"].includes(String(token).toLowerCase()))) {
    add("output_limiter_present");
    add("output_limiter_select_first");
  }

  if (head === "git" && subcmd === "log") {
    add("output_limiter_present");
    rawTokens.forEach((token, index) => {
      let value = flagValue(token, "-n");
      if (value === null && String(token).toLowerCase() === "-n" && index + 1 < rawTokens.length) {
        value = integerFromToken(rawTokens[index + 1]);
      }
      if (value !== null) {
        add("output_limiter_max_count");
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    });
  }

  if ((head === "docker" || head === "kubectl") && subcmd === "logs") {
    add("output_limiter_present");
    rawTokens.forEach((token, index) => {
      let value = flagValue(token, "--tail");
      if (value === null && String(token).toLowerCase() === "--tail" && index + 1 < rawTokens.length) {
        value = integerFromToken(rawTokens[index + 1]);
      }
      if (value !== null) {
        add("output_limiter_tail_flag");
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    });
  }

  if (head === "rg") {
    let foundLimiter = false;
    rawTokens.forEach((token, index) => {
      const lowered = String(token).toLowerCase();
      let value = flagValue(token, "-m");
      if (value === null && ["-m", "--max-count"].includes(lowered) && index + 1 < rawTokens.length) {
        value = integerFromToken(rawTokens[index + 1]);
      }
      if (value === null && lowered.startsWith("--max-count=")) {
        value = integerFromToken(String(token).split("=", 2)[1]);
      }
      if (value !== null) {
        foundLimiter = true;
        add("output_limiter_max_count");
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    });
    if (foundLimiter) {
      add("output_limiter_present");
    }
  }

  rawTokens.forEach((token, index) => {
    const lowered = String(token).toLowerCase();
    if (["--quiet", "-q"].includes(lowered)) {
      add("output_quiet_flag");
      add("output_limiter_present");
    } else if (lowered === "--silent") {
      add("output_silent_flag");
      add("output_limiter_present");
    } else if (lowered === "-s" && OUTPUT_LIMITER_SILENT_SHORT_HEADS.has(head)) {
      add("output_silent_short_flag");
      add("output_limiter_present");
    }

    if (lowered === "-first" && head === "select-object" && index + 1 < rawTokens.length) {
      const value = integerFromToken(rawTokens[index + 1]);
      if (value !== null) {
        add("output_limiter_present");
        add("output_limiter_select_first");
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    }

    if ((head === "head" || head === "tail") && lowered === "-n" && index + 1 < rawTokens.length) {
      const value = integerFromToken(rawTokens[index + 1]);
      if (value !== null) {
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    }

    if ((head === "head" || head === "tail") && lowered.startsWith("-n")) {
      const value = flagValue(token, "-n");
      if (value !== null) {
        add(bucketTokens("output_limiter_size", value, OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    }
  });

  return classified;
}

function numericFieldBucket(value) {
  return bucketTokens("field_limit_value", Math.abs(Number(value) || 0), FIELD_LIMIT_VALUE_BUCKETS);
}

function normalizeFieldName(value, maxLen = FIELD_NAME_LIMIT) {
  return normalizeFeatureName(value, maxLen);
}

function fieldClassesForName(name) {
  const normalized = normalizeFieldName(name);
  const parts = new Set(normalized.split("_").filter(Boolean));
  const classes = [];
  for (const [label, hints] of Object.entries(FIELD_CLASS_HINTS)) {
    if ([...parts].some((part) => hints.has(part)) || hints.has(normalized)) {
      classes.push(label);
    }
  }
  return classes;
}

function isLimitLikeField(name) {
  const normalized = normalizeFieldName(name);
  const parts = new Set(normalized.split("_").filter(Boolean));
  return [...parts].some((part) => FIELD_CLASS_HINTS.field_output_limit.has(part)) || normalized.endsWith("limit") || normalized.endsWith("count");
}

function buildModelText(command, workdir = "") {
  const rawTokens = shellishTokens(command);
  const classified = [];
  const counts = {
    flag: 0,
    path: 0,
    url: 0,
    number: 0,
    glob: 0,
    quoted: 0,
    command_subst: 0,
    param_expansion: 0,
    op: 0,
    identifier: 0,
    chain: 0,
    pipe: 0,
    redirect: 0,
    paren: 0,
  };

  for (const token of rawTokens) {
    const labels = classifyToken(token);
    classified.push(...labels);
    for (const label of labels) {
      if (Object.prototype.hasOwnProperty.call(counts, label)) {
        counts[label] += 1;
      }
      if (label.startsWith("op_")) counts.op += 1;
      if (label === "op_chain_and" || label === "op_chain_or" || label === "op_chain_seq") counts.chain += 1;
      if (label === "op_pipe") counts.pipe += 1;
      if (label === "op_redirect_out" || label === "op_redirect_append" || label === "op_redirect_in" || label === "op_heredoc") counts.redirect += 1;
      if (label === "op_lparen" || label === "op_rparen") counts.paren += 1;
    }
  }

  const [head, subcmd] = headAndSubcommand(rawTokens);
  classified.push(head ? `head_${head}` : "head_empty");
  classified.push(subcmd ? `subcmd_${subcmd}` : "subcmd_none");

  for (const token of new Set([head, subcmd])) {
    if (GENERIC_INTENTS.has(token)) {
      classified.push(`intent_${token}`);
    }
  }

  classified.push(...outputLimiterFeatures(rawTokens, head, subcmd));
  classified.push(...flagFeatures(rawTokens));

  appendBucketToken(classified, "command_char_count", String(command || "").length, COMMAND_CHAR_BUCKETS);
  appendBucketToken(classified, "command_line_count", (String(command || "").match(/\n/g) || []).length + (String(command || "").trim() ? 1 : 0), COMMAND_LINE_BUCKETS);
  appendBucketToken(classified, "chain_count", counts.chain, CHAIN_BUCKETS);
  appendBucketToken(classified, "pipe_count", counts.pipe, PIPE_BUCKETS);
  appendBucketToken(classified, "redirect_count", counts.redirect, REDIRECT_BUCKETS);
  appendBucketToken(classified, "substitution_count", counts.command_subst, SUBSTITUTION_BUCKETS);
  appendBucketToken(classified, "paren_count", counts.paren, PAREN_BUCKETS);

  appendBucketToken(classified, "token_count", rawTokens.length, TOKEN_BUCKETS);
  appendBucketToken(classified, "flag_count", counts.flag, FLAG_BUCKETS);
  appendBucketToken(classified, "path_count", counts.path, PATH_BUCKETS);
  appendBucketToken(classified, "url_count", counts.url, URL_BUCKETS);
  appendBucketToken(classified, "number_count", counts.number, NUMBER_BUCKETS);
  appendBucketToken(classified, "glob_count", counts.glob, GLOB_BUCKETS);
  appendBucketToken(classified, "quoted_count", counts.quoted, QUOTED_BUCKETS);
  appendBucketToken(classified, "command_subst_count", counts.command_subst, SUBSTITUTION_BUCKETS);
  appendBucketToken(classified, "param_expansion_count", counts.param_expansion, PARAM_EXPANSION_BUCKETS);
  appendBucketToken(classified, "op_count", counts.op, OP_BUCKETS);
  appendBucketToken(classified, "identifier_count", counts.identifier, IDENTIFIER_BUCKETS);
  appendBucketToken(classified, "chain_density", Math.floor((counts.chain * 10) / Math.max(1, rawTokens.length)), CHAIN_DENSITY_BUCKETS);

  if (workdir) {
    const depth = String(workdir).split(/[\\/]+/).filter(Boolean).length;
    appendBucketToken(classified, "cwd_depth", depth, CWD_DEPTH_BUCKETS);
    classified.push(`cwd_has_space_${Number(String(workdir).includes(" "))}`);
    classified.push(`cwd_has_drive_${Number(/^[a-z]:/i.test(String(workdir)))}`);
  }

  return classified.join(" ");
}

function stableJsonValue(value) {
  if (Array.isArray(value)) return value.map(stableJsonValue);
  if (value && typeof value === "object") {
    const result = {};
    for (const key of Object.keys(value).sort()) {
      const item = value[key];
      if (typeof item !== "undefined") {
        result[key] = stableJsonValue(item);
      }
    }
    return result;
  }
  return typeof value === "undefined" ? null : value;
}

function canonicalJson(value) {
  try {
    return JSON.stringify(stableJsonValue(value)) || "null";
  } catch {
    return "null";
  }
}

function normalizeIdentifier(value) {
  const normalized = String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return (normalized || "empty").slice(0, 80);
}

function normalizeFeatureName(value, maxLen = FLAG_NAME_MAX_LEN) {
  const normalized = String(value || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_+|_+$/g, "");
  return (normalized || "empty").slice(0, maxLen);
}

function flagValueBucket(value) {
  return bucketTokens("flag_numeric_value", Math.abs(Number(value) || 0), NUMERIC_VALUE_BUCKETS);
}

function looksLikeShortFlagGroup(body) {
  return /^[A-Za-z]{2,3}$/.test(body) && !/[aeiouy]/i.test(body);
}

function looksLikeAttachedShortValue(body) {
  return body.length > 1 && /^[A-Za-z]/.test(body[0]) && !/^\d+$/.test(body.slice(1));
}

function flagControlTokens(normalized) {
  const parts = new Set(String(normalized || "").split("_").filter(Boolean));
  const tokens = [];
  if (LIMITER_FLAG_HINTS.has(normalized) || [...parts].some((part) => LIMITER_FLAG_HINTS.has(part))) {
    tokens.push("output_limit_flag", "output_limiter_present");
  }
  if (QUIET_FLAG_HINTS.has(normalized) || [...parts].some((part) => QUIET_FLAG_HINTS.has(part))) {
    tokens.push("output_quiet_flag", "output_limiter_present");
    if (parts.has("silent")) tokens.push("output_silent_flag");
  }
  if (EXPANDER_FLAG_HINTS.has(normalized) || [...parts].some((part) => EXPANDER_FLAG_HINTS.has(part))) {
    tokens.push("output_expander_flag");
  }
  if (FORMAT_FLAG_HINTS.has(normalized) || [...parts].some((part) => FORMAT_FLAG_HINTS.has(part))) {
    tokens.push("output_format_flag");
    if (["json", "yaml", "xml", "format", "output"].includes(normalized)) tokens.push("output_structured_output_flag");
  }
  if (normalized === "output" || normalized.includes("structured_output")) {
    tokens.push("output_structured_output_flag");
  }
  return tokens;
}

function flagNumericValue(rawTokens, index, token, flagName) {
  const lowered = String(token || "").toLowerCase();
  const flagLower = String(flagName || "").toLowerCase();
  if (lowered.startsWith(`${flagLower}=`) || lowered.startsWith(`${flagLower}:`)) {
    return integerFromToken(String(token).slice(flagName.length + 1));
  }
  if (lowered === flagLower && index + 1 < rawTokens.length) {
    return integerFromToken(rawTokens[index + 1]);
  }
  return null;
}

function flagFeatures(rawTokens) {
  const features = [];
  const seen = new Set();
  const add = (token) => appendUnique(features, seen, token);

  rawTokens.forEach((token, index) => {
    const stripped = stripQuotes(String(token || "")).trim();
    if (!stripped.startsWith("-") || stripped === "-" || stripped === "--") return;

    const lowered = stripped.toLowerCase();
    if (lowered.startsWith("--")) {
      const flagName = stripped.slice(2).split(/[=:]/, 1)[0];
      const normalized = normalizeFeatureName(flagName);
      add(`flag_name_${normalized}`);
      add("flag_long");
      add("flag");
      for (const extra of flagControlTokens(normalized)) add(extra);
      const value = flagNumericValue(rawTokens, index, stripped, `--${flagName}`);
      if (value !== null) {
        add(flagValueBucket(value));
        if (seen.has("output_limit_flag") || seen.has("output_limiter_present")) {
          add(bucketTokens("output_limiter_size", Math.abs(Number(value) || 0), OUTPUT_LIMITER_SIZE_BUCKETS));
        }
      }
      return;
    }

    const body = stripped.slice(1);
    if (body.length === 1) {
      const short = body.toLowerCase();
      add(`flag_short_${short}`);
      add("flag");
      add("flag_short");
      if (short === "q") add("output_quiet_flag");
      else if (short === "v") add("output_expander_flag");
      const value = short === "n" ? integerFromToken(rawTokens[index + 1]) : null;
      if (value !== null) add(flagValueBucket(value));
      return;
    }

    if (looksLikeShortFlagGroup(body)) {
      add("flag_short_group");
      add("flag");
      for (const short of body.toLowerCase()) {
        add(`flag_short_${short}`);
        if (short === "q") add("output_quiet_flag");
        else if (short === "v") add("output_expander_flag");
      }
      return;
    }

    if (looksLikeAttachedShortValue(body)) {
      const short = body[0].toLowerCase();
      add(`flag_short_${short}`);
      add("flag");
      add("flag_short");
      return;
    }

    const shortMatch = body.match(/^([A-Za-z])(?:=)?([+-]?\d+)$/);
    if (shortMatch) {
      const short = shortMatch[1].toLowerCase();
      const value = integerFromToken(shortMatch[2]);
      add(`flag_short_${short}`);
      add("flag");
      add("flag_short");
      if (short === "q") add("output_quiet_flag");
      else if (short === "v") add("output_expander_flag");
      if (value !== null) add(flagValueBucket(value));
      return;
    }

    const flagName = body.split(/[=:]/, 1)[0];
    const normalized = normalizeFeatureName(flagName);
    add(`flag_name_${normalized}`);
    add("flag_long");
    add("flag");
    for (const extra of flagControlTokens(normalized)) add(extra);
    const value = flagNumericValue(rawTokens, index, stripped, `-${flagName}`);
    if (value !== null) {
      add(flagValueBucket(value));
      if (seen.has("output_limit_flag") || seen.has("output_limiter_present")) {
        add(bucketTokens("output_limiter_size", Math.abs(Number(value) || 0), OUTPUT_LIMITER_SIZE_BUCKETS));
      }
    }
  });

  return features;
}

function toolIdentityFeature(tool, mode) {
  if (mode === "none" || !tool) return null;
  if (mode === "raw") return `tool_raw_${normalizeIdentifier(tool)}`;
  return `tool_hash_${createHash("sha1").update(String(tool)).digest("hex").slice(0, 12)}`;
}

function isNativePathLike(value) {
  const text = String(value || "").trim();
  if (!text || /https?:\/\/\S+/i.test(text)) return false;
  if (WINDOWS_PATH_RE.test(text) || POSIX_PATH_RE.test(text)) return true;
  return (text.includes("/") || text.includes("\\")) && !/\s/.test(text) && PATH_EXTENSION_RE.test(text);
}

function nativeWordCount(value) {
  return (String(value || "").match(/\w+/g) || []).length;
}

function emptyNativeStats() {
  return {
    input_json_chars: 0,
    max_depth: 0,
    object_count: 0,
    array_count: 0,
    field_count: 0,
    max_object_fields: 0,
    max_array_length: 0,
    string_count: 0,
    string_total_chars: 0,
    string_max_chars: 0,
    string_total_lines: 0,
    string_max_lines: 0,
    number_count: 0,
    boolean_count: 0,
    null_count: 0,
    url_count: 0,
    path_like_count: 0,
    markdown_like_count: 0,
    code_like_count: 0,
    json_like_string_count: 0,
    long_string_count: 0,
    query_like_count: 0,
    query_word_max: 0,
    field_names: new Set(),
    field_classes: new Set(),
    field_limit_values: new Set(),
    value_markers: new Set(),
  };
}

function inspectNativeString(value, stats) {
  const text = String(value || "");
  const stripped = text.trim();
  const lineCount = text ? text.split(/\r\n|\r|\n/).length : 0;
  const words = nativeWordCount(text);

  stats.string_count += 1;
  stats.string_total_chars += text.length;
  stats.string_max_chars = Math.max(stats.string_max_chars, text.length);
  stats.string_total_lines += lineCount;
  stats.string_max_lines = Math.max(stats.string_max_lines, lineCount);
  stats.url_count += (text.match(URL_VALUE_RE) || []).length;

  if (isNativePathLike(text)) stats.path_like_count += 1;
  if (MARKDOWN_VALUE_RE.test(text)) stats.markdown_like_count += 1;
  if (CODE_VALUE_RE.test(text)) stats.code_like_count += 1;
  if ((stripped.startsWith("{") && stripped.endsWith("}")) || (stripped.startsWith("[") && stripped.endsWith("]"))) stats.json_like_string_count += 1;
  if (text.length >= 1000) stats.long_string_count += 1;
  if (words >= 3 && words <= 80 && !isNativePathLike(text)) {
    stats.query_like_count += 1;
    stats.query_word_max = Math.max(stats.query_word_max, words);
  }

  const marker = stripped.toLowerCase();
  if (GENERIC_VALUE_MARKERS.has(marker)) stats.value_markers.add(marker);
}

function walkNativeInput(value, stats, depth = 0) {
  stats.max_depth = Math.max(stats.max_depth, depth);

  if (Array.isArray(value)) {
    stats.array_count += 1;
    stats.max_array_length = Math.max(stats.max_array_length, value.length);
    for (const item of value) walkNativeInput(item, stats, depth + 1);
    return;
  }

  if (value && typeof value === "object") {
    const entries = Object.entries(value);
    stats.object_count += 1;
    stats.field_count += entries.length;
    stats.max_object_fields = Math.max(stats.max_object_fields, entries.length);
    for (const [key, item] of entries) {
      if (stats.field_names.size < FIELD_NAME_TOKEN_LIMIT) stats.field_names.add(normalizeFieldName(key));
      for (const label of fieldClassesForName(key)) stats.field_classes.add(label);
      if (typeof item === "number" && Number.isFinite(item) && isLimitLikeField(key)) stats.field_limit_values.add(numericFieldBucket(item));
      walkNativeInput(item, stats, depth + 1);
    }
    return;
  }

  if (typeof value === "string") {
    inspectNativeString(value, stats);
    return;
  }
  if (typeof value === "boolean") {
    stats.boolean_count += 1;
    return;
  }
  if (typeof value === "number") {
    stats.number_count += 1;
    return;
  }
  if (value === null || typeof value === "undefined") {
    stats.null_count += 1;
  }
}

function nativeBucketToken(prefix, value, buckets) {
  return bucketTokens(prefix, Number(value) || 0, buckets);
}

function nativeTokens(stats, identity) {
  const tokens = ["family_native"];
  if (identity) tokens.push(identity);
  tokens.push(nativeBucketToken("json_char_count", stats.input_json_chars, NATIVE_CHAR_BUCKETS));
  tokens.push(nativeBucketToken("json_depth", stats.max_depth, NATIVE_DEPTH_BUCKETS));

  for (const field of [
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
  ]) {
    tokens.push(nativeBucketToken(field, stats[field], NATIVE_COUNT_BUCKETS));
  }

  tokens.push(nativeBucketToken("input_json_chars", stats.input_json_chars, NATIVE_CHAR_BUCKETS));
  for (const field of ["string_total_chars", "string_max_chars", "string_total_lines", "string_max_lines"]) {
    tokens.push(nativeBucketToken(field, stats[field], NATIVE_CHAR_BUCKETS));
  }
  for (const marker of Array.from(stats.value_markers).sort()) {
    tokens.push(`value_marker_${marker}`);
  }
  for (const fieldName of Array.from(stats.field_names).sort()) {
    tokens.push(`field_name_${fieldName}`);
  }
  for (const label of Array.from(stats.field_classes).sort()) {
    tokens.push(label);
  }
  for (const token of Array.from(stats.field_limit_values).sort()) {
    tokens.push(token);
  }
  return tokens;
}

function buildNativeModelText(inputRaw, tool, toolIdentityMode = "hash") {
  const stats = emptyNativeStats();
  stats.input_json_chars = canonicalJson(inputRaw).length;
  walkNativeInput(inputRaw, stats);
  const identity = toolIdentityFeature(tool, toolIdentityMode);
  return nativeTokens(stats, identity).join(" ");
}

function resolveDirectory(directory) {
  if (typeof directory === "string") return directory;
  if (directory && typeof directory === "object") {
    if (typeof directory.path === "string") return directory.path;
    if (typeof directory.cwd === "string") return directory.cwd;
    if (typeof directory.directory === "string") return directory.directory;
  }
  return "";
}

function resolveCommand(input, output) {
  const candidates = [
    output && output.args && output.args.command,
    input && input.args && input.args.command,
    input && input.command,
    output && output.command,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function resolveWorkdir(input, output, fallback) {
  const candidates = [
    output && output.args && (output.args.workdir || output.args.cwd),
    input && input.args && (input.args.workdir || input.args.cwd),
    input && (input.workdir || input.cwd),
    output && (output.workdir || output.cwd),
    fallback,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) return value.trim();
  }
  return "";
}

function setRuntimeConfig(cfg) {
  const defaultAgent =
    (cfg && typeof cfg.default_agent === "string" && cfg.default_agent.trim()) ||
    (cfg && typeof cfg.defaultAgent === "string" && cfg.defaultAgent.trim()) ||
    "";

  const agentModes = new Map();
  const agents = cfg && typeof cfg.agent === "object" && cfg.agent ? cfg.agent : null;
  if (agents) {
    for (const [name, spec] of Object.entries(agents)) {
      if (!spec || typeof spec !== "object") continue;
      const mode = typeof spec.mode === "string" ? spec.mode.trim().toLowerCase() : "";
      if (mode) {
        agentModes.set(name, mode);
      }
    }
  }

  const agentsDir = path.join(os.homedir(), ".config", "opencode", "agents");
  if (existsSync(agentsDir)) {
    for (const entry of readdirSync(agentsDir, { withFileTypes: true })) {
      if (!entry.isFile() || !entry.name.toLowerCase().endsWith(".md")) continue;
      const name = path.basename(entry.name, ".md");
      if (agentModes.has(name)) continue;
      try {
        const text = readFileSync(path.join(agentsDir, entry.name), "utf8");
        const modeMatch = text.match(/^---\s*[\s\S]*?^mode:\s*([a-z]+)\s*$/im);
        if (modeMatch) {
          agentModes.set(name, modeMatch[1].trim().toLowerCase());
        }
      } catch {
        // Ignore unreadable agent definitions and fall back to runtime hints.
      }
    }
  }

  runtimeConfig = { defaultAgent, agentModes };
}

function resolveAgentName(input, output) {
  const candidates = [
    input && input.agent,
    output && output.agent,
    input && input.context && input.context.agent,
    output && output.context && output.context.agent,
    input && input.toolContext && input.toolContext.agent,
    output && output.toolContext && output.toolContext.agent,
  ];
  for (const value of candidates) {
    if (typeof value === "string" && value.trim()) {
      return value.trim();
    }
  }
  return "";
}

function resolveAgentRole(agentName) {
  if (!agentName) return "main";
  const mode = runtimeConfig.agentModes.get(agentName);
  if (mode === "subagent") return "subagent";
  if (runtimeConfig.defaultAgent && agentName === runtimeConfig.defaultAgent) return "main";
  if (mode === "primary" || mode === "all") return "main";
  return runtimeConfig.defaultAgent ? "subagent" : "main";
}

function rememberSessionRole(sessionID, agentName) {
  if (typeof sessionID !== "string" || !sessionID.trim()) return;
  sessionRoles.set(sessionID, resolveAgentRole(agentName));
}

function rememberSessionAgent(sessionID, agentName) {
  if (typeof sessionID !== "string" || !sessionID.trim()) return;
  if (typeof agentName !== "string" || !agentName.trim()) return;
  sessionAgents.set(sessionID, agentName.trim());
}

function extractPositiveProbability(outputMap) {
  if (!outputMap || typeof outputMap !== "object") return null;
  const entries = Object.entries(outputMap);
  if (!entries.length) return null;

  let chosen = entries[entries.length - 1][1];
  for (const [name, value] of entries) {
    if (String(name).toLowerCase().includes("prob")) {
      chosen = value;
      break;
    }
  }

  if (chosen && typeof chosen === "object") {
    if (Array.isArray(chosen.data)) {
      const data = chosen.data;
      if (Array.isArray(chosen.dims) && chosen.dims.length >= 2 && chosen.dims[1] >= 2 && data.length >= 2) {
        return Number(data[1]);
      }
      if (data.length >= 1) {
        return Number(data[data.length - 1]);
      }
    }
    if (ArrayBuffer.isView(chosen.data)) {
      const data = Array.from(chosen.data);
      if (Array.isArray(chosen.dims) && chosen.dims.length >= 2 && chosen.dims[1] >= 2 && data.length >= 2) {
        return Number(data[1]);
      }
      if (data.length >= 1) {
        return Number(data[data.length - 1]);
      }
    }
    if (typeof chosen === "number") return chosen;
    if (typeof chosen[1] === "number") return chosen[1];
    if (typeof chosen["1"] === "number") return chosen["1"];
  }

  return null;
}

async function readJsonIfExists(filePath) {
  try {
    return JSON.parse(await fs.readFile(filePath, "utf8"));
  } catch {
    return null;
  }
}

function modelSupportsNativeTools(manifest) {
  if (!manifest || typeof manifest !== "object") return false;
  const template = String(manifest.text_template || "").toLowerCase();
  if (template.includes("structured tool input")) return true;
  const train = manifest.dataset && manifest.dataset.train;
  return Boolean(train && train.by_family && train.by_family.native);
}

function nativeToolIdentityMode(manifest) {
  const mode = manifest && manifest.feature_extraction && manifest.feature_extraction.native_tool_identity;
  if (mode === "raw" || mode === "none" || mode === "hash") return mode;
  return "hash";
}

async function ensureState() {
  if (!statePromise) {
    statePromise = (async () => {
      try {
        const [thresholdRaw, manifest] = await Promise.all([
          fs.readFile(THRESHOLD_PATH, "utf8"),
          readJsonIfExists(MANIFEST_PATH),
          fs.access(MODEL_PATH),
        ]);
        const threshold = JSON.parse(thresholdRaw);
        const session = await ort.InferenceSession.create(MODEL_PATH);
        const inputNames = session.inputNames || Object.keys(session.inputMetadata || {});
        const outputNames = session.outputNames || Object.keys(session.outputMetadata || {});
        if (!inputNames.length) throw new Error("ONNX model exposes no inputs");
        return {
          session,
          inputName: inputNames[0],
          outputNames,
          blockThreshold: Number(threshold.block_threshold),
          warnThreshold: Number(threshold.warn_threshold || Math.max(0.05, Number(threshold.block_threshold) * 0.5)),
          supportsNative: modelSupportsNativeTools(manifest),
          toolIdentityMode: nativeToolIdentityMode(manifest),
        };
      } catch (error) {
        await logShellGuard("error", "plugin disabled", {
          error: error instanceof Error ? error.message : String(error),
        });
        return null;
      }
    })().then((value) => {
      if (!value) {
        statePromise = null;
      }
      return value;
    });
  }
  return statePromise;
}

async function runModelScore(state, modelText) {
  const tensor = new ort.Tensor("string", [modelText], [1, 1]);
  const outputs = await state.session.run({ [state.inputName]: tensor });
  const score = extractPositiveProbability(outputs);
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  return { score, state, modelText };
}

async function scoreCommand(command, workdir) {
  const state = await ensureState();
  if (!state) return null;
  return runModelScore(state, `family_shell ${buildModelText(command, workdir)}`);
}

async function scoreNativeTool(tool, inputRaw) {
  const state = await ensureState();
  if (!state || !state.supportsNative) return null;
  return runModelScore(state, buildNativeModelText(inputRaw, tool, state.toolIdentityMode));
}

function truncate(text, limit = 240) {
  const value = String(text || "").replace(/\s+/g, " ").trim();
  if (value.length <= limit) return value;
  return `${value.slice(0, limit - 3)}...`;
}

async function logShellGuard(level, message, extra = {}) {
  if (!pluginClient || !pluginClient.app || typeof pluginClient.app.log !== "function") return;
  try {
    await pluginClient.app.log({
      body: {
        service: "shell-guard",
        level,
        message,
        extra,
      },
    });
  } catch {
    // Logging should never break command protection.
  }
}

function isNestedOpenCodeSessionCommand(command) {
  return /^\s*opencode(?:\.exe|\.cmd|\.bat)?\s+(?:-s|--session)\b/i.test(String(command || ""));
}

function normalizeToolName(tool) {
  return String(tool || "").trim().toLowerCase().replace(/^functions\./, "");
}

function isNeverBlockNativeTool(tool) {
  return NEVER_BLOCK_NATIVE_TOOLS.has(normalizeToolName(tool));
}

function isBlockCapableNativeTool(tool) {
  const name = normalizeToolName(tool);
  if (!name || name === "bash") return false;
  return !isNeverBlockNativeTool(name);
}

function resolveNativeInputRaw(input, output) {
  const candidates = [
    output && output.args,
    input && input.args,
    output && output.input,
    input && input.input,
    output && output.params,
    input && input.params,
  ];
  for (const value of candidates) {
    if (value && typeof value === "object") return value;
  }
  return {};
}

async function handleShellToolBefore(input, output, defaultWorkdir, agentRole, agentName) {
  const command = resolveCommand(input, output);
  if (!command) return;
  const workdir = resolveWorkdir(input, output, defaultWorkdir);

  if (agentRole !== "subagent" && isNestedOpenCodeSessionCommand(command)) {
    throw new Error(`shell guard blocked nested opencode session command: ${truncate(command)}`);
  }

  let result;
  try {
    result = await scoreCommand(command, workdir);
  } catch (error) {
    await logShellGuard("error", "scoring failed", {
      error: error instanceof Error ? error.message : String(error),
    });
    return;
  }

  if (!result) return;
  const { score, state } = result;
  const scoreText = score.toFixed(3);
  const commandText = truncate(command);

  if (agentRole === "subagent") {
    if (score >= state.warnThreshold) {
      await logShellGuard("warn", "subagent warning", {
        tool: "bash",
        role: agentRole,
        agent: agentName || "<unknown>",
        score: scoreText,
        cwd: workdir || "<unknown>",
        cmd: commandText,
      });
    }
    return;
  }

  if (score >= state.blockThreshold) {
    throw new Error(
      buildBlockedCallMessage({
        toolLabel: "shell command",
        agentRole,
        agentName,
        scoreText,
        thresholdText: state.blockThreshold.toFixed(3),
        detailLabel: "command",
        detailText: commandText,
      })
    );
  }

  if (score >= state.warnThreshold) {
    await logShellGuard("warn", "command warning", {
      tool: "bash",
      role: agentRole,
      agent: agentName || "<unknown>",
      score: scoreText,
      threshold: state.warnThreshold.toFixed(3),
      cwd: workdir || "<unknown>",
      cmd: commandText,
    });
  }
}

async function handleNativeToolBefore(input, output, agentRole, agentName) {
  const tool = String(input && input.tool || "").trim();
  if (!tool) return;

  let result;
  try {
    result = await scoreNativeTool(tool, resolveNativeInputRaw(input, output));
  } catch (error) {
    await logShellGuard("error", "native scoring failed", {
      tool,
      error: error instanceof Error ? error.message : String(error),
    });
    return;
  }

  if (!result) return;
  const { score, state } = result;
  const scoreText = score.toFixed(3);
  const policy = isBlockCapableNativeTool(tool) ? "block_capable" : "never_block";

  if (agentRole === "subagent") {
    if (score >= state.warnThreshold) {
      await logShellGuard("warn", "subagent native warning", {
        tool,
        policy: "warn_only",
        role: agentRole,
        agent: agentName || "<unknown>",
        score: scoreText,
        threshold: state.warnThreshold.toFixed(3),
      });
    }
    return;
  }

  if (policy === "block_capable" && score >= state.blockThreshold) {
    const inputText = truncate(JSON.stringify(resolveNativeInputRaw(input, output)));
    throw new Error(
      buildBlockedCallMessage({
        toolLabel: `${tool} tool`,
        agentRole,
        agentName,
        scoreText,
        thresholdText: state.blockThreshold.toFixed(3),
        detailLabel: "input",
        detailText: inputText,
      })
    );
  }

  if (score >= state.warnThreshold) {
    await logShellGuard("warn", "native tool warning", {
      tool,
      policy,
      role: agentRole,
      agent: agentName || "<unknown>",
      score: scoreText,
      threshold: state.warnThreshold.toFixed(3),
    });
  }
}

async function handleToolExecuteBefore(input, output, defaultWorkdir) {
  if (!input || !input.tool) return;
  const sessionID = typeof input.sessionID === "string" ? input.sessionID : "";
  const agentName = resolveAgentName(input, output) || sessionAgents.get(sessionID) || "";
  const agentRole = sessionRoles.get(sessionID) || resolveAgentRole(agentName);

  if (normalizeToolName(input.tool) === "bash") {
    await handleShellToolBefore(input, output, defaultWorkdir, agentRole, agentName);
    return;
  }

  await handleNativeToolBefore(input, output, agentRole, agentName);
}

export const ShellGuard = async function CasifierShellGuard(input) {
  const { directory, client } = input || {};
  pluginClient = client || null;
  const defaultWorkdir = resolveDirectory(directory);

  return {
    config(cfg) {
      setRuntimeConfig(cfg);
    },
    "chat.params"(input) {
      rememberSessionRole(input && input.sessionID, input && input.agent);
      rememberSessionAgent(input && input.sessionID, input && input.agent);
    },
    "chat.message"(input) {
      rememberSessionRole(input && input.sessionID, input && input.agent);
      rememberSessionAgent(input && input.sessionID, input && input.agent);
    },
    async "tool.execute.before"(input, output) {
      await handleToolExecuteBefore(input, output, defaultWorkdir);
    },
  };
};

export default ShellGuard;
