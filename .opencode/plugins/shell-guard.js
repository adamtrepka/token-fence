import fs from "node:fs/promises";
import { existsSync, readFileSync, readdirSync } from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";
import os from "node:os";

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

let statePromise = null;
let runtimeConfig = {
  defaultAgent: "",
  agentModes: new Map(),
};
let pluginClient = null;
const sessionRoles = new Map();
const sessionAgents = new Map();

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
    if (lowered.length > 2 && /[a-z]/i.test(lowered[1])) return ["flag_short_group", "flag"];
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

async function ensureState() {
  if (!statePromise) {
    statePromise = (async () => {
      try {
        const [thresholdRaw] = await Promise.all([
          fs.readFile(THRESHOLD_PATH, "utf8"),
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

async function scoreCommand(command, workdir) {
  const state = await ensureState();
  if (!state) return null;
  const modelText = buildModelText(command, workdir);
  const tensor = new ort.Tensor("string", [modelText], [1, 1]);
  const outputs = await state.session.run({ [state.inputName]: tensor });
  const score = extractPositiveProbability(outputs);
  if (typeof score !== "number" || Number.isNaN(score)) return null;
  return { score, state, modelText };
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

async function handleToolExecuteBefore(input, output, defaultWorkdir) {
  if (!input || input.tool !== "bash") return;
  const command = resolveCommand(input, output);
  if (!command) return;
  const workdir = resolveWorkdir(input, output, defaultWorkdir);
  const sessionID = typeof input.sessionID === "string" ? input.sessionID : "";
  const agentName = resolveAgentName(input, output) || sessionAgents.get(sessionID) || "";
  const agentRole = sessionRoles.get(sessionID) || resolveAgentRole(agentName);

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
      `shell guard blocked command for ${agentRole} agent${agentName ? ` (${agentName})` : ""} (score=${scoreText}, threshold=${state.blockThreshold.toFixed(3)}): ${commandText}`
    );
  }

  if (score >= state.warnThreshold) {
    await logShellGuard("warn", "command warning", {
      role: agentRole,
      agent: agentName || "<unknown>",
      score: scoreText,
      threshold: state.warnThreshold.toFixed(3),
      cwd: workdir || "<unknown>",
      cmd: commandText,
    });
  }
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
