/**
 * OES AI Studio API — Cloudflare Worker
 * Gemini plans and edits OES files; GitHub changes are submitted as a pull request.
 * Required bindings: OES_SESSIONS (KV)
 * Required secrets: GEMINI_API_KEY, GITHUB_CLIENT_ID
 */
const OWNER = "tuhingm";
const REPO = "almamun";
const DEFAULT_MODEL = "gemini-3.8-flash";
const ALLOWED_ORIGIN = "https://tuhingm.github.io";
const MAX_FILES = 10;
const MAX_FILE_CHARS = 120000;
const MAX_TOTAL_CHARS = 420000;

function cors(origin) {
  const allowed = origin === ALLOWED_ORIGIN || /^https?:\/\/(localhost|127\.0\.0\.1)(:\d+)?$/.test(origin || "");
  return {
    "Access-Control-Allow-Origin": allowed ? origin : ALLOWED_ORIGIN,
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type,X-OES-Session",
    "Access-Control-Max-Age": "86400",
    "Vary": "Origin"
  };
}
function json(data, status, origin) {
  return new Response(JSON.stringify(data), {
    status: status || 200,
    headers: { "Content-Type": "application/json; charset=utf-8", ...cors(origin) }
  });
}
function safePath(path) {
  return typeof path === "string" && path.length < 240 &&
    !path.startsWith("/") && !path.includes("..") &&
    (path.startsWith("oes/") || path.startsWith("assets/"));
}
async function github(path, token, options) {
  const res = await fetch("https://api.github.com" + path, {
    ...options,
    headers: {
      "Accept": "application/vnd.github+json",
      "Authorization": "Bearer " + token,
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "OES-AI-Studio",
      ...(options && options.headers || {})
    }
  });
  const body = await res.text();
  let data;
  try { data = body ? JSON.parse(body) : {}; } catch { data = { message: body }; }
  if (!res.ok) throw new Error("GitHub API " + res.status + ": " + (data.message || body.slice(0, 300)));
  return data;
}
async function getSession(request, env) {
  const sid = request.headers.get("X-OES-Session") || "";
  if (!/^[a-f0-9]{48,80}$/.test(sid)) throw new Error("Connect your GitHub account first.");
  const token = await env.OES_SESSIONS.get("session:" + sid);
  if (!token) throw new Error("Your GitHub session has expired. Reconnect GitHub.");
  return token;
}
async function githubFile(path, token) {
  const data = await github("/repos/" + OWNER + "/" + REPO + "/contents/" + path.split("/").map(encodeURIComponent).join("/") + "?ref=main", token);
  if (Array.isArray(data) || data.type !== "file") throw new Error("Not a file: " + path);
  const binary = atob(data.content.replace(/\n/g, ""));
  const bytes = Uint8Array.from(binary, c => c.charCodeAt(0));
  const content = new TextDecoder().decode(bytes);
  return { path, sha: data.sha, content };
}
async function startDeviceAuth(request, env) {
  const form = new URLSearchParams({ client_id: env.GITHUB_CLIENT_ID, scope: "public_repo" });
  const res = await fetch("https://github.com/login/device/code", {
    method: "POST", headers: { "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: form
  });
  const data = await res.json();
  if (!res.ok || !data.device_code) throw new Error(data.error_description || data.error || "Could not start GitHub device authorization.");
  return data;
}
async function pollDeviceAuth(body, env) {
  const form = new URLSearchParams({
    client_id: env.GITHUB_CLIENT_ID,
    device_code: body.device_code,
    grant_type: "urn:ietf:params:oauth:grant-type:device_code"
  });
  const res = await fetch("https://github.com/login/oauth/access_token", {
    method: "POST", headers: { "Accept": "application/json", "Content-Type": "application/x-www-form-urlencoded" }, body: form
  });
  const data = await res.json();
  if (data.access_token) {
    const user = await github("/user", data.access_token);
    if (!user.login || user.login.toLowerCase() !== OWNER.toLowerCase()) {
      return { authorized: false, error: "access_denied", message: "Only the repository owner account @" + OWNER + " can connect to this AI workspace." };
    }
    const sid = Array.from(crypto.getRandomValues(new Uint8Array(24))).map(b => b.toString(16).padStart(2, "0")).join("");
    await env.OES_SESSIONS.put("session:" + sid, data.access_token, { expirationTtl: 604800 });
    return { authorized: true, session: sid, user: user.login };
  }
  return { authorized: false, error: data.error || "authorization_pending", message: data.error_description || "" };
}
async function listFiles(token) {
  const repo = await github("/repos/" + OWNER + "/" + REPO, token);
  const branch = repo.default_branch || "main";
  const ref = await github("/repos/" + OWNER + "/" + REPO + "/git/ref/heads/" + encodeURIComponent(branch), token);
  const commit = await github("/repos/" + OWNER + "/" + REPO + "/git/commits/" + ref.object.sha, token);
  const tree = await github("/repos/" + OWNER + "/" + REPO + "/git/trees/" + commit.tree.sha + "?recursive=1", token);
  return {
    branch,
    files: (tree.tree || []).filter(x => x.type === "blob" && (x.path.startsWith("oes/") || x.path.startsWith("assets/")))
      .map(x => ({ path: x.path, size: x.size || 0 })).filter(x => x.size <= MAX_FILE_CHARS)
  };
}
async function generate(body, token, env) {
  const prompt = String(body.prompt || "").trim();
  if (!prompt) throw new Error("Enter an instruction for Gemini.");
  const paths = Array.isArray(body.paths) ? [...new Set(body.paths)].filter(safePath).slice(0, MAX_FILES) : [];
  if (!paths.length) throw new Error("Select at least one OES source file for Gemini to inspect.");
  const files = [];
  let total = 0;
  for (const path of paths) {
    const f = await githubFile(path, token);
    if (f.content.length > MAX_FILE_CHARS) throw new Error("File too large for one editing request: " + path);
    total += f.content.length;
    if (total > MAX_TOTAL_CHARS) throw new Error("Selected source files exceed the context limit. Select fewer files.");
    files.push(f);
  }
  const history = Array.isArray(body.history) ? body.history.slice(-8).map(m =>
    (m.role === "model" ? "GEMINI" : "USER") + ": " + String(m.text || "").slice(0, 12000)
  ).join("\n\n") : "";
  const context = files.map(f => "===== FILE: " + f.path + " =====\n" + f.content).join("\n\n");
  const system = [
    "You are the principal software engineer for Open Econometric Systems (OES), a browser-based scientific econometrics platform.",
    "Work against the supplied real repository files. Preserve existing working behavior, model definitions, estimator mathematics, browser-side computation, accessibility, SEO and visual identity.",
    "Never invent a model, estimator, numerical result, test, file, or API. Do not claim code was run or tested unless a tool actually ran it. Prefer small, coherent changes.",
    "Return ONLY valid JSON matching this schema: { message: string, files: [{ path: string, content: string, reason: string }] }.",
    "For every changed file, return its COMPLETE replacement content, not a diff. Do not include unchanged files. Do not create paths outside oes/ and assets/.",
    "If the instruction is ambiguous or required source files are missing, return files: [] and explain exactly what is needed in message.",
    "Do not output markdown fences or any text outside the JSON object."
  ].join("\n");
  const userText = (history ? "RECENT CONVERSATION:\n" + history + "\n\n" : "") +
    "USER INSTRUCTION:\n" + prompt + "\n\nCURRENT REPOSITORY FILES:\n" + context;
  const res = await fetch("https://generativelanguage.googleapis.com/v1beta/interactions", {
    method: "POST",
    headers: { "Content-Type": "application/json", "x-goog-api-key": env.GEMINI_API_KEY },
    body: JSON.stringify({
      model: env.GEMINI_MODEL || DEFAULT_MODEL,
      system_instruction: system,
      input: userText,
      store: false
    })
  });
  const raw = await res.text();
  let result;
  try { result = JSON.parse(raw); } catch { throw new Error("Gemini returned an unreadable response: " + raw.slice(0, 500)); }
  if (!res.ok) throw new Error("Gemini API " + res.status + ": " + (result.error && result.error.message || raw.slice(0, 400)));
  const textPart = (result.steps || []).filter(s => s.type === "model_output")
    .flatMap(s => s.content || []).filter(p => p.type === "text").map(p => p.text || "").join("") ||
    result.output_text || "";
  let output;
  try { output = JSON.parse(textPart); } catch { throw new Error("Gemini response was not valid structured JSON."); }
  if (!Array.isArray(output.files)) output.files = [];
  output.files = output.files.map(f => ({ path: String(f.path || ""), content: String(f.content || ""), reason: String(f.reason || "") }));
  for (const f of output.files) {
    if (!safePath(f.path)) throw new Error("Gemini proposed a disallowed file path: " + f.path);
    if (f.content.length > MAX_FILE_CHARS) throw new Error("Gemini output file is too large: " + f.path);
  }
  return { message: String(output.message || ""), files: output.files, model: env.GEMINI_MODEL || DEFAULT_MODEL };
}
async function createPullRequest(body, token) {
  const files = Array.isArray(body.files) ? body.files : [];
  if (!files.length || files.length > MAX_FILES) throw new Error("No valid proposed files to publish.");
  let total = 0;
  for (const f of files) {
    if (!safePath(f.path)) throw new Error("Disallowed path: " + f.path);
    if (typeof f.content !== "string" || f.content.length > MAX_FILE_CHARS) throw new Error("Invalid or oversized file: " + f.path);
    total += f.content.length;
  }
  if (total > MAX_TOTAL_CHARS) throw new Error("Proposed changes exceed the publishing limit.");
  const repo = await github("/repos/" + OWNER + "/" + REPO, token);
  const base = repo.default_branch || "main";
  const baseRef = await github("/repos/" + OWNER + "/" + REPO + "/git/ref/heads/" + encodeURIComponent(base), token);
  const parentSha = baseRef.object.sha;
  const parentCommit = await github("/repos/" + OWNER + "/" + REPO + "/git/commits/" + parentSha, token);
  const stamp = new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "Z").toLowerCase();
  const branch = "oes-ai-studio-" + stamp;
  await github("/repos/" + OWNER + "/" + REPO + "/git/refs", token, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ref: "refs/heads/" + branch, sha: parentSha })
  });
  const entries = [];
  for (const f of files) {
    const blob = await github("/repos/" + OWNER + "/" + REPO + "/git/blobs", token, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ content: f.content, encoding: "utf-8" })
    });
    entries.push({ path: f.path, mode: "100644", type: "blob", sha: blob.sha });
  }
  const tree = await github("/repos/" + OWNER + "/" + REPO + "/git/trees", token, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ base_tree: parentCommit.tree.sha, tree: entries })
  });
  const commit = await github("/repos/" + OWNER + "/" + REPO + "/git/commits", token, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message: "OES AI Studio: " + String(body.title || "AI-assisted update").slice(0, 100), tree: tree.sha, parents: [parentSha] })
  });
  await github("/repos/" + OWNER + "/" + REPO + "/git/refs/heads/" + encodeURIComponent(branch), token, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ sha: commit.sha })
  });
  const pr = await github("/repos/" + OWNER + "/" + REPO + "/pulls", token, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      title: String(body.title || "OES AI Studio proposed changes").slice(0, 120),
      head: branch, base,
      body: "## OES AI Studio proposed changes\n\n" + files.map(f => "- `" + f.path + "`: " + String(f.reason || "AI-assisted change")).join("\n") + "\n\nGenerated with Gemini and submitted for human review. No changes were committed directly to the default branch."
    })
  });
  return { number: pr.number, url: pr.html_url, branch, files: files.map(f => f.path), commit: commit.sha };
}
export default {
  async fetch(request, env) {
    const origin = request.headers.get("Origin") || "";
    if (request.method === "OPTIONS") return new Response(null, { status: 204, headers: cors(origin) });
    const url = new URL(request.url);
    try {
      if (url.pathname === "/api/health" && request.method === "GET") {
        return json({ ok: true, service: "OES AI Studio API", model: env.GEMINI_MODEL || DEFAULT_MODEL }, 200, origin);
      }
      if (url.pathname === "/api/auth/device" && request.method === "POST") {
        return json(await startDeviceAuth(request, env), 200, origin);
      }
      if (url.pathname === "/api/auth/poll" && request.method === "POST") {
        const data = await pollDeviceAuth(await request.json(), env);
        return json(data, 200, origin);
      }
      const token = await getSession(request, env);
      if (url.pathname === "/api/repo/tree" && request.method === "GET") return json(await listFiles(token), 200, origin);
      if (url.pathname === "/api/ai/edit" && request.method === "POST") {
        return json(await generate(await request.json(), token, env), 200, origin);
      }
      if (url.pathname === "/api/repo/pull-request" && request.method === "POST") {
        return json(await createPullRequest(await request.json(), token), 201, origin);
      }
      if (url.pathname === "/api/auth/logout" && request.method === "POST") {
        const sid = request.headers.get("X-OES-Session") || "";
        if (/^[a-f0-9]{48,80}$/.test(sid)) await env.OES_SESSIONS.delete("session:" + sid);
        return json({ ok: true }, 200, origin);
      }
      return json({ error: "Not found" }, 404, origin);
    } catch (error) {
      return json({ error: error.message || "Unexpected server error" }, 400, origin);
    }
  }
};
