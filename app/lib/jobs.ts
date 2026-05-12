import fs from "node:fs";
import path from "node:path";
import { spawn, spawnSync } from "node:child_process";
import { randomUUID } from "node:crypto";

export const PROJECT_ROOT = path.resolve(process.cwd(), "..");
export const JOB_ROOT = path.join(PROJECT_ROOT, "runs", "web");
export const PYMOL_SCRIPT = path.join(PROJECT_ROOT, "app", "pymol", "render_pdb.py");
export const MAX_CONCURRENT_JOBS = 2;
export const FEEDBACK_LOG = path.join(JOB_ROOT, "feedback.jsonl");

export type MucoRequest = {
  sequence: string;
  K: number;
  M: number;
  downloadEnabled: boolean;
};

export function ensureDir(dir: string) {
  fs.mkdirSync(dir, { recursive: true });
}

export function jobDir(jobId: string) {
  return path.join(JOB_ROOT, jobId);
}

export function assertJobId(jobId: string) {
  if (!/^(\d{8}T\d{6}_)?[0-9a-f-]{36}$/.test(jobId)) throw new Error("Invalid job id");
}

function timestampPrefix() {
  return new Date().toISOString().replace(/[-:]/g, "").replace(/\.\d{3}Z$/, "");
}

export function readJson<T>(file: string, fallback: T): T {
  try {
    return JSON.parse(fs.readFileSync(file, "utf8")) as T;
  } catch {
    return fallback;
  }
}

function writeJson(file: string, value: unknown) {
  fs.writeFileSync(file, JSON.stringify(value, null, 2));
}

function progressPath(jobId: string) {
  return path.join(jobDir(jobId), "progress.json");
}

function listJobIds() {
  ensureDir(JOB_ROOT);
  return fs.readdirSync(JOB_ROOT).filter((name) => /^(\d{8}T\d{6}_)?[0-9a-f-]{36}$/.test(name) && fs.statSync(jobDir(name)).isDirectory());
}

function jobProgress(jobId: string) {
  return readJson<Record<string, unknown>>(progressPath(jobId), {});
}

function queuedJobIds() {
  return listJobIds()
    .map((id) => ({ id, progress: jobProgress(id), created: readJson<{ created_at?: number }>(path.join(jobDir(id), "process.json"), {}).created_at ?? 0 }))
    .filter((job) => job.progress.status === "queued")
    .sort((a, b) => a.created - b.created)
    .map((job) => job.id);
}

function runningJobIds() {
  return listJobIds().filter((id) => {
    const progress = jobProgress(id);
    if (progress.status !== "running") return false;
    const proc = readJson<{ pid?: number }>(path.join(jobDir(id), "process.json"), {});
    if (!proc.pid) return false;
    try {
      process.kill(proc.pid, 0);
      return true;
    } catch {
      writeJson(progressPath(id), { ...progress, status: "failed", stage: "failed", updated_at: Date.now() / 1000 });
      return false;
    }
  });
}

export function queueAhead(jobId: string) {
  const queued = queuedJobIds();
  const index = queued.indexOf(jobId);
  if (index < 0) return 0;
  return index + 1;
}

function startJob(jobId: string) {
  const dir = jobDir(jobId);
  const req = readJson<MucoRequest>(path.join(dir, "request.json"), { sequence: "", K: 1, M: 1, downloadEnabled: true });
  const inputPath = path.join(dir, "input.json");
  const outputDir = path.join(dir, "output");
  const pPath = progressPath(jobId);
  const logPath = path.join(dir, "muco.log");
  writeJson(pPath, { status: "running", stage: "requesting", queue_ahead: 1, updated_at: Date.now() / 1000 });

  const args = [
    "run", "-n", "muco", "python", path.join(PROJECT_ROOT, "muco_infer.py"), inputPath,
    "--output", outputDir,
    "--K", String(req.K),
    "--M", String(req.M),
    "--device", process.env.MUCO_DEVICE ?? "0",
    "--relax_platform", process.env.MUCO_RELAX_PLATFORM ?? "CPU",
    "--progress_json", pPath,
  ];
  if (req.downloadEnabled) args.push("--make_zip");

  const log = fs.openSync(logPath, "a");
  const child = spawn("conda", args, { cwd: PROJECT_ROOT, detached: true, stdio: ["ignore", log, log] });
  child.on("exit", (code) => {
    const current = readJson<Record<string, unknown>>(pPath, {});
    if (current.status !== "done") {
      writeJson(pPath, { ...current, status: code === 0 ? "done" : "failed", stage: code === 0 ? "done" : "failed", exit_code: code, updated_at: Date.now() / 1000 });
    }
    dispatchQueue();
  });
  child.unref();
  writeJson(path.join(dir, "process.json"), { pid: child.pid, created_at: readJson<{ created_at?: number }>(path.join(dir, "process.json"), {}).created_at ?? Date.now() / 1000, started_at: Date.now() / 1000 });
}

export function dispatchQueue() {
  const available = Math.max(0, MAX_CONCURRENT_JOBS - runningJobIds().length);
  if (available === 0) return;
  for (const id of queuedJobIds().slice(0, available)) startJob(id);
}

export function createJob(req: MucoRequest) {
  ensureDir(JOB_ROOT);
  const id = `${timestampPrefix()}_${randomUUID()}`;
  const dir = jobDir(id);
  ensureDir(dir);
  const inputPath = path.join(dir, "input.json");

  fs.writeFileSync(inputPath, JSON.stringify({ K: req.K, M: req.M, samples: [{ id: "peptide", sequence: req.sequence }] }, null, 2));
  fs.writeFileSync(path.join(dir, "request.json"), JSON.stringify(req, null, 2));
  writeJson(progressPath(id), { status: "queued", stage: "queued", queue_ahead: 1, updated_at: Date.now() / 1000 });
  writeJson(path.join(dir, "process.json"), { created_at: Date.now() / 1000 });
  dispatchQueue();
  return { id };
}

export function renderPdb(jobId: string, pdbPath: string) {
  const imageDir = path.join(jobDir(jobId), "renders");
  ensureDir(imageDir);
  const imageName = `${path.basename(pdbPath, ".pdb")}.png`;
  const imagePath = path.join(imageDir, imageName);
  if (fs.existsSync(imagePath)) return imagePath;
  spawnSync("conda", ["run", "-n", "pymol", "python", PYMOL_SCRIPT, pdbPath, imagePath], { cwd: PROJECT_ROOT, stdio: "ignore", timeout: 120000 });
  return imagePath;
}

export function fileToken(kind: "pdb" | "png", k: number, m: number) {
  return `${kind}_${k}_${m}`;
}

export function publicFileUrl(jobId: string, token: string) {
  return `/api/jobs/${jobId}/files?file=${encodeURIComponent(token)}`;
}

export function appendFeedback(value: unknown) {
  ensureDir(JOB_ROOT);
  fs.appendFileSync(FEEDBACK_LOG, `${JSON.stringify(value)}\n`);
}
