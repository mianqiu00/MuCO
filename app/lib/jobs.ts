export const MUCO_API_URL = process.env.MUCO_API_URL ?? "http://127.0.0.1:8000";

export type MucoRequest = {
  sequence: string;
  K: number;
  M: number;
  downloadEnabled: boolean;
  backboneSteps?: number;
  sidechainSteps?: number;
  sidechainCoeff?: number;
  noiseScale?: number;
  minT?: number;
  relaxPlatform?: "CUDA" | "CPU" | "OpenCL";
};

export function assertJobId(jobId: string) {
  if (!/^(\d{8}T\d{6}_)?[0-9a-f-]{36}$/.test(jobId)) throw new Error("Invalid job id");
}

export async function apiFetch(path: string, init?: RequestInit) {
  const res = await fetch(`${MUCO_API_URL}${path}`, { ...init, cache: "no-store" });
  if (!res.ok) {
    let message = `MuCO API request failed: ${res.status}`;
    try {
      const body = await res.json();
      message = body.detail || body.error || message;
    } catch {
      // Keep the status-derived message for non-JSON responses.
    }
    throw new Error(message);
  }
  return res;
}

export async function createJob(req: MucoRequest) {
  const payload = {
    sequence: req.sequence,
    K: req.K,
    M: req.M,
    make_zip: req.downloadEnabled,
    backbone_steps: req.backboneSteps,
    sidechain_steps: req.sidechainSteps,
    sidechain_coeff: req.sidechainCoeff,
    noise_scale: req.noiseScale,
    min_t: req.minT,
    relax_platform: req.relaxPlatform,
  };
  const res = await apiFetch("/jobs", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  return { id: data.job_id ?? data.id };
}

export function fileToken(kind: "pdb" | "png", k: number, m: number) {
  return `${kind}_${k}_${m}`;
}

export function publicFileUrl(jobId: string, token: string) {
  return `/api/jobs/${jobId}/files?file=${encodeURIComponent(token)}`;
}

export async function appendFeedback(value: unknown) {
  await apiFetch("/feedback", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(value),
  }).catch(() => undefined);
}
