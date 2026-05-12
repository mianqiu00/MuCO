import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { assertJobId, dispatchQueue, fileToken, jobDir, publicFileUrl, queueAhead, readJson, renderPdb } from "@/lib/jobs";

type SummaryRow = {
  id: string;
  sequence: string;
  k: number;
  m: number;
  relaxed_pdb?: string | null;
  sidechain_pdb?: string | null;
  success_zip?: string;
  relax?: { cyclized?: boolean; efinal?: number } | null;
};

export async function GET(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }
  const dir = jobDir(id);
  dispatchQueue();
  const progress = readJson<Record<string, unknown>>(path.join(dir, "progress.json"), { status: "missing", stage: "missing" });
  if (progress.status === "queued") progress.queue_ahead = queueAhead(id);
  const request = readJson(path.join(dir, "request.json"), null);
  const summaryPath = path.join(dir, "output", "summary.json");
  const rows = readJson<SummaryRow[]>(summaryPath, []);
  const successful = rows
    .filter((row) => row.relax?.cyclized)
    .map((row) => {
      const pdb = row.relaxed_pdb || row.sidechain_pdb;
      const image = pdb ? renderPdb(id, pdb) : null;
      return {
        k: row.k,
        m: row.m,
        energy: row.relax?.efinal ?? null,
        pdbUrl: pdb ? publicFileUrl(id, fileToken("pdb", row.k, row.m)) : null,
        imageUrl: image && fs.existsSync(image) ? publicFileUrl(id, fileToken("png", row.k, row.m)) : null,
      };
    });
  const zip = rows.find((row) => row.success_zip)?.success_zip;
  return NextResponse.json({
    id,
    request,
    progress,
    results: successful,
    downloadUrl: zip ? `/api/jobs/${id}/download` : null,
  });
}
