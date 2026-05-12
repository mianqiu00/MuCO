import fs from "node:fs";
import path from "node:path";
import { NextResponse } from "next/server";
import { assertJobId, jobDir, readJson } from "@/lib/jobs";

export async function GET(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }
  const request = readJson<{ sequence?: string } | null>(path.join(jobDir(id), "request.json"), null);
  const safeSeq = (request?.sequence ?? "muco").replace(/[^A-Za-z0-9_-]/g, "") || "muco";
  const zipPath = path.join(jobDir(id), "output", `${safeSeq}.zip`);
  if (!fs.existsSync(zipPath)) return NextResponse.json({ error: "No successful zip yet" }, { status: 404 });
  return new NextResponse(fs.readFileSync(zipPath), {
    headers: {
      "content-type": "application/zip",
      "content-disposition": `attachment; filename="${safeSeq}.zip"`,
    },
  });
}
