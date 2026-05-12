import fs from "node:fs";
import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { assertJobId, jobDir, readJson } from "@/lib/jobs";

type SummaryRow = {
  k: number;
  m: number;
  relaxed_pdb?: string | null;
  relax?: { cyclized?: boolean } | null;
};

function safeResolve(root: string, filePath: string) {
  const resolved = path.resolve(filePath);
  const normalizedRoot = path.resolve(root) + path.sep;
  if (!resolved.startsWith(normalizedRoot)) return null;
  return resolved;
}

export async function GET(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }
  const token = request.nextUrl.searchParams.get("file");
  if (!token) return NextResponse.json({ error: "Missing file token" }, { status: 400 });
  const match = /^(pdb|png)_(\d+)_(\d+)$/.exec(token);
  if (!match) return NextResponse.json({ error: "Invalid file token" }, { status: 400 });
  const [, kind, kText, mText] = match;
  const k = Number.parseInt(kText, 10);
  const m = Number.parseInt(mText, 10);
  const root = jobDir(id);
  const rows = readJson<SummaryRow[]>(path.join(root, "output", "summary.json"), []);
  const row = rows.find((item) => item.k === k && item.m === m && item.relax?.cyclized && item.relaxed_pdb);
  if (!row?.relaxed_pdb) return NextResponse.json({ error: "File not found" }, { status: 404 });
  const filePath = kind === "pdb"
    ? safeResolve(root, row.relaxed_pdb)
    : safeResolve(root, path.join(root, "renders", `${path.basename(row.relaxed_pdb, ".pdb")}.png`));
  if (!filePath || !fs.existsSync(filePath)) {
    return NextResponse.json({ error: "File not found" }, { status: 404 });
  }
  const data = fs.readFileSync(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const contentType = ext === ".png" ? "image/png" : ext === ".pdb" ? "chemical/x-pdb" : "application/octet-stream";
  return new NextResponse(data, { headers: { "content-type": contentType } });
}
