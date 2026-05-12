import path from "node:path";
import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { appendFeedback, assertJobId, jobDir, readJson } from "@/lib/jobs";

type SummaryRow = {
  id: string;
  sequence: string;
  k: number;
  m: number;
  relaxed_pdb?: string | null;
  relax?: { cyclized?: boolean; efinal?: number } | null;
};

const schema = z.object({
  k: z.number().int().min(1).max(3),
  m: z.number().int().min(1).max(5),
  action: z.enum(["select", "deselect"]),
});

function requestMeta(request: NextRequest) {
  return {
    ip: request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ?? request.headers.get("x-real-ip") ?? null,
    user_agent: request.headers.get("user-agent"),
    referer: request.headers.get("referer"),
    accept_language: request.headers.get("accept-language"),
  };
}

export async function POST(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }

  const body = await request.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });

  const root = jobDir(id);
  const requestPayload = readJson(path.join(root, "request.json"), null);
  const rows = readJson<SummaryRow[]>(path.join(root, "output", "summary.json"), []);
  const row = rows.find((item) => item.k === parsed.data.k && item.m === parsed.data.m && item.relax?.cyclized);
  if (!row) return NextResponse.json({ error: "Selected result is not available" }, { status: 404 });

  appendFeedback({
    job_id: id,
    created_at: Date.now() / 1000,
    user_input: requestPayload,
    selected: {
      sequence: row.sequence,
      k: row.k,
      m: row.m,
      action: parsed.data.action,
      energy: row.relax?.efinal ?? null,
      relaxed_pdb_name: row.relaxed_pdb ? path.basename(row.relaxed_pdb) : null,
    },
    client: requestMeta(request),
  });
  return NextResponse.json({ ok: true });
}
