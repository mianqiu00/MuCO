import { NextRequest, NextResponse } from "next/server";
import { apiFetch, assertJobId } from "@/lib/jobs";

type SummaryRow = {
  sequence?: string;
  k: number;
  m: number;
};


function safeFilenamePart(value: string) {
  return value.replace(/[^A-Za-z0-9_-]/g, "_") || "muco";
}

export async function GET(request: NextRequest, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }
  const token = request.nextUrl.searchParams.get("file");
  const match = /^(pdb|png)_(\d+)_(\d+)$/.exec(token ?? "");
  if (!match) return NextResponse.json({ error: "Invalid file token" }, { status: 400 });

  try {
    const upstream = await apiFetch(`/jobs/${id}/files?path=${encodeURIComponent(token ?? "")}`);
    const data = await upstream.arrayBuffer();
    const headers: Record<string, string> = { "content-type": match[1] === "png" ? "image/png" : "chemical/x-pdb" };
    if (match[1] === "pdb") {
      const rows: SummaryRow[] = await (await apiFetch(`/jobs/${id}/summary`)).json();
      const k = Number.parseInt(match[2], 10);
      const m = Number.parseInt(match[3], 10);
      const row = rows.find((item) => item.k === k && item.m === m);
      const sequence = safeFilenamePart(row?.sequence ?? "muco");
      headers["content-disposition"] = `attachment; filename="${sequence}_K${k}_M${m}.pdb"`;
    }
    return new NextResponse(data, { headers });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "File proxy failed" }, { status: 502 });
  }
}
