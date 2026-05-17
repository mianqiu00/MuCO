import { NextResponse } from "next/server";
import { apiFetch, assertJobId } from "@/lib/jobs";

export async function GET(_: Request, context: { params: Promise<{ id: string }> }) {
  const { id } = await context.params;
  try {
    assertJobId(id);
  } catch {
    return NextResponse.json({ error: "Invalid job id" }, { status: 400 });
  }
  try {
    const upstream = await apiFetch(`/jobs/${id}/download`);
    const data = await upstream.arrayBuffer();
    return new NextResponse(data, {
      headers: {
        "content-type": upstream.headers.get("content-type") ?? "application/zip",
        "content-disposition": upstream.headers.get("content-disposition") ?? `attachment; filename="${id}.zip"`,
      },
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Download proxy failed" }, { status: 502 });
  }
}
