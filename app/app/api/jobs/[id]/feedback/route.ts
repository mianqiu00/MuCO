import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { apiFetch, assertJobId } from "@/lib/jobs";

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

  try {
    await apiFetch("/feedback", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ job_id: id, selected: parsed.data, client: requestMeta(request) }),
    });
    return NextResponse.json({ ok: true });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Feedback proxy failed" }, { status: 502 });
  }
}
