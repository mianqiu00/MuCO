import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { AA_PATTERN } from "@/lib/utils";
import { createJob } from "@/lib/jobs";

const schema = z.object({
  sequence: z.string().min(2).max(30).regex(AA_PATTERN),
  K: z.number().int().min(1).max(3),
  M: z.number().int().min(1).max(5),
  downloadEnabled: z.boolean().default(true),
});

export async function POST(request: NextRequest) {
  const body = await request.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }
  const job = createJob(parsed.data);
  return NextResponse.json(job);
}
