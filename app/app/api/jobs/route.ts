import { NextRequest, NextResponse } from "next/server";
import { z } from "zod";
import { AA_PATTERN } from "@/lib/utils";
import { createJob } from "@/lib/jobs";

const schema = z.object({
  sequence: z.string().min(2).max(30).regex(AA_PATTERN),
  K: z.number().int().min(1).max(3),
  M: z.number().int().min(1).max(5),
  downloadEnabled: z.boolean().default(true),
  backboneSteps: z.number().int().min(2).max(200).optional(),
  sidechainSteps: z.number().int().min(1).max(50).optional(),
  sidechainCoeff: z.number().min(0).max(20).optional(),
  noiseScale: z.number().min(0).max(5).optional(),
  minT: z.number().min(0.0001).max(1).optional(),
  relaxPlatform: z.enum(["CUDA", "CPU", "OpenCL"]).optional(),
});

export async function POST(request: NextRequest) {
  const body = await request.json();
  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    return NextResponse.json({ error: parsed.error.flatten() }, { status: 400 });
  }
  try {
    const job = await createJob(parsed.data);
    return NextResponse.json(job);
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "Failed to start MuCO job" }, { status: 502 });
  }
}
