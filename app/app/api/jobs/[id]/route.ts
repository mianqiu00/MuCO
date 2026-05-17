import { NextResponse } from "next/server";
import { apiFetch, assertJobId, fileToken, publicFileUrl } from "@/lib/jobs";

type SummaryRow = {
  id: string;
  sequence: string;
  k: number;
  m: number;
  relaxed_pdb?: string | null;
  sidechain_pdb?: string | null;
  render_png?: string | null;
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
  try {
    const progress = await (await apiFetch(`/jobs/${id}`)).json();
    let rows: SummaryRow[] = [];
    if (progress.status === "done") {
      rows = await (await apiFetch(`/jobs/${id}/summary`)).json();
    }
    const successful = rows
      .filter((row) => row.relax?.cyclized || row.sidechain_pdb)
      .map((row) => {
        const pdb = row.relaxed_pdb || row.sidechain_pdb;
        return {
          k: row.k,
          m: row.m,
          energy: row.relax?.efinal ?? null,
          pdbUrl: pdb ? publicFileUrl(id, fileToken("pdb", row.k, row.m)) : null,
          imageUrl: row.render_png ? publicFileUrl(id, fileToken("png", row.k, row.m)) : null,
        };
      });
    const zip = rows.find((row) => row.success_zip)?.success_zip;
    return NextResponse.json({
      id,
      request: null,
      progress,
      results: successful,
      downloadUrl: zip ? `/api/jobs/${id}/download` : null,
    });
  } catch (error) {
    return NextResponse.json({ error: error instanceof Error ? error.message : "MuCO API unavailable" }, { status: 502 });
  }
}
