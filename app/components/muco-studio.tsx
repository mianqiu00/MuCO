"use client";

import { useEffect, useMemo, useState, startTransition } from "react";
import { Download, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { AA_PATTERN } from "@/lib/utils";

type Progress = {
  status?: string;
  stage?: string;
  queue_ahead?: number;
  stage1?: { done: number; total: number };
  stage2?: { done: number; total: number };
  stage3?: { done: number; total: number };
};

type Result = { k: number; m: number; energy: number | null; pdbUrl: string | null; imageUrl: string | null };
type JobState = { id: string; progress: Progress; results: Result[]; downloadUrl: string | null };

function pct(item?: { done: number; total: number }) {
  if (!item?.total) return 0;
  return Math.min(100, Math.round((item.done / item.total) * 100));
}

function ProgressBar({ label, item }: { label: string; item?: { done: number; total: number } }) {
  const value = pct(item);
  return (
    <div className="space-y-2">
      <div className="flex items-center justify-between text-sm text-teal-50/80">
        <span>{label}</span>
        <span>{item?.done ?? 0}/{item?.total ?? 0}</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-slate-800">
        <div className="h-full rounded-full bg-gradient-to-r from-teal-300 to-cyan-400 transition-all" style={{ width: `${value}%` }} />
      </div>
    </div>
  );
}

export function MucoStudio() {
  const [sequence, setSequence] = useState("ACDEFGHIKLMNPQ");
  const [k, setK] = useState(1);
  const [m, setM] = useState(1);
  const [downloadEnabled, setDownloadEnabled] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const cleanSequence = sequence.replace(/\s+/g, "").toUpperCase();
  const valid = cleanSequence.length >= 2 && AA_PATTERN.test(cleanSequence);
  const resultMap = useMemo(() => new Map(job?.results.map((r) => [`${r.k}-${r.m}`, r]) ?? []), [job]);
  const isRunning = submitting || job?.progress?.status === "queued" || (!!jobId && !["done", "failed"].includes(job?.progress?.status ?? ""));

  async function submit() {
    if (isRunning) {
      setError("A MuCO job is already running. Please wait and keep this page open until it finishes.");
      return;
    }
    if (!valid) {
      setError("Use one-letter amino-acid codes only; minimum length is 2.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setJob({ id: "requesting", progress: { status: "queued", stage: "requesting" }, results: [], downloadUrl: null });
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ sequence: cleanSequence, K: k, M: m, downloadEnabled }),
    });
    const data = await res.json();
    setSubmitting(false);
    if (!res.ok) {
      setError("Failed to start MuCO job.");
      return;
    }
    setJobId(data.id);
  }

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    const poll = async () => {
      const res = await fetch(`/api/jobs/${jobId}`, { cache: "no-store" });
      if (!res.ok || cancelled) return;
      const data = await res.json();
      startTransition(() => setJob(data));
      if (!["done", "failed"].includes(data.progress?.status ?? "")) {
        setTimeout(poll, 1800);
      } else if ((data.results ?? []).some((r: Result) => !r.imageUrl)) {
        setTimeout(poll, 1800);
      }
    };
    poll();
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  return (
    <div className="min-h-screen molecule-grid px-4 py-8 md:px-10">
      <main className="mx-auto max-w-7xl space-y-8">
        <section className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
          <div className="space-y-6 pt-8">
            <div className="inline-flex rounded-full border border-teal-200/20 bg-teal-300/10 px-4 py-2 text-sm text-teal-100">
              MuCO online cyclization studio
            </div>
            <a
              href="https://arxiv.org/abs/2602.11189"
              target="_blank"
              rel="noreferrer"
              className="ml-3 inline-flex rounded-full border border-cyan-200/20 bg-cyan-300/10 px-4 py-2 text-sm text-cyan-100 hover:bg-cyan-300/20"
            >
              Paper: arXiv:2602.11189
            </a>
            <div className="space-y-4">
              <h1 className="max-w-3xl text-5xl font-black tracking-tight text-white md:text-7xl">
                Generate cyclic peptides from sequence.
              </h1>
              <p className="max-w-2xl text-lg leading-8 text-slate-300">
                Multi-stage backbone generation, FlowPacker side-chain packing, and OpenMM relaxation in one streamlined interface.
              </p>
            </div>
          </div>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl"><Sparkles className="text-teal-300" /> Generation Setup</CardTitle>
              <CardDescription>K is capped at 3 and M at 5 for online use.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Peptide sequence</Label>
                <Textarea value={sequence} onChange={(e) => setSequence(e.target.value)} placeholder="ACDEFGHIK" />
                <div className="text-xs text-slate-400">Length: {cleanSequence.length} residues</div>
              </div>
              <div className="grid gap-5 md:grid-cols-2">
                <div className="space-y-3">
                  <Label>Backbone samples K: {k}</Label>
                  <Slider min={1} max={3} step={1} value={[k]} onValueChange={(v) => setK(v[0])} />
                </div>
                <div className="space-y-3">
                  <Label>Side-chain samples M: {m}</Label>
                  <Slider min={1} max={5} step={1} value={[m]} onValueChange={(v) => setM(v[0])} />
                </div>
              </div>
              <label className="flex items-center justify-between rounded-2xl border border-teal-200/15 bg-slate-950/30 px-4 py-3 text-sm">
                <span>Enable ZIP download for successful molecules</span>
                <input type="checkbox" checked={downloadEnabled} onChange={(e) => setDownloadEnabled(e.target.checked)} className="h-5 w-5 accent-teal-300" />
              </label>
              {error && <div className="rounded-xl border border-red-300/30 bg-red-500/10 p-3 text-sm text-red-100">{error}</div>}
              {isRunning && (
                <div className="rounded-xl border border-teal-300/20 bg-teal-300/10 p-3 text-sm text-teal-50">
                  A MuCO job is running. Please wait and keep this page open; starting another generation is disabled until this one finishes.
                </div>
              )}
              {job?.progress?.status === "failed" && (
                <div className="rounded-xl border border-red-300/30 bg-red-500/10 p-3 text-sm text-red-100">
                  The MuCO job failed. Please check the server log or try a shorter sequence.
                </div>
              )}
              <Button onClick={submit} disabled={isRunning} size="lg" className="w-full">
                {isRunning ? <Loader2 className="animate-spin" /> : <Sparkles />} {isRunning ? "Generation in progress" : `Generate ${k * m} conformations`}
              </Button>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[0.75fr_1.25fr]">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Pipeline Progress</CardTitle>
              <CardDescription>{jobId ? `Job ${jobId.slice(0, 8)}` : "Submit a sequence to start."}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="rounded-2xl border border-teal-200/15 bg-slate-950/40 p-4">
                <div className="text-sm text-muted-foreground">Queue</div>
                <div className="mt-1 text-3xl font-black text-teal-100">
                  {job ? `${job.progress.queue_ahead ?? (job.progress.status === "queued" ? 1 : 0)}` : "0"}
                </div>
                <div className="mt-1 text-xs text-slate-400">jobs ahead, including model loading overhead</div>
              </div>
              <ProgressBar label="Stage 1 diffusion" item={job?.progress.stage1} />
              <ProgressBar label="Stage 2 diffusion" item={job?.progress.stage2} />
              <ProgressBar label="Stage 3 relaxed molecules" item={job?.progress.stage3} />
              {job?.downloadUrl && downloadEnabled && (
                <Button asChild variant="outline" className="w-full">
                  <a href={job.downloadUrl}><Download /> Download successful PDB ZIP</a>
                </Button>
              )}
            </CardContent>
          </Card>

          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Successful Relaxed Structures</CardTitle>
              <CardDescription>Columns are K, rows are M. Failed cyclization outputs are hidden.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${k}, minmax(0, 1fr))` }}>
                {Array.from({ length: m }).flatMap((_, mi) =>
                  Array.from({ length: k }).map((_, ki) => {
                    const item = resultMap.get(`${ki + 1}-${mi + 1}`);
                    return (
                      <div key={`${ki}-${mi}`} className="min-h-[220px] rounded-2xl border border-teal-200/15 bg-slate-950/40 p-3">
                        <div className="mb-2 flex justify-between text-xs text-slate-300"><span>K{ki + 1} / M{mi + 1}</span><span>{item?.energy != null ? `${item.energy.toFixed(1)} kcal/mol` : "pending"}</span></div>
                        {item?.imageUrl ? <img src={item.imageUrl} alt={`K${ki + 1} M${mi + 1}`} className="h-40 w-full rounded-xl object-contain bg-white" /> : <div className="flex h-40 items-center justify-center rounded-xl bg-slate-900 text-sm text-slate-500">No successful molecule yet</div>}
                        {item?.pdbUrl && <a className="mt-3 block text-center text-sm text-teal-200 hover:text-teal-100" href={item.pdbUrl}>Download PDB</a>}
                      </div>
                    );
                  }),
                )}
              </div>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}
