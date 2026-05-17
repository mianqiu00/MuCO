"use client";

import { useEffect, useMemo, useState, startTransition } from "react";
import { CheckCircle2, Download, Loader2, Sparkles } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Textarea } from "@/components/ui/textarea";
import { AA_PATTERN, sanitizeSequence } from "@/lib/utils";

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
  const [backboneSteps, setBackboneSteps] = useState(100);
  const [sidechainSteps, setSidechainSteps] = useState(10);
  const [advancedOpen, setAdvancedOpen] = useState(false);
  const [downloadEnabled, setDownloadEnabled] = useState(true);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<JobState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [selectedGood, setSelectedGood] = useState<Set<string>>(new Set());

  const cleanSequence = sequence.replace(/\s+/g, "").toUpperCase();
  const valid = cleanSequence.length >= 2 && cleanSequence.length <= 30 && AA_PATTERN.test(cleanSequence);
  const resultMap = useMemo(() => new Map(job?.results.map((r) => [`${r.k}-${r.m}`, r]) ?? []), [job]);
  const isRunning = submitting || job?.progress?.status === "queued" || (!!jobId && !["done", "failed"].includes(job?.progress?.status ?? ""));

  async function submit() {
    if (isRunning) {
      setError("A MuCO job is already running. Please wait and keep this page open until it finishes.");
      return;
    }
    if (!valid) {
      setError("Use one-letter amino-acid codes only; sequence length must be 2-30 residues.");
      return;
    }
    setSubmitting(true);
    setError(null);
    setSelectedGood(new Set());
    setJob({ id: "requesting", progress: { status: "queued", stage: "requesting" }, results: [], downloadUrl: null });
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({
        sequence: cleanSequence,
        K: k,
        M: m,
        downloadEnabled,
        backboneSteps,
        sidechainSteps,
      }),
    });
    const data = await res.json();
    setSubmitting(false);
    if (!res.ok) {
      setError("Failed to start MuCO job.");
      return;
    }
    setJobId(data.id);
  }

  async function markGood(item: Result) {
    if (!jobId) return;
    const key = `${item.k}-${item.m}`;
    const action = selectedGood.has(key) ? "deselect" : "select";
    setSelectedGood((prev) => {
      const next = new Set(prev);
      if (action === "select") next.add(key);
      else next.delete(key);
      return next;
    });
    await fetch(`/api/jobs/${jobId}/feedback`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ k: item.k, m: item.m, action }),
    });
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
        <header className="flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <img src="/logo.png" alt="MuCO logo" className="h-12 w-12 rounded-2xl bg-white/95 object-contain p-1 shadow-glow" />
            <div>
              <div className="text-lg font-black text-white">MuCO</div>
              <div className="text-xs uppercase tracking-[0.28em] text-teal-200/70">Cyclic peptide generation</div>
            </div>
          </div>
          <a href="https://arxiv.org/abs/2602.11189" target="_blank" rel="noreferrer" className="rounded-full border border-teal-200/20 bg-white/10 px-4 py-2 text-sm font-semibold text-teal-50 hover:bg-white/15">
            Read the paper
          </a>
        </header>

        <section className="grid gap-6 lg:grid-cols-[1.15fr_0.85fr]">
          <div className="space-y-6">
            <div className="inline-flex rounded-full border border-teal-200/20 bg-teal-300/10 px-4 py-2 text-sm text-teal-100">Multi-stage conformation optimization</div>
            <div className="space-y-4">
              <h1 className="max-w-2xl text-4xl font-black leading-[1.08] tracking-tight text-white md:text-6xl">
                MuCO: Generative Peptide Cyclization
              </h1>
              <p className="max-w-2xl text-lg leading-8 text-slate-300">
                Enter a peptide sequence and explore backbone sampling, side-chain packing, and physics-aware relaxation in a single workflow.
              </p>
            </div>
          </div>

          <Card className="border-amber-200/40 bg-amber-300/10 shadow-glow">
            <CardContent className="p-5">
              <div className="text-sm font-semibold uppercase tracking-[0.25em] text-amber-100">Citation</div>
              <div className="mt-2 text-lg font-bold text-white">MuCO: Generative Peptide Cyclization Empowered by Multi-stage Conformation Optimization</div>
              <a className="mt-2 inline-block text-sm font-semibold text-amber-100 underline underline-offset-4" href="https://arxiv.org/abs/2602.11189" target="_blank" rel="noreferrer">arXiv:2602.11189</a>
            </CardContent>
          </Card>
        </section>

        <section className="grid gap-6 lg:grid-cols-[1fr_0.78fr]">
          <Card className="glass-card">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-2xl"><Sparkles className="text-teal-300" /> Start a MuCO run</CardTitle>
              <CardDescription>Generate up to 3 backbone samples and 5 side-chain packings per backbone.</CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <Label>Peptide sequence</Label>
                <Textarea value={sequence} onChange={(e) => setSequence(sanitizeSequence(e.target.value))} placeholder="ACDEFGHIK" maxLength={30} />
                <div className="text-xs text-slate-400">Length: {cleanSequence.length}/30 residues. Unsupported characters are ignored.</div>
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
              <div className="rounded-2xl border border-teal-200/15 bg-slate-950/30">
                <button
                  type="button"
                  onClick={() => setAdvancedOpen((value) => !value)}
                  className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-teal-50"
                >
                  <span>Advanced Settings</span>
                  <span className="text-teal-200/70">{advancedOpen ? "Hide" : "Show"}</span>
                </button>
                {advancedOpen && (
                  <div className="grid gap-5 border-t border-teal-200/10 p-4 md:grid-cols-2">
                    <div className="space-y-3">
                      <Label>Backbone steps: {backboneSteps}</Label>
                      <Slider min={2} max={200} step={1} value={[backboneSteps]} onValueChange={(v) => setBackboneSteps(v[0])} />
                    </div>
                    <div className="space-y-3">
                      <Label>Side-chain steps: {sidechainSteps}</Label>
                      <Slider min={1} max={50} step={1} value={[sidechainSteps]} onValueChange={(v) => setSidechainSteps(v[0])} />
                    </div>
                  </div>
                )}
              </div>
              <label className="flex items-center justify-between rounded-2xl border border-teal-200/15 bg-slate-950/30 px-4 py-3 text-sm">
                <span>Package successful relaxed structures as a ZIP</span>
                <input type="checkbox" checked={downloadEnabled} onChange={(e) => setDownloadEnabled(e.target.checked)} className="h-5 w-5 accent-teal-300" />
              </label>
              {error && <div className="rounded-xl border border-red-300/30 bg-red-500/10 p-3 text-sm text-red-100">{error}</div>}
              {isRunning && (
                <div className="rounded-xl border border-teal-300/20 bg-teal-300/10 p-3 text-sm text-teal-50">
                  Your MuCO run is in progress. Please keep this page open; another generation can be started after this run finishes.
                </div>
              )}
              {job?.progress?.status === "failed" && (
                <div className="rounded-xl border border-red-300/30 bg-red-500/10 p-3 text-sm text-red-100">
                  This MuCO run failed. Please try again or use a shorter sequence.
                </div>
              )}
              <Button onClick={submit} disabled={isRunning} size="lg" className="w-full">
                {isRunning ? <Loader2 className="animate-spin" /> : <Sparkles />} {isRunning ? "Generation in progress" : `Generate ${k * m} conformations`}
              </Button>
            </CardContent>
          </Card>
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Generation Progress</CardTitle>
              <CardDescription>{jobId ? `Run ${jobId.slice(0, 8)}` : "Submit a sequence to begin."}</CardDescription>
            </CardHeader>
            <CardContent className="space-y-5">
              <div className="rounded-2xl border border-teal-200/15 bg-slate-950/40 p-4">
                <div className="text-sm text-muted-foreground">Queue</div>
                <div className="mt-1 text-3xl font-black text-teal-100">
                  {job ? `${job.progress.queue_ahead ?? (job.progress.status === "queued" ? 1 : 0)}` : "0"}
                </div>
                <div className="mt-1 text-xs text-slate-400">jobs ahead in the queue</div>
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
        </section>

        <section>
          <Card className="glass-card">
            <CardHeader>
              <CardTitle>Choose the Best Structure</CardTitle>
              <CardDescription>Columns are backbone samples K; rows are side-chain packings M. Only successful relaxed structures are shown.</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid gap-4" style={{ gridTemplateColumns: `repeat(${k}, minmax(0, 1fr))` }}>
                {Array.from({ length: m }).flatMap((_, mi) =>
                  Array.from({ length: k }).map((_, ki) => {
                    const item = resultMap.get(`${ki + 1}-${mi + 1}`);
                    return (
                      <div key={`${ki}-${mi}`} className="min-h-[220px] rounded-2xl border border-teal-200/15 bg-slate-950/40 p-3">
                        <div className="mb-2 flex justify-between text-xs text-slate-300"><span>K{ki + 1} / M{mi + 1}</span><span>{item?.energy != null ? `${item.energy.toFixed(1)} kcal/mol` : "pending"}</span></div>
                        {item?.imageUrl ? <img src={item.imageUrl} alt={`K${ki + 1} M${mi + 1}`} className="h-40 w-full rounded-xl object-contain bg-white" /> : <div className="flex h-40 items-center justify-center rounded-xl bg-slate-900 text-sm text-slate-500">No successful structure yet</div>}
                        {item && (
                          <div className="mt-3 grid grid-cols-2 gap-2">
                            <Button size="sm" variant={selectedGood.has(`${item.k}-${item.m}`) ? "default" : "outline"} onClick={() => markGood(item)}>
                              <CheckCircle2 /> {selectedGood.has(`${item.k}-${item.m}`) ? "Good" : "Looks good"}
                            </Button>
                            {item.pdbUrl && <a className="inline-flex h-9 items-center justify-center rounded-full border border-teal-300/40 text-sm text-teal-100 hover:bg-teal-400/10" href={item.pdbUrl}>PDB</a>}
                          </div>
                        )}
                      </div>
                    );
                  }),
                )}
              </div>
            </CardContent>
          </Card>
        </section>

        <section>
          <Card className="glass-card border-teal-200/10 bg-slate-950/30">
            <CardHeader>
              <CardTitle>Acknowledgments</CardTitle>
              <CardDescription>
                MuCO builds on open scientific software for protein modeling, equivariant learning, and molecular simulation.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 text-sm leading-6 text-slate-300 md:grid-cols-2">
              <a className="font-semibold text-teal-100 underline underline-offset-4" href="https://github.com/DreamFold/FoldFlow" target="_blank" rel="noreferrer">FoldFlow</a>
              <a className="font-semibold text-teal-100 underline underline-offset-4" href="https://github.com/atomicarchitects/equiformer_v2" target="_blank" rel="noreferrer">EquiformerV2</a>
              <a className="font-semibold text-teal-100 underline underline-offset-4" href="https://github.com/aqlaboratory/openfold" target="_blank" rel="noreferrer">OpenFold</a>
              <a className="font-semibold text-teal-100 underline underline-offset-4" href="https://github.com/facebookresearch/esm" target="_blank" rel="noreferrer">ESM</a>
              <a className="font-semibold text-teal-100 underline underline-offset-4" href="https://openmm.org/" target="_blank" rel="noreferrer">OpenMM</a>
              <span>CHARMM36 force field for physics-aware all-atom refinement.</span>
            </CardContent>
          </Card>
        </section>
      </main>
    </div>
  );
}
