#!/usr/bin/env node
// Chunk padding + batching vs tile padding (segments padded to 32*SP, even split + all-to-all KV write), and two
// attention-side optimisations.
// On each scenario's best feature stack and topology from results/study.json:
//   chunk : chunk C in {128..5120} x batch budget in {off, 4k..32k} (a request takes whole chunks)
//   tile  : budget in {4k..32k} (segments padded to 32*SP only), attention 'request' or 'fused' (one call per batch)
//   modes : base     = attn 'request' (one attention call / prefix gather per request per batch), no prefetch
//           nodedup  = one attention call and prefix gather per chunk (a cold prefill split into many chunks
//                      re-gathers its prefix for every unit)
//           prefetch = KV-prefix gathers overlap the layer's non-collective compute
//   --inf : the same with an infinite KV cache (no evictions, no lanes/arena limit, no SSD tier): compute-bound view
// Usage: JOB=<slurm job> ./on_node.sh node tools/layout_ab.js [--workers 38] [--quick] [--inf] [--out results/layout_ab.json]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, laneCount, CONCS, SLO } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const CHUNKS = [128, 256, 512, 1024, 2048, 5120];
let BUDGETS = [4096, 8192, 16384, 32768];
const MODES = { base: {}, nodedup: { attn: 'chunk' }, prefetch: { prefetchKV: true } };
// --slo none : peak useful tok/s at any TTFT (sweep never stops early, bisects the peak) instead of goodput
// --budgets 4096,...,65536 ; --extra-concs 5120,6144,8192 (high-concurrency peaks with an infinite cache)

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const noSlo = get('--slo', '10') === 'none';
  const slo = noSlo ? 1e6 : Number(get('--slo', SLO));
  if (get('--budgets', null)) BUDGETS = get('--budgets').split(',').map(Number);
  const extra = get('--extra-concs', null) ? get('--extra-concs').split(',').map(Number) : [];
  const concs = (args.includes('--quick') ? CONCS.filter((_, i) => i % 3 === 0) : CONCS).concat(extra);
  const t0 = Date.now();
  const out = {};
  await Promise.all(Object.entries(study.scenarios).map(async ([key, R]) => {
    const keys = R.bestKeys.filter((k) => k !== 'reqPad' && k !== 'fused' && k !== 'batch');
    const g = R.grid[0].extra;
    const topo = { stages: g.stages, mesh: g.mesh, replicas: g.replicas };
    const lanes = args.includes('--inf') ? { cache: 'inf', hostTier: false, laneArena: false }
      : g.arenaTokens ? { laneArena: true, arenaTokens: g.arenaTokens } : { lanes: g.lanes };
    const jobs = [];
    for (const [mode, md] of Object.entries(MODES)) {
      for (const C of CHUNKS) {
        if (mode !== 'nodedup' && C >= 1024) jobs.push({ mode, md, reqPad: 'chunk', attn: 'request', chunk: C, batch: false });
        for (const B of BUDGETS) if (B > C) jobs.push({ mode, md, reqPad: 'chunk', attn: 'request', chunk: C, batch: true, budget: B });
        // fused attention with chunk padding, for the small chunks
        if (mode !== 'nodedup' && C <= 256) for (const B of BUDGETS) jobs.push({ mode, md, reqPad: 'chunk', attn: 'fused', chunk: C, batch: true, budget: B });
      }
      if (mode !== 'nodedup') for (const attn of ['request', 'fused']) for (const B of BUDGETS) jobs.push({ mode, md, reqPad: 'tile', attn, chunk: 5120, batch: true, budget: B });
    }
    out[key] = await Promise.all(jobs.map((j) => {
      const { mode, md, ...d } = j;
      const cfg = laneCount(Object.assign(withFeatures(R.base, keys), topo, lanes, d, md));
      const label = `${mode} ${d.reqPad} ${d.attn} C=${d.chunk} B=${d.budget || 0}`;
      return pool.evalCfg(key + label, cfg, concs, slo, noSlo ? 1e9 : 4, noSlo ? { extend: 0, refinePeak: 3 } : {})
        .then((r) => Object.assign({ mode, reqPad: d.reqPad, attn: d.attn, chunk: d.chunk, budget: d.budget || 0 }, summarize(r.points, slo)));
    }));
  }));
  pool.close();
  const res = { concs, chunks: CHUNKS, budgets: BUDGETS, scenarios: {} };
  const kk = (x) => (x ? (x.goodput / 1e3).toFixed(1) : '-');
  for (const [key, rows] of Object.entries(out)) {
    rows.sort((a, b) => b.goodput - a.goodput);
    res.scenarios[key] = rows.map((r) => ({ mode: r.mode, reqPad: r.reqPad, attn: r.attn, chunk: r.chunk, budget: r.budget, goodput: r.goodput,
      at: r.at && { conc: r.at.conc, ttftP50: r.at.ttftP50, ttftP90: r.at.ttftP90, padFrac: r.at.padFrac, avgSegsPerChunk: r.at.avgSegsPerChunk } }));
    const S = study.scenarios[key];
    console.log(`\n== ${key} (${S.label}), ${S.grid[0].extra.stages}x[${S.grid[0].extra.mesh}]`);
    for (const mode of Object.keys(MODES)) {
      const f = (p) => rows.filter((r) => r.mode === mode && p(r))[0];
      const bf = f((r) => r.reqPad === 'chunk' && r.budget && r.attn === 'request'), ff = f((r) => r.reqPad === 'chunk' && r.attn === 'fused');
      const vs = f((r) => r.reqPad === 'tile' && r.attn === 'request'), vf = f((r) => r.reqPad === 'tile' && r.attn === 'fused');
      const rq = (x) => (x && x.at ? `, ${x.at.avgSegsPerChunk.toFixed(1)} req/chunk` : '');
      console.log(`  ${mode.padEnd(8)} best chunk+batch ${kk(bf)} (C=${bf && bf.chunk} B=${bf && bf.budget / 1024}k${rq(bf)})`
        + (ff ? ` | chunk+fused ${kk(ff)} (C=${ff.chunk} B=${ff.budget / 1024}k)` : '')
        + (vs ? ` | tile request ${kk(vs)} (B=${vs.budget / 1024}k) | tile fused ${kk(vf)} (B=${vf.budget / 1024}k${rq(vf)})` : ''));
      console.log('    chunk:  ' + ['noB', ...BUDGETS.map((b) => 'B' + b / 1024 + 'k')].map((h) => h.padStart(6)).join(''));
      for (const C of CHUNKS) {
        console.log(`    C=${String(C).padStart(4)}` + [0, ...BUDGETS].map((B) => {
          const r = rows.find((x) => x.mode === mode && x.reqPad === 'chunk' && x.attn === 'request' && x.chunk === C && x.budget === B);
          return kk(r).padStart(6);
        }).join(''));
      }
    }
  }
  fs.writeFileSync(get('--out', 'results/layout_ab.json'), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
