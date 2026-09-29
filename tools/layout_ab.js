#!/usr/bin/env node
// Fixed chunk + batching vs variable layout ("ragged": token-packed segments), and two attention-side optimisations.
// On each scenario's best feature stack and topology from results/study.json:
//   fixed : chunk C in {128..5120} x batch budget in {off, 4k..32k} (a request takes whole C-token units)
//   var   : budget in {4k..32k} (segments padded to 32*SP only), attention 'seq' or 'fused' (one call per chunk)
//   modes : base     = kvDedup (one attention call / prefix gather per request per chunk), no prefetch
//           nodedup  = one attention call and prefix gather per C-unit (a cold prefill split into many units
//                      re-gathers its prefix for every unit)
//           prefetch = KV-prefix gathers overlap the layer's non-collective compute
// Usage: JOB=<slurm job> ./on_node.sh node tools/layout_ab.js [--workers 38] [--quick] [--out results/layout_ab.json]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const CHUNKS = [128, 256, 512, 1024, 2048, 5120], BUDGETS = [4096, 8192, 16384, 32768];
const MODES = { base: {}, nodedup: { kvDedup: false }, prefetch: { prefetchKV: true } };

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const concs = args.includes('--quick') ? CONCS.filter((_, i) => i % 3 === 0) : CONCS;
  const t0 = Date.now();
  const out = {};
  await Promise.all(Object.entries(study.scenarios).map(async ([key, R]) => {
    const keys = R.bestKeys.filter((k) => k !== 'var' && k !== 'fused' && k !== 'batch');
    const g = R.grid[0].extra;
    const topo = { stages: g.stages, mesh: g.mesh, replicas: g.replicas };
    const lanes = g.arenaTokens ? { laneArena: true, arenaTokens: g.arenaTokens } : { lanes: g.lanes };
    const jobs = [];
    for (const [mode, md] of Object.entries(MODES)) {
      for (const C of CHUNKS) {
        if (mode !== 'nodedup' && C >= 1024) jobs.push({ mode, md, layout: 'fixed', attn: 'seq', chunk: C, batch: false });
        for (const B of BUDGETS) if (B > C) jobs.push({ mode, md, layout: 'fixed', attn: 'seq', chunk: C, batch: true, budget: B });
      }
      if (mode !== 'nodedup') for (const attn of ['seq', 'fused']) for (const B of BUDGETS) jobs.push({ mode, md, layout: 'var', attn, chunk: 5120, batch: true, budget: B });
    }
    out[key] = await Promise.all(jobs.map((j) => {
      const { mode, md, ...d } = j;
      const cfg = Object.assign(withFeatures(R.base, keys), topo, lanes, d, md);
      const label = `${mode} ${d.layout} ${d.attn} C=${d.chunk} B=${d.budget || 0}`;
      return pool.evalCfg(key + label, cfg, concs, SLO).then((r) => Object.assign({ mode, layout: d.layout, attn: d.attn, chunk: d.chunk, budget: d.budget || 0 }, summarize(r.points, SLO)));
    }));
  }));
  pool.close();
  const res = { concs, chunks: CHUNKS, budgets: BUDGETS, scenarios: {} };
  const kk = (x) => (x ? (x.goodput / 1e3).toFixed(1) : '-');
  for (const [key, rows] of Object.entries(out)) {
    rows.sort((a, b) => b.goodput - a.goodput);
    res.scenarios[key] = rows.map((r) => ({ mode: r.mode, layout: r.layout, attn: r.attn, chunk: r.chunk, budget: r.budget, goodput: r.goodput,
      at: r.at && { conc: r.at.conc, ttftP50: r.at.ttftP50, ttftP90: r.at.ttftP90, padFrac: r.at.padFrac, avgSegsPerChunk: r.at.avgSegsPerChunk } }));
    const S = study.scenarios[key];
    console.log(`\n== ${key} (${S.label}), ${S.grid[0].extra.stages}x[${S.grid[0].extra.mesh}]`);
    for (const mode of Object.keys(MODES)) {
      const f = (p) => rows.filter((r) => r.mode === mode && p(r))[0];
      const bf = f((r) => r.layout === 'fixed' && r.budget), vs = f((r) => r.layout === 'var' && r.attn === 'seq'), vf = f((r) => r.layout === 'var' && r.attn === 'fused');
      console.log(`  ${mode.padEnd(8)} best fixed+batch ${kk(bf)} (C=${bf && bf.chunk} B=${bf && bf.budget / 1024}k)` + (vs ? ` | var seq ${kk(vs)} (B=${vs.budget / 1024}k) | var fused ${kk(vf)} (B=${vf.budget / 1024}k)` : ''));
      console.log('    fixed:  ' + ['noB', ...BUDGETS.map((b) => 'B' + b / 1024 + 'k')].map((h) => h.padStart(6)).join(''));
      for (const C of CHUNKS) {
        console.log(`    C=${String(C).padStart(4)}` + [0, ...BUDGETS].map((B) => {
          const r = rows.find((x) => x.mode === mode && x.layout === 'fixed' && x.chunk === C && x.budget === B);
          return kk(r).padStart(6);
        }).join(''));
      }
    }
  }
  fs.writeFileSync(get('--out', 'results/layout_ab.json'), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
