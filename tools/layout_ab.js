#!/usr/bin/env node
// Fixed chunk + batching vs variable layout ("ragged": token-packed segments) with and without fused attention.
// On each scenario's best feature stack and topology from results/study.json:
//   fixed : chunk C in {1024..5120} x batch budget in {off, 4k..32k} (a request takes whole C-token units)
//   var   : budget in {4k..32k} (segments padded to 32*SP only)
//   attn  : 'seq' (one attention call per request) or 'fused' (one call per chunk over all segments)
// Usage: JOB=<slurm job> ./on_node.sh node tools/layout_ab.js [--workers 38] [--quick]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const CHUNKS = [1024, 2048, 3072, 4096, 5120], BUDGETS = [4096, 8192, 16384, 32768];

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
    for (const attn of ['seq', 'fused']) {
      for (const C of CHUNKS) {
        jobs.push({ layout: 'fixed', attn, chunk: C, batch: false, label: `fixed C=${C} no batch` });
        for (const B of BUDGETS) if (B >= C && B > C) jobs.push({ layout: 'fixed', attn, chunk: C, batch: true, budget: B, label: `fixed C=${C} B=${B}` });
      }
      for (const B of BUDGETS) jobs.push({ layout: 'var', attn, chunk: 5120, batch: true, budget: B, label: `var B=${B}` });
    }
    out[key] = await Promise.all(jobs.filter((j) => j.attn === 'seq' || j.batch).map((j) => {
      const { label, ...d } = j;
      const cfg = Object.assign(withFeatures(R.base, keys), topo, lanes, d);
      return pool.evalCfg(key + label + j.attn, cfg, concs, SLO).then((r) => Object.assign({ label, attn: j.attn, layout: j.layout, chunk: j.chunk, budget: j.budget }, summarize(r.points, SLO)));
    }));
  }));
  pool.close();
  const res = { concs, scenarios: {} };
  for (const [key, rows] of Object.entries(out)) {
    rows.sort((a, b) => b.goodput - a.goodput);
    res.scenarios[key] = rows.map((r) => ({ label: r.label, attn: r.attn, layout: r.layout, chunk: r.chunk, budget: r.budget, goodput: r.goodput, at: r.at && { conc: r.at.conc, ttftP90: r.at.ttftP90, ttftP50: r.at.ttftP50, padFrac: r.at.padFrac, avgSegsPerChunk: r.at.avgSegsPerChunk } }));
    const best = (f) => rows.filter(f)[0];
    const k = (x) => (x ? `${(x.goodput / 1e3).toFixed(1)}k (${x.label}, ${x.attn}, p50/p90 ${x.at ? x.at.ttftP50.toFixed(1) + '/' + x.at.ttftP90.toFixed(1) : '-'} s, pad ${x.at ? (100 * x.at.padFrac).toFixed(0) : '-'}%)` : '-');
    console.log(`\n== ${key} (${study.scenarios[key].label}), ${study.scenarios[key].grid[0].extra.stages}x[${study.scenarios[key].grid[0].extra.mesh}]`);
    console.log('  best fixed, no batching   ', k(best((r) => r.layout === 'fixed' && !r.budget)));
    console.log('  best fixed + batch, seq   ', k(best((r) => r.layout === 'fixed' && r.budget && r.attn === 'seq')));
    console.log('  best fixed + batch, fused ', k(best((r) => r.layout === 'fixed' && r.budget && r.attn === 'fused')));
    console.log('  best var + batch, seq     ', k(best((r) => r.layout === 'var' && r.attn === 'seq')));
    console.log('  best var + batch, fused   ', k(best((r) => r.layout === 'var' && r.attn === 'fused')));
    const tab = CHUNKS.map((C) => `    C=${String(C).padStart(4)}: ` + [undefined, ...BUDGETS].map((B) => {
      const r = rows.find((x) => x.layout === 'fixed' && x.attn === 'seq' && x.chunk === C && x.budget === B);
      return `${B ? 'B' + B / 1024 + 'k' : 'noB'} ${r ? (r.goodput / 1e3).toFixed(1).padStart(5) : '    -'}`;
    }).join('  '));
    console.log('  fixed, seq attention (goodput k tok/s):\n' + tab.join('\n'));
  }
  fs.writeFileSync(get('--out', 'results/layout_ab.json'), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
