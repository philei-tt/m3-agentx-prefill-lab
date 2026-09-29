#!/usr/bin/env node
// Batched chunks sized to their tokens (batchDynShape, traces per size) vs one static budget-sized shape.
//   A) today's 4-gx config (16x[2,4], chunk 2048) + pool + host tier
//   B) each scenario's best stack and topology from results/study.json, fixed chunk 256 and variable layout
// Usage: JOB=<slurm job> ./on_node.sh node tools/batch_shape_ab.js [--workers 38]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const BUDGETS = [4096, 8192, 16384, 32768];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const groups = [];
  const today = { galaxies: 4, stages: 16, mesh: [2, 4], chunk: 2048, split: 'auto', cache: 'pool', lanes: 4, hostTier: true, layout: 'fixed' };
  groups.push({ name: 'today 4gx 16x[2,4] C=2048 + pool + host', base: today, noBatch: today });
  for (const [key, R] of Object.entries(study.scenarios)) {
    const g = R.grid[0].extra;
    const keys = R.bestKeys.filter((k) => !['var', 'fused', 'batch'].includes(k));
    const b = Object.assign(withFeatures(R.base, keys), { stages: g.stages, mesh: g.mesh, replicas: g.replicas },
      g.arenaTokens ? { laneArena: true, arenaTokens: g.arenaTokens } : { lanes: g.lanes });
    groups.push({ name: `${key} best stack, fixed C=256`, base: Object.assign({}, b, { layout: 'fixed', chunk: 256 }), noBatch: Object.assign({}, b, { layout: 'fixed', chunk: 2048 }) });
    groups.push({ name: `${key} best stack, var layout`, base: Object.assign({}, b, { layout: 'var', chunk: 5120 }) });
  }
  const run = (id, cfg) => pool.evalCfg(id, cfg, CONCS, SLO).then((r) => summarize(r.points, SLO));
  const res = await Promise.all(groups.map(async (G) => {
    const rows = await Promise.all(BUDGETS.flatMap((B) => [true, false].map((dyn) =>
      run(`${G.name} ${B} ${dyn}`, Object.assign({}, G.base, { batch: true, budget: B, batchDynShape: dyn })).then((s) => ({ B, dyn, s })))));
    const nb = G.noBatch ? await run(`${G.name} nobatch`, Object.assign({}, G.noBatch, { batch: false })) : null;
    return { G, rows, nb };
  }));
  pool.close();
  const f = (s) => (s && s.at ? `${(s.goodput / 1e3).toFixed(1).padStart(6)}k (req/chunk ${s.at.avgSegsPerChunk.toFixed(2)}, pad ${(100 * s.at.padFrac).toFixed(0)}%)` : '     -');
  for (const { G, rows, nb } of res) {
    console.log(`\n== ${G.name}${nb ? `  | no batching: ${f(nb)}` : ''}`);
    for (const B of BUDGETS) {
      const d = rows.find((r) => r.B === B && r.dyn).s, st = rows.find((r) => r.B === B && !r.dyn).s;
      console.log(`  budget ${String(B / 1024).padStart(2)}k: sized to tokens ${f(d)} | static budget shape ${f(st)} | x${(st.goodput / Math.max(1, d.goodput)).toFixed(2)}`);
    }
  }
}
main();
