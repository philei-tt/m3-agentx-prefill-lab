#!/usr/bin/env node
// How the goodput point moves with the p90 TTFT SLO: goodput, concurrency, requests per chunk, chunk size, p50/p90.
// Configs: today's 4-gx config + pool + host tier + batching, and each scenario's best config from results/study.json.
// 'none' = no SLO (the peak of a sweep that never stops early).
// Usage: JOB=<slurm job> ./on_node.sh node tools/slo_ab.js [--workers 38]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const SLOS = [5, 10, 20, 30, 60, 120, 1e6];
const GRID = CONCS.concat([5120, 6144, 8192]);

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const cfgs = [['today 4gx + pool + host + batch 16k', { galaxies: 4, stages: 16, mesh: [2, 4], chunk: 2048, cache: 'pool', lanes: 4, lanesOverride: true, hostTier: true, batch: true, budget: 16384, unaligned: true }]];
  for (const [key, R] of Object.entries(study.scenarios)) {
    const best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    cfgs.push([`${key} best`, best]);
    // the same with an infinite cache: how far batches fill when KV capacity no longer caps concurrency
    cfgs.push([`${key} best, infinite cache`, Object.assign({}, best, { cache: 'inf', hostTier: false })]);
  }
  const pool = new Pool(Number(get('--workers', 38)));
  const res = await Promise.all(cfgs.flatMap(([name, cfg]) => SLOS.map((slo) =>
    pool.evalCfg(`${name} ${slo}`, cfg, GRID, slo, slo >= 1e6 ? 1e9 : 4, slo >= 1e6 ? { extend: 0, refinePeak: 4 } : {}).then((r) => ({ name, slo, s: summarize(r.points, slo) })))));
  pool.close();
  for (const [name] of cfgs) {
    console.log(`\n== ${name}`);
    for (const { slo, s } of res.filter((r) => r.name === name)) {
      const a = slo >= 1e6 ? s.peak : s.at;
      if (!a) { console.log(`  SLO ${slo}: no passing point`); continue; }
      const g = slo >= 1e6 ? a.usefulTps : s.goodput;
      console.log(`  SLO ${(slo >= 1e6 ? 'none' : slo + ' s').padStart(6)}: goodput ${(g / 1e3).toFixed(1).padStart(6)}k @C=${String(a.conc).padStart(5)} | req/chunk ${a.avgSegsPerChunk.toFixed(2)} | chunk ${String(Math.round(a.avgChunkTok)).padStart(5)} tok | busy ${(100 * a.maxUtil).toFixed(0).padStart(3)}% | hit ${(100 * a.hitRate).toFixed(1)}% | p50/p90 ${a.ttftP50.toFixed(1)}/${a.ttftP90.toFixed(1)} s${s.atEdge && slo < 1e6 ? ' (at grid edge)' : ''}`);
    }
  }
}
main();
