#!/usr/bin/env node
// Why do batches hold fewer requests with a real cache than with an infinite one? Same stack, same concurrency,
// KV residency varied step by step from the study's best config towards the infinite cache.
// Usage: JOB=<slurm job> ./on_node.sh node tools/fill_diag.js [--workers 38]
'use strict';
const fs = require('fs');
const { Pool } = require('../lib/pool.js');
const { withFeatures } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const CASES = {
  g4_k0: [480, 744],   // real-cache peak, infinite-cache peak (tools/slo_ab.js)
  g8_k0: [1136, 1656],
};
const VARIANTS = [
  ['best (pool lanes/arena + offload tiers)', {}],
  ['paging + offload tiers', { cache: 'paging', laneArena: false }],
  ['paging + offload tiers, 1 TB SSD/gx', { cache: 'paging', laneArena: false, ssdTBPerGalaxy: 1 }],
  ['paging + offload tiers + PCIe 1 TB/s', { cache: 'paging', laneArena: false, pcieGBsPerGalaxy: 1000 }],
  ['infinite cache', { cache: 'inf', hostTier: false, laneArena: false }],
];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const jobs = [];
  for (const [key, concs] of Object.entries(CASES)) {
    const R = study.scenarios[key];
    const best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    for (const [name, d] of VARIANTS) for (const c of concs)
      jobs.push(pool.evalCfg(`${key}|${name}|${c}`, Object.assign({}, best, d), [c], 1e9, 1e9, { extend: 0, refine: 0 }).then((r) => ({ key, name, c, p: r.points[0] })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const key of Object.keys(CASES)) for (const c of CASES[key]) {
    console.log(`\n== ${key} at concurrency ${c}`);
    for (const { name, p } of res.filter((r) => r.key === key && r.c === c)) {
      const k = (x) => (x / 1e3).toFixed(1).padStart(6) + 'k';
      console.log(`  ${name.padEnd(38)} useful ${k(p.usefulTps)} proc ${k(p.processedTps)} | req/chunk ${p.avgSegsPerChunk.toFixed(2)} chunk ${String(Math.round(p.avgChunkTok)).padStart(5)} | busy ${(100 * p.maxUtil).toFixed(0).padStart(3)}% | hit ${(100 * p.hitRate).toFixed(1)}% (inf ${(100 * p.infHitRate).toFixed(1)}%) re-prefill ${(100 * p.reprefillFrac).toFixed(0).padStart(2)}% | wait-to-start ${p.laneWaitMean.toFixed(2)} s | host fetch ${((p.hostTok || 0) / 1e6).toFixed(0)}M tok | p50/p90 ${p.ttftP50.toFixed(1)}/${p.ttftP90.toFixed(1)} s`);
    }
  }
}
main();
