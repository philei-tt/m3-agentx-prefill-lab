#!/usr/bin/env node
// What does each KV offload tier cost at a fixed load? Same stack, same concurrency (the study's goodput point of the
// best config, and of the same config with an infinite cache), one tier assumption varied at a time, as in the study's
// sensitivities: no host DRAM tier (SSD only), half the SSD bandwidth, PCIe 181 GB/s (x8 relay), infinite cache.
// Usage: JOB=<slurm job> ./on_node.sh node tools/fill_diag.js [--scenarios g4_k0,...] [--workers 38]
'use strict';
const fs = require('fs');
const { Pool } = require('../lib/pool.js');
const { withFeatures } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const VARIANTS = [
  ['best (pool + host DRAM + 32 TB SSD)', {}],
  ['paging + offload tiers', { cache: 'paging', laneArena: false }],
  ['no host DRAM tier (SSD only)', { hostDramGBPerGalaxy: 0 }],
  ['SSD half bandwidth', { ssdReadGBsPerGalaxy: 15.75, ssdWriteGBsPerGalaxy: 13.6 }],
  ['PCIe 181 GB/s/gx (x8 relay)', { pcieGBsPerGalaxy: 181 }],
  ['infinite cache', { cache: 'inf', hostTier: false }],
];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const keys = get('--scenarios', Object.keys(study.scenarios).join(',')).split(',');
  // the best config's goodput point and the infinite cache's (results/study.json)
  const cases = {};
  for (const key of keys) {
    const R = study.scenarios[key];
    cases[key] = [...new Set([R.grid[0].at.conc, R.sens.find((s) => s.name === 'inf cache').at.conc])];
  }
  const pool = new Pool(Number(get('--workers', 38)));
  const jobs = [];
  for (const key of keys) {
    const R = study.scenarios[key];
    const best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    for (const [name, d] of VARIANTS) for (const c of cases[key])
      jobs.push(pool.evalCfg(`${key}|${name}|${c}`, Object.assign({}, best, d), [c], 1e9, 1e9, { extend: 0, refine: 0 }).then((r) => ({ key, name, c, p: r.points[0] })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const key of keys) for (const c of cases[key]) {
    const S = study.scenarios[key], g = S.grid[0].extra;
    console.log(`\n== ${key} (${S.label}, ${g.stages}x[${g.mesh}]) at concurrency ${c}`);
    for (const { name, p } of res.filter((r) => r.key === key && r.c === c)) {
      const k = (x) => (x / 1e3).toFixed(1).padStart(6) + 'k';
      const pct = (x) => (100 * (x || 0)).toFixed(0).padStart(3) + '%';
      console.log(`  ${name.padEnd(36)} useful ${k(p.usefulTps)} proc ${k(p.processedTps)} | req/chunk ${p.avgSegsPerChunk.toFixed(2)} chunk ${String(Math.round(p.avgChunkTok)).padStart(5)} | busy ${pct(p.maxUtil)} | hit ${(100 * p.hitRate).toFixed(1)}% (inf ${(100 * p.infHitRate).toFixed(1)}%) re-prefill ${(100 * p.reprefillFrac).toFixed(0).padStart(2)}% | wait-to-start ${p.laneWaitMean.toFixed(2)} s`
        + ` | read host ${k(p.hostReadTps || 0)} ssd ${k(p.ssdReadTps || 0)} tok/s, busy PCIe h2d ${pct(p.pcieH2DUtil)} d2h ${pct(p.pcieD2HUtil)} ssd ${pct(p.ssdUtil)} | p50/p90 ${p.ttftP50.toFixed(1)}/${p.ttftP90.toFixed(1)} s`);
    }
  }
}
main();
