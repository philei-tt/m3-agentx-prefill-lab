#!/usr/bin/env node
// The page's sweep (default grid, SLO 10 s) of each study winner with one KV offload tier assumption varied at a time
// (as in the study's sensitivities): full curves, so the goodput point (and its requests per chunk, re-prefill and
// tier traffic) can be read against concurrency.
// Usage: JOB=<slurm job> ./on_node.sh node tools/host_curve.js [--scenarios g4_k0,...] [--workers 38]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const PAGE_GRID = [16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 1536, 2048]; // the page's DEFAULT_CONCS (the sweep extends past it)
const VARIANTS = [
  ['best (host DRAM + 16 TB SSD)', {}],
  ['no host DRAM tier (SSD only)', { hostDramGBPerGalaxy: 0 }],
  ['SSD half bandwidth', { ssdReadGBsPerGalaxy: 15.75, ssdWriteGBsPerGalaxy: 13.6 }],
  ['PCIe 181 GB/s/gx (x8 relay)', { pcieGBsPerGalaxy: 181 }],
  ['infinite cache', { cache: 'inf', hostTier: false }],
];
async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const jobs = [];
  for (const key of get('--scenarios', Object.keys(study.scenarios).join(',')).split(',')) {
    const R = study.scenarios[key];
    const best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    for (const [name, d] of VARIANTS)
      jobs.push(pool.evalCfg(`${key} ${name}`, Object.assign({}, best, d), PAGE_GRID, 10).then((r) => ({ key, name, pts: r.points })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const { key, name, pts } of res) {
    const s = summarize(pts, 10);
    console.log(`\n== ${key} ${name}: goodput ${(s.goodput / 1e3).toFixed(1)}k at C=${s.at && s.at.conc} (req/chunk ${s.at && s.at.avgSegsPerChunk.toFixed(2)})${s.atEdge ? ' [at grid edge]' : ''}`);
    for (const p of pts.slice().sort((a, b) => a.conc - b.conc))
      console.log(`  C=${String(p.conc).padStart(5)} useful ${(p.usefulTps / 1e3).toFixed(1).padStart(6)}k | p90 ${p.ttftP90.toFixed(1).padStart(6)} s | req/chunk ${p.avgSegsPerChunk.toFixed(2)} | chunk ${String(Math.round(p.avgChunkTok)).padStart(5)} | busy ${(100 * p.maxUtil).toFixed(0).padStart(3)}% | hit ${(100 * p.hitRate).toFixed(1)}% | re-prefill ${(100 * p.reprefillFrac).toFixed(0)}%`
        + ` | PCIe h2d ${(100 * (p.pcieH2DUtil || 0)).toFixed(0).padStart(3)}% ssd ${(100 * (p.ssdUtil || 0)).toFixed(0).padStart(3)}%${p === s.at ? '  <- goodput point' : ''}`);
  }
}
main();
