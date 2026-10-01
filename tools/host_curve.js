#!/usr/bin/env node
// The page's sweep (default grid, SLO 10 s) of a study winner with paging and a 1 TB vs 8 TB host tier: full curves,
// so the goodput point (and its requests per chunk) can be read against concurrency.
// Usage: JOB=<slurm job> ./on_node.sh node tools/host_curve.js [--scenarios g4_k0,g8_k0] [--workers 38]
'use strict';
const fs = require('fs');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const PAGE_GRID = [16, 32, 64, 96, 128, 192, 256, 384, 512, 640, 768, 1024, 1536, 2048];
async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const pool = new Pool(Number(get('--workers', 38)));
  const jobs = [];
  for (const key of get('--scenarios', 'g4_k0,g8_k0').split(',')) {
    const R = study.scenarios[key];
    const best = Object.assign(withFeatures(R.base, R.bestKeys), R.grid[0].extra);
    for (const [name, d] of [['paging + host 1 TB/gx', { cache: 'paging', laneArena: false, hostGBPerGalaxy: 1024 }],
      ['paging + host 8 TB/gx', { cache: 'paging', laneArena: false, hostGBPerGalaxy: 8192 }]])
      jobs.push(pool.evalCfg(`${key} ${name}`, Object.assign({}, best, d), PAGE_GRID, 10).then((r) => ({ key, name, pts: r.points })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const { key, name, pts } of res) {
    const s = summarize(pts, 10);
    console.log(`\n== ${key} ${name}: goodput ${(s.goodput / 1e3).toFixed(1)}k at C=${s.at && s.at.conc} (req/chunk ${s.at && s.at.avgSegsPerChunk.toFixed(2)})${s.atEdge ? ' [at grid edge]' : ''}`);
    for (const p of pts.slice().sort((a, b) => a.conc - b.conc))
      console.log(`  C=${String(p.conc).padStart(5)} useful ${(p.usefulTps / 1e3).toFixed(1).padStart(6)}k | p90 ${p.ttftP90.toFixed(1).padStart(6)} s | req/chunk ${p.avgSegsPerChunk.toFixed(2)} | chunk ${String(Math.round(p.avgChunkTok)).padStart(5)} | busy ${(100 * p.maxUtil).toFixed(0).padStart(3)}% | hit ${(100 * p.hitRate).toFixed(1)}% | re-prefill ${(100 * p.reprefillFrac).toFixed(0)}%${p === s.at ? '  <- goodput point' : ''}`);
  }
}
main();
