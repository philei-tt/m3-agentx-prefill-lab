#!/usr/bin/env node
// Why does the SSD tier behind static slots not remove the eviction losses, and why does the paged pool on top
// still add throughput? Today's 4-gx config (16x[2,4], chunk 2048), KV residency varied, SSD 1 TB / 8 TB / 64 TB per
// galaxy: goodput point plus a fixed-concurrency comparison.
// Usage: JOB=<slurm job> ./on_node.sh node tools/tier_diag.js [--workers 38] [--conc 128,256]
'use strict';
const { Pool, summarize } = require('../lib/pool.js');
const { CONCS, SLO } = require('../study.js');

const base = { galaxies: 4, stages: 16, mesh: [2, 4], chunk: 2048, split: 'auto' };
const V = [
  ['slots', { cache: 'slots' }],
  ['slots + SSD 1 TB/gx', { cache: 'slots', hostTier: true, hostGBPerGalaxy: 1024 }],
  ['slots + SSD 8 TB/gx', { cache: 'slots', hostTier: true, hostGBPerGalaxy: 8192 }],
  ['slots + SSD 64 TB/gx', { cache: 'slots', hostTier: true, hostGBPerGalaxy: 65536 }],
  ['pool', { cache: 'pool' }],
  ['pool + SSD 1 TB/gx', { cache: 'pool', hostTier: true, hostGBPerGalaxy: 1024 }],
  ['pool + SSD 8 TB/gx', { cache: 'pool', hostTier: true, hostGBPerGalaxy: 8192 }],
  ['paging + SSD 8 TB/gx', { cache: 'paging', hostTier: true, hostGBPerGalaxy: 8192 }],
  ['infinite cache', { cache: 'inf' }],
];
(async () => {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const fixed = get('--conc', '128,256').split(',').map(Number);
  const pool = new Pool(Number(get('--workers', 38)));
  const sweeps = Promise.all(V.map(([n, d]) => pool.evalCfg(n, Object.assign({}, base, d), CONCS, SLO).then((r) => [n, summarize(r.points, SLO)])));
  const points = Promise.all(V.flatMap(([n, d]) => fixed.map((c) =>
    pool.evalCfg(`${n}@${c}`, Object.assign({}, base, d), [c], 1e9, 1e9, { extend: 0, refine: 0 }).then((r) => [n, c, r.points[0]]))));
  const [S, P] = await Promise.all([sweeps, points]);
  pool.close();
  const k = (x) => (x / 1e3).toFixed(1).padStart(6) + 'k';
  const line = (p) => `useful ${k(p.usefulTps)} processed ${k(p.processedTps)} | hit ${(100 * p.hitRate).toFixed(1)}% (inf ${(100 * p.infHitRate).toFixed(1)}%) | re-prefill ${(100 * p.reprefillFrac).toFixed(0).padStart(2)}% | SSD read ${((p.hostTok || 0) / 1e6).toFixed(0).padStart(4)}M tok | slot evictions ${String(p.slotEvictions || 0).padStart(5)} | busy ${(100 * p.maxUtil).toFixed(0).padStart(3)}% | p50/p90 ${p.ttftP50.toFixed(1)}/${p.ttftP90.toFixed(1)} s`;
  console.log('== goodput point (p90 TTFT <= 10 s)');
  for (const [n, s] of S) console.log(`  ${n.padEnd(22)} goodput ${k(s.goodput)} @C=${String(s.at ? s.at.conc : '-').padStart(4)} | ${s.at ? line(s.at) : ''}`);
  for (const c of fixed) {
    console.log(`\n== at concurrency ${c}`);
    for (const [n, cc, p] of P) if (cc === c) console.log(`  ${n.padEnd(22)} ${line(p)}`);
  }
})();
