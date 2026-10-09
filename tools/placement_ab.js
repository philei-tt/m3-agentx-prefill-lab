#!/usr/bin/env node
// Tile padding: even split + all-to-all KV write vs owner placement (each token computed on the SP rank that owns
// its KV row, no all-to-all, uneven ranks), against chunk padding at C=128. Each scenario's best stack and topology from
// results/study.json, infinite KV cache, peak useful tok/s at any TTFT (the comparison basis of
// results/layout_ab_inf_peak.json, whose best tile-padding budget per scenario is reused here).
//   batched  : chunk C=128 | tile even C=5120 | tile owner C in {128, 512, 2048, 5120}; attention seq and fused
//   unbatched: chunk | tile even | tile owner, at C in {2048, 5120}
// Usage: JOB=<slurm job> ./on_node.sh node tools/placement_ab.js [--workers 38] [--out results/placement_ab_inf_peak.json]
'use strict';
const fs = require('fs');
const path = require('path');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, laneCount, CONCS } = require('../study.js');
const { STUDY, RESULTS } = require('../lib/paths.js');

const OWNER_CHUNKS = [128, 512, 2048, 5120];

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const peak = JSON.parse(fs.readFileSync(path.join(RESULTS, 'layout_ab_inf_peak.json')));
  const pool = new Pool(Number(get('--workers', 38)));
  const concs = CONCS.concat([5120, 6144, 8192]);
  const t0 = Date.now();
  const out = {};
  await Promise.all(Object.entries(study.scenarios).map(async ([key, R]) => {
    const keys = R.bestKeys.filter((k) => k !== 'segPad' && k !== 'fused' && k !== 'batch');
    const g = R.grid[0].extra;
    const sp = g.mesh[0];
    const topo = { stages: g.stages, mesh: g.mesh, replicas: g.replicas, cache: 'inf', hostTier: false, laneArena: false };
    const budget = peak.scenarios[key].filter((r) => r.mode === 'base' && r.segPad === 'tile' && r.attn === 'fused').sort((a, b) => b.goodput - a.goodput)[0].budget;
    const jobs = [];
    for (const attn of ['seq', 'fused']) {
      const b = { batch: true, budget, attn };
      jobs.push(Object.assign({ v: 'chunk', segPad: 'chunk', chunk: 128 }, b));
      jobs.push(Object.assign({ v: 'even', segPad: 'tile', chunk: 5120 }, b));
      for (const C of OWNER_CHUNKS) if (C % (32 * sp) === 0) jobs.push(Object.assign({ v: 'owner', segPad: 'tile', placement: 'owner', chunk: C }, b));
    }
    for (const C of [2048, 5120]) {
      jobs.push({ v: 'chunk', segPad: 'chunk', chunk: C, batch: false, attn: 'seq' });
      jobs.push({ v: 'even', segPad: 'tile', chunk: C, batch: false, attn: 'seq' });
      jobs.push({ v: 'owner', segPad: 'tile', placement: 'owner', chunk: C, batch: false, attn: 'seq' });
    }
    out[key] = await Promise.all(jobs.map((j) => {
      const { v, ...d } = j;
      const cfg = laneCount(Object.assign(withFeatures(R.base, keys), topo, d));
      return pool.evalCfg(`${key} ${v} ${d.attn} C=${d.chunk} B=${d.batch ? d.budget : 0}`, cfg, concs, 1e6, 1e9, { extend: 0, refinePeak: 3 })
        .then((r) => Object.assign({ v, attn: d.attn, chunk: d.chunk, budget: d.batch ? d.budget : 0 }, summarize(r.points, 1e6)));
    }));
  }));
  pool.close();
  const res = { concs, scenarios: {} };
  const kk = (x) => (x ? (x.goodput / 1e3).toFixed(1) : '-');
  const at = (x) => (x && x.at ? ` (pad ${(100 * x.at.padFrac).toFixed(1)}%, ${x.at.avgSegsPerChunk.toFixed(1)} req/chunk)` : '');
  for (const [key, rows] of Object.entries(out)) {
    res.scenarios[key] = rows.map((r) => ({ v: r.v, attn: r.attn, chunk: r.chunk, budget: r.budget, goodput: r.goodput,
      at: r.at && { conc: r.at.conc, ttftP50: r.at.ttftP50, ttftP90: r.at.ttftP90, padFrac: r.at.padFrac, avgSegsPerChunk: r.at.avgSegsPerChunk } }));
    const S = study.scenarios[key], g = S.grid[0].extra;
    const B = rows.find((r) => r.budget).budget;
    console.log(`\n== ${key} (${S.label}), ${g.stages}x[${g.mesh}], peak useful k tok/s, infinite cache`);
    for (const attn of ['seq', 'fused']) {
      const f = (v, C) => rows.find((r) => r.budget && r.attn === attn && r.v === v && (C === undefined || r.chunk === C));
      console.log(`  batched B=${B / 1024}k, ${attn}: chunk C=128 ${kk(f('chunk'))}${at(f('chunk'))} | even C=5120 ${kk(f('even'))}${at(f('even'))}`);
      console.log('    owner: ' + OWNER_CHUNKS.map((C) => (f('owner', C) ? `C=${C} ${kk(f('owner', C))}${at(f('owner', C))}` : null)).filter(Boolean).join(' | '));
    }
    for (const C of [2048, 5120]) {
      const f = (v) => rows.find((r) => !r.budget && r.chunk === C && r.v === v);
      console.log(`  unbatched C=${C}: chunk ${kk(f('chunk'))}${at(f('chunk'))} | even ${kk(f('even'))}${at(f('even'))} | owner ${kk(f('owner'))}${at(f('owner'))}`);
    }
  }
  fs.writeFileSync(get('--out', path.join(RESULTS, 'placement_ab_inf_peak.json')), JSON.stringify(res));
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
