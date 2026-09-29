#!/usr/bin/env node
// Torus A/B: does running ring collectives on the galaxy's 4x4 tori change the best topology?
//   torus 'off'  = line collectives everywhere (how everything was measured)
//   torus 'full' = only [4,4] stages (a whole 4x4 torus) run rings            (the model default)
//   torus 'axes' = any 4-long axis spanning a torus row/column runs a ring    ([2,4] TP, [4,2] SP, [8,4] TP)
//   1) idle pipeline: period (bottleneck stage) and latency (sum over stages) of one 5120-token chunk
//   2) goodput on each scenario's best feature stack from results/study.json, every topology x torus mode
// Usage: JOB=<slurm job> ./on_node.sh node tools/torus_ab.js [--workers 38] [--quick]
'use strict';
const fs = require('fs'), path = require('path');
const SIM = require('../sim_core.js');
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const { STUDY } = require('../lib/paths.js');

const TOPOS = {
  4: [[16, [2, 4], 1], [8, [4, 4], 1], [4, [8, 4], 1], [16, [4, 2], 1], [8, [2, 4], 2], [4, [4, 4], 2]],
  8: [[32, [2, 4], 1], [16, [4, 4], 1], [8, [8, 4], 1], [16, [2, 4], 2], [32, [4, 2], 1], [8, [4, 4], 2]],
};
const tname = ([S, m, r]) => `${r > 1 ? r + 'x' : ''}${S}x[${m}]`;
// 'full' and 'axes' coincide on [4,4]; 'full' and 'off' coincide on everything else
const modesFor = (mesh) => (mesh[0] === 4 && mesh[1] === 4 ? ['off', 'full'] : ['off', 'axes']);

async function main() {
  const args = process.argv.slice(2);
  const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
  const study = JSON.parse(fs.readFileSync(STUDY));
  const cal = SIM.calibrate(JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'calib_data.json'))));

  console.log('== idle chunk: 5120 new tokens after 0 / 100k cached, bounded gather; period ms (system tok/s) / latency ms');
  for (const opEff of [0, 1]) for (const g of [4, 8]) for (const topo of TOPOS[g]) {
    const [S, mesh, replicas] = topo;
    const row = [];
    for (const torus of modesFor(mesh)) for (const k of [0, 102400]) {
      const plan = SIM.makePlan({ galaxies: g, stages: S, mesh, replicas, chunk: 5120, boundedDense: true, torus, opEff }, cal);
      const o = new Float64Array(S);
      SIM.chunkStageMs(plan, 5120, [{ n: 5120, na: 5120, k, cap: k + 5120 }], o);
      const mx = Math.max(...o), sum = o.reduce((a, b) => a + b, 0);
      row.push(`${torus.padEnd(4)} ${k ? '100k' : '  0k'} ${mx.toFixed(1).padStart(6)} (${(replicas * 5120 / mx).toFixed(1).padStart(5)}k) / ${sum.toFixed(0).padStart(4)}`);
    }
    console.log(`k${opEff} ${g}gx ${tname(topo).padEnd(12)} ${row.join(' | ')}`);
  }

  const pool = new Pool(Number(get('--workers', 38)));
  const concs = args.includes('--quick') ? CONCS.filter((_, i) => i % 3 === 0) : CONCS;
  const t0 = Date.now();
  const out = {};
  await Promise.all(Object.entries(study.scenarios).map(async ([key, R]) => {
    const best = R.grid[0].extra;
    out[key] = await Promise.all(TOPOS[R.base.galaxies].flatMap((topo) => modesFor(topo[1]).map((torus) => {
      const [S, mesh, replicas] = topo;
      const cfg = Object.assign(withFeatures(R.base, R.bestKeys), best, { stages: S, mesh, replicas, torus });
      return pool.evalCfg(key + tname(topo) + torus, cfg, concs, SLO).then((r) => Object.assign({ topo: tname(topo), torus }, summarize(r.points, SLO)));
    })));
  }));
  pool.close();
  for (const [key, rows] of Object.entries(out)) {
    console.log(`\n== ${key} (${study.scenarios[key].label}): best stack ${JSON.stringify(study.scenarios[key].grid[0].extra)}`);
    rows.sort((a, b) => b.goodput - a.goodput);
    for (const r of rows) console.log(`  ${r.topo.padEnd(12)} ${r.torus.padEnd(4)}  goodput ${(r.goodput / 1e3).toFixed(1).padStart(6)}k  at C=${r.at ? r.at.conc : '-'} p90 ${r.at ? r.at.ttftP90.toFixed(1) : '-'} s`);
  }
  console.log(`\n${((Date.now() - t0) / 1000).toFixed(0)} s`);
}
main();
