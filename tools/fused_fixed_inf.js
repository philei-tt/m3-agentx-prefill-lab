// Infinite cache: separate tile padding from fused attention: fused attention with chunk padding at C=128
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const study = require(require('../lib/paths.js').STUDY);
// budgets: the best one for chunk C=128 with an infinite cache (results/layout_ab_inf.json), plus twice it if that is
// the largest budget tried there
const LAB = JSON.parse(require('fs').readFileSync(require('path').join(require('../lib/paths.js').RESULTS, 'layout_ab_inf.json')));
const BUD = {};
for (const [key, rows] of Object.entries(LAB.scenarios)) {
  const b = rows.find((x) => x.mode === 'base' && x.reqPad === 'chunk' && x.attn === 'seq' && x.chunk === 128).budget;
  BUD[key] = b === Math.max(...LAB.budgets) ? [b, 2 * b] : [b];
}
(async () => {
  const i = process.argv.indexOf('--workers');
  const pool = new Pool(i >= 0 ? Number(process.argv[i + 1]) : 38);
  const jobs = [];
  for (const [key, R] of Object.entries(study.scenarios)) {
    const keys = R.bestKeys.filter((k) => !['reqPad', 'fused', 'batch'].includes(k));
    const g = R.grid[0].extra;
    const base = Object.assign(withFeatures(R.base, keys), { stages: g.stages, mesh: g.mesh, replicas: g.replicas, cache: 'inf', hostTier: false, laneArena: false, batch: true });
    for (const B of BUD[key]) for (const [v, d] of [['chunk C=128 seq', { reqPad: 'chunk', chunk: 128, attn: 'seq' }], ['chunk C=128 fused', { reqPad: 'chunk', chunk: 128, attn: 'fused' }],
      ['tile seq', { reqPad: 'tile', chunk: 5120, attn: 'seq' }], ['tile fused', { reqPad: 'tile', chunk: 5120, attn: 'fused' }], ['chunk C=128 fused + prefetch', { reqPad: 'chunk', chunk: 128, attn: 'fused', prefetchKV: true }]])
      jobs.push(pool.evalCfg(`${key} ${v} ${B}`, Object.assign({}, base, d, { budget: B }), CONCS.concat([5120, 6144]), SLO).then((r) => ({ key, v, B, s: summarize(r.points, SLO) })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const r of res) console.log(`${r.key} B=${r.B / 1024}k ${r.v.padEnd(30)} ${(r.s.goodput / 1e3).toFixed(1)}k @C=${r.s.at && r.s.at.conc} req/chunk ${r.s.at && r.s.at.avgSegsPerChunk.toFixed(2)}`);
})();
