// Infinite cache: separate "variable chunk" from "ragged attention": fused attention on the fixed C=128 layout
const { Pool, summarize } = require('../lib/pool.js');
const { withFeatures, CONCS, SLO } = require('../study.js');
const study = require(require('../lib/paths.js').STUDY);
const BUD = { g4_k0: [16384], g8_k0: [16384], g4_k1: [32768, 65536], g8_k1: [32768, 65536] };
(async () => {
  const i = process.argv.indexOf('--workers');
  const pool = new Pool(i >= 0 ? Number(process.argv[i + 1]) : 38);
  const jobs = [];
  for (const [key, R] of Object.entries(study.scenarios)) {
    const keys = R.bestKeys.filter((k) => !['var', 'fused', 'batch'].includes(k));
    const g = R.grid[0].extra;
    const base = Object.assign(withFeatures(R.base, keys), { stages: g.stages, mesh: g.mesh, replicas: g.replicas, cache: 'inf', hostTier: false, laneArena: false, batch: true });
    for (const B of BUD[key]) for (const [v, d] of [['fixed C=128 seq', { layout: 'fixed', chunk: 128, attn: 'seq' }], ['fixed C=128 fused', { layout: 'fixed', chunk: 128, attn: 'fused' }],
      ['var seq', { layout: 'var', chunk: 5120, attn: 'seq' }], ['var fused', { layout: 'var', chunk: 5120, attn: 'fused' }], ['fixed C=128 fused + prefetch', { layout: 'fixed', chunk: 128, attn: 'fused', prefetchKV: true }]])
      jobs.push(pool.evalCfg(`${key} ${v} ${B}`, Object.assign({}, base, d, { budget: B }), CONCS.concat([5120, 6144]), SLO).then((r) => ({ key, v, B, s: summarize(r.points, SLO) })));
  }
  const res = await Promise.all(jobs);
  pool.close();
  for (const r of res) console.log(`${r.key} B=${r.B / 1024}k ${r.v.padEnd(30)} ${(r.s.goodput / 1e3).toFixed(1)}k @C=${r.s.at && r.s.at.conc} req/chunk ${r.s.at && r.s.at.avgSegsPerChunk.toFixed(2)}`);
})();
