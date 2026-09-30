#!/usr/bin/env node
// Assemble the single-file artifact: template + sim core + calibration + traffic + study summary.
// Usage: node build_artifact.js [--study results/study.json] [--data DIR] [--out FILE]
'use strict';
const fs = require('fs'), path = require('path');
const SIM = require('./sim_core.js');
const { FEATURES, withFeatures } = require('./study.js');
const PRESETS = require('./presets.js');

const args = process.argv.slice(2);
const get = (k, d) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : d; };
const P = require('./lib/paths.js');
const dataDir = get('--data', P.DATA);
const studyF = get('--study', P.STUDY);
// --standalone wraps the page in a full HTML document (for server.js); the artifact host adds its own skeleton
const standalone = args.includes('--standalone');
const outF = get('--out', standalone ? path.join(P.DIST, 'index.html') : path.join(path.dirname(P.RESULTS), 'artifact', 'm3_agentx_lab.html'));

const calData = JSON.parse(fs.readFileSync(path.join(__dirname, 'calib_data.json')));
const cal = SIM.calibrate(calData);
delete cal.fit.samples;
const header = JSON.parse(fs.readFileSync(path.join(dataDir, 'traffic.json')));
const bin = fs.readFileSync(path.join(dataDir, 'traffic.bin'));
const study = JSON.parse(fs.readFileSync(studyF));

// ---- rough implementation complexity (low / med / high) + what it takes, per feature
const SCOPE = require('./lib/scope.js');

// ---- trim the study for the page
const slim = (a) => (a ? { conc: a.conc, usefulTps: a.usefulTps, processedTps: a.processedTps, ttftP50: a.ttftP50, ttftP90: a.ttftP90, hitRate: a.hitRate, infHitRate: a.infHitRate, reprefillFrac: a.reprefillFrac, padFrac: a.padFrac, maxUtil: a.maxUtil, avgChunkTok: a.avgChunkTok, avgSegsPerChunk: a.avgSegsPerChunk } : null);
const DETAILS = require('./feature_details.js');
// studies run before the host-DRAM tier was renamed to the SSD tier store the old sensitivity names
const SENS_RENAME = { 'host 0.5 TB/gx': 'SSD 0.5 TB/gx', 'host 2 TB/gx': 'SSD 2 TB/gx', 'PCIe 16 GB/s/gx': 'SSD 16 GB/s/gx', 'PCIe 256 GB/s/gx': 'SSD 256 GB/s/gx' };
const S = { slo: study.slo, features: FEATURES.map((f) => ({ key: f.key, name: f.name, requires: f.requires || [], scope: SCOPE[f.key], detail: DETAILS[f.key] })), scenarios: {} };
const presets = PRESETS.list.map((p) => ({ name: p.name, desc: p.desc, cfg: p.cfg }));
for (const [key, R] of Object.entries(study.scenarios)) {
  S.scenarios[key] = {
    label: R.label, base: R.base, bestKeys: R.bestKeys,
    greedy: R.greedy.map((g) => ({ add: g.add, goodput: g.goodput, inf: g.inf ? g.inf.goodput : null, at: slim(g.at), counts: g.plan && g.plan.counts, capTok: g.plan && g.plan.capTok, candidates: (g.candidates || []).map((c) => ({ key: c.key, goodput: c.goodput })) })),
    full: R.full ? { goodput: R.full.goodput, at: slim(R.full.at) } : null,
    loo: R.loo.map((l) => ({ remove: l.remove, goodput: l.goodput })),
    grid: R.grid.slice(0, 12).map((g) => ({ extra: g.extra, goodput: g.goodput, at: slim(g.at), counts: g.plan.counts, capTok: g.plan.capTok, poolTok: g.plan.poolTok })),
    sens: (R.sens || []).map((s) => ({ name: SENS_RENAME[s.name] || s.name, goodput: s.goodput, peak: s.peak, at: slim(s.at) })),
    curves: {
      base: R.greedy[0].points.map(slim),
      best: R.grid[0].points.map(slim),
    },
  };
  const best = R.grid[0];
  const cfg = Object.assign(withFeatures(R.base, R.bestKeys), best.extra);
  presets.push({ name: `best-${key}`, desc: `Study winner: ${R.label}. Features: ${R.bestKeys.join(', ')}; ${best.extra.stages} x [${best.extra.mesh}]${best.extra.replicas > 1 ? ' x ' + best.extra.replicas + ' replicas' : ''}`, cfg });
  if (key === 'g4_k0') {
    const p0 = R.greedy.slice(1, 4).map((g) => g.add);
    presets.push({ name: 'next-3-features-4gx', desc: `4 gx, today's kernels + the first three roadmap features (${p0.join(', ')})`, cfg: withFeatures(R.base, p0) });
  }
}

// ---- validation table (#57827 cells)
const val = [];
for (const run of ['A', 'B', 'C']) {
  const r = calData.pipeline[run]; const counts = r.layers.split(',').map(Number);
  for (const row of calData.tables[run]) {
    const m = SIM.matrixCell(cal, { chunk: r.chunk, split: counts, stages: 16 }, row.cached, row.new, 4, 30);
    val.push([run, row.cached, row.new, Math.round(m.newTps), row.loaded_steady_new_tps, Math.round(m.idleTtftMs), row.idle_ttft_ms_median]);
  }
}

const tpl = fs.readFileSync(path.join(__dirname, 'artifact', 'template.html'), 'utf8');
const simSrc = fs.readFileSync(path.join(__dirname, 'sim_core.js'), 'utf8');
const js = (x) => JSON.stringify(x).replace(/</g, '\\u003c');
let html = tpl
  .replace('/*__SIM_CORE__*/', () => simSrc)
  .replace('/*__SWEEP__*/', () => fs.readFileSync(path.join(__dirname, 'lib', 'sweep.js'), 'utf8'))
  .replace('"__CAL__"', () => js(cal))
  .replace('"__STUDY__"', () => js(S))
  .replace('"__PRESETS__"', () => js(presets))
  .replace('"__VALIDATION__"', () => js(val))
  .replace('"__TRAFFIC_HEADER__"', () => js({ layout: header.layout, stats: header.stats }))
  .replace('__TRAFFIC_B64__', () => bin.toString('base64'));
if (standalone) {
  html = '<!doctype html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
    + '<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">\n'
    + '<style>:root{color-scheme:light}body{margin:0}[hidden]{display:none!important}img{max-width:100%}</style>\n'
    + '</head>\n<body>\n' + html + '\n</body>\n</html>\n';
}
fs.mkdirSync(path.dirname(outF), { recursive: true });
fs.writeFileSync(outF, html);
console.log('wrote', outF, (html.length / 1e6).toFixed(2), 'MB', presets.length, 'presets');
