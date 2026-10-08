// Input-token prices for the revenue estimate: SIM.PRICE (OpenRouter's MiniMax-M3 list price) unless the command line
// overrides it with --price-in / --price-cached (USD per million input tokens, cache miss / prefix hit).
'use strict';
const { PRICE } = require('../sim_core.js');
function priceFromArgv(argv) {
  const p = Object.assign({}, PRICE);
  for (const [flag, key] of [['--price-in', 'inUsdPerM'], ['--price-cached', 'cachedUsdPerM']]) {
    const i = argv.indexOf(flag);
    if (i < 0) continue;
    const v = Number(argv[i + 1]);
    if (!(v >= 0)) throw new Error(`${flag} needs a price in USD per million tokens, got ${argv[i + 1]}`);
    p[key] = v;
  }
  return p;
}
// e.g. "$1,290" or "$42.60"
const usd = (x) => (x == null || !isFinite(x) ? '–' : '$' + (x >= 100 ? Math.round(x).toLocaleString('en-US') : x.toFixed(2)));
// "$0.30/M input, $0.06/M cached" (at least cents, more digits if the price has them)
const dollars = (x) => '$' + x.toFixed(Math.max(2, (String(x).split('.')[1] || '').length));
const priceTxt = (p) => `${dollars(p.inUsdPerM)}/M input, ${dollars(p.cachedUsdPerM)}/M cached`;
module.exports = { priceFromArgv, usd, priceTxt };
