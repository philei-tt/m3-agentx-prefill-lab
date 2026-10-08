// Prices and costs for the revenue and margin estimates: SIM.PRICE (OpenRouter's MiniMax-M3 list price) and SIM.COST
// unless the command line overrides them: --price-in / --price-cached / --price-out (USD per million tokens: cache
// miss, prefix hit, output), --galaxy-usd (USD per galaxy-hour), --decode-galaxies.
'use strict';
const { PRICE, COST } = require('../sim_core.js');
function fromArgv(argv, base, flags, what) {
  const p = Object.assign({}, base);
  for (const [flag, key] of flags) {
    const i = argv.indexOf(flag);
    if (i < 0) continue;
    const v = Number(argv[i + 1]);
    if (!(argv[i + 1] !== undefined && argv[i + 1] !== '' && v >= 0)) throw new Error(`${flag} needs ${what}, got ${argv[i + 1]}`);
    p[key] = v;
  }
  return p;
}
const priceFromArgv = (argv) => fromArgv(argv, PRICE, [['--price-in', 'inUsdPerM'], ['--price-cached', 'cachedUsdPerM'], ['--price-out', 'outUsdPerM']], 'a price in USD per million tokens');
const costFromArgv = (argv) => fromArgv(argv, COST, [['--galaxy-usd', 'galaxyUsdPerH'], ['--decode-galaxies', 'decodeGalaxies']], 'a number >= 0');
// e.g. "$1,290" or "$42.60"
const usd = (x) => (x == null || !isFinite(x) ? '–' : (x < 0 ? '−$' : '$') + (Math.abs(x) >= 100 ? Math.round(Math.abs(x)).toLocaleString('en-US') : Math.abs(x).toFixed(2)));
// "$0.30/M input, $0.06/M cached" (at least cents, more digits if the price has them)
const dollars = (x) => '$' + x.toFixed(Math.max(2, (String(x).split('.')[1] || '').length));
const priceTxt = (p) => `${dollars(p.inUsdPerM)}/M input, ${dollars(p.cachedUsdPerM)}/M cached`;
module.exports = { priceFromArgv, costFromArgv, usd, priceTxt };
