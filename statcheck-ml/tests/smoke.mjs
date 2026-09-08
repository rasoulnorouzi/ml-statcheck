// Check that the browser port loads and runs.
//
// `parity.mjs` already executes `normalize.js`, so a fault there fails loudly.
// `extract.js` had no test at all, and it is the file a browser calls first.
// This runs it without PDF.js, by handing `joinItems` the shape PDF.js returns,
// so the test needs no network and no extra dependency.
//
// Usage: node tests/smoke.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

let failures = 0;
function check(name, condition, detail) {
  if (condition) {
    console.log(`  ok    ${name}`);
  } else {
    console.log(`  FAIL  ${name}${detail ? ' — ' + detail : ''}`);
    failures++;
  }
}

const spec = JSON.parse(fs.readFileSync(
  path.join(root, 'src', 'statcheck_ml', 'spec', 'normalize.json'), 'utf8'));
const charmap = JSON.parse(fs.readFileSync(
  path.join(root, 'src', 'statcheck_ml', 'spec', 'charmap.json'), 'utf8'));

console.log('normalize.js');
const nm = await import(
  pathToFileURL(path.join(root, 'js', 'normalize.js')).href);
check('module loads', typeof nm.createNormalizer === 'function');

const { normalize, reflow, canonicalise } = nm.createNormalizer(spec, charmap);
check('reflow runs', reflow('a b c\n') === 'a b c\n');
check('normalize runs on plain text',
      normalize('F(1, 17) = 3.5, p = .04').text === 'F(1, 17) = 3.5, p = .04');

// A character the model knows must survive, and one it does not must not.
const star = normalize('significant, * p < .05').text;
check('a character the model reads is left alone', star.includes('*'), star);
const damaged = normalize('F(1, 17) ϭ 35.72').text;
check('a character the model cannot read is renamed',
      !damaged.includes('ϭ') && /[-]/.test(damaged), damaged);

check('empty input is safe', normalize('').text === '');
check('canonicalise reports its mapping',
      canonicalise('t(9) ϭ 2.0').mapping instanceof Map);

console.log('extract.js');
const ex = await import(
  pathToFileURL(path.join(root, 'js', 'extract.js')).href);
check('module loads', typeof ex.extractText === 'function'
      && typeof ex.joinItems === 'function');

// The shape PDF.js returns: pieces with a position, and no lines.
const joined = ex.joinItems({
  items: [
    { str: 'F(1, 17)', transform: [1, 0, 0, 1, 0, 700] },
    { str: ' = 3.5', transform: [1, 0, 0, 1, 60, 700] },
    { str: 'next line', transform: [1, 0, 0, 1, 0, 680] },
  ],
});
check('pieces on one line are joined', joined.includes('F(1, 17) = 3.5'), joined);
check('a change of height starts a line', joined.split('\n').length === 2, joined);
check('an item with no text is skipped',
      ex.joinItems({ items: [{ transform: [1, 0, 0, 1, 0, 700] }] }) === '');

console.log(failures ? `\nFAIL: ${failures} check(s)` : '\nPASS');
process.exit(failures ? 1 : 0);
