// Check that the JavaScript port of the normalisation stage matches Python.
//
// The cases and the answers come from `parity_cases.json`, which
// `make_parity_cases.py` writes. This check needs no corpus and no Python, so
// it can gate a commit.
//
// Usage: node tests/parity.mjs

import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(here, '..');

const { createNormalizer } = await import(
  pathToFileURL(path.join(root, 'js', 'normalize.js')).href);

const spec = JSON.parse(fs.readFileSync(
  path.join(root, 'src', 'statcheck_ml', 'spec', 'normalize.json'), 'utf8'));
const charmap = JSON.parse(fs.readFileSync(
  path.join(root, 'src', 'statcheck_ml', 'spec', 'charmap.json'), 'utf8'));
const cases = JSON.parse(fs.readFileSync(
  path.join(here, 'parity_cases.json'), 'utf8')).sections.normalise;

const { normalize, reflow } = createNormalizer(spec, charmap);

let pass = 0;
const failures = [];
for (const c of cases) {
  const got = normalize(c.text).text;
  if (got === c.expected) { pass++; continue; }
  let at = null;
  for (let i = 0; i <= Math.max(got.length, c.expected.length); i++) {
    if (got[i] !== c.expected[i]) { at = i; break; }
  }
  failures.push({
    name: `${c.engine}/${c.name}`,
    reflowMatches: reflow(c.text) === c.expected_reflow,
    at,
    want: JSON.stringify((c.expected ?? '').slice(at, at + 30)),
    got: JSON.stringify((got ?? '').slice(at, at + 30)),
  });
}

console.log(`JavaScript port: ${pass}/${cases.length} cases match Python`);
for (const f of failures.slice(0, 5)) {
  console.log(`  ${f.name}  reflow matches ${f.reflowMatches}  at ${f.at}`);
  console.log(`    want ${f.want}`);
  console.log(`    got  ${f.got}`);
}
if (failures.length) {
  console.log('\nFAIL: the JavaScript port does not match the Python port.');
  process.exit(1);
}
console.log('PASS');
