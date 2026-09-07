#!/usr/bin/env node
/**
 * Inventory every extension defined in this repo and report whether its
 * frontmatter would actually load.
 *
 * This is the verification step for the new-extension skill: a file that is
 * listed as INVALID here will silently not appear in a Claude Code session.
 *
 * Usage: node scripts/list-extensions.js
 * Exits 1 if any extension is invalid, so it can gate a commit.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const root = process.cwd();

/** Parse leading YAML frontmatter into a flat map of top-level keys. */
function frontmatter(file) {
  let text;
  try {
    text = fs.readFileSync(file, 'utf8');
  } catch {
    return null;
  }
  const m = /^---\r?\n([\s\S]*?)\r?\n---(\r?\n|$)/.exec(text);
  if (!m) return null;

  const out = {};
  for (const line of m[1].split(/\r?\n/)) {
    const km = /^([A-Za-z_][A-Za-z0-9_-]*)\s*:\s*(.*)$/.exec(line);
    if (km) out[km[1]] = km[2].trim();
  }
  return out;
}

function dirs(p) {
  try {
    return fs.readdirSync(path.join(root, p), { withFileTypes: true })
      .filter((e) => e.isDirectory()).map((e) => e.name).sort();
  } catch {
    return [];
  }
}

function files(p, ext) {
  try {
    return fs.readdirSync(path.join(root, p)).filter((f) => f.endsWith(ext)).sort();
  } catch {
    return [];
  }
}

const problems = [];

/** Print one extension row and record any frontmatter problem. */
function report(kind, rel, required, expectedName) {
  const fm = frontmatter(path.join(root, rel));
  if (!fm) {
    problems.push(`${rel}: no frontmatter block`);
    console.log(`  INVALID  ${rel}  — no frontmatter`);
    return;
  }
  const missing = required.filter((k) => !(k in fm));
  if (missing.length) {
    problems.push(`${rel}: missing ${missing.join(', ')}`);
    console.log(`  INVALID  ${rel}  — missing ${missing.join(', ')}`);
    return;
  }
  if (expectedName && fm.name !== expectedName) {
    problems.push(`${rel}: name "${fm.name}" != directory "${expectedName}"`);
    console.log(`  INVALID  ${rel}  — name "${fm.name}" != directory "${expectedName}"`);
    return;
  }
  const label = fm.name || path.basename(rel, '.md');
  console.log(`  ok       ${kind.padEnd(7)} ${label}`);
}

const scopes = [
  ['project', '.claude'],
  ...dirs('plugins').map((p) => [`plugin:${p}`, path.join('plugins', p)]),
];

for (const [scopeName, base] of scopes) {
  console.log(`\n${scopeName}  (${base})`);
  let found = 0;

  for (const name of dirs(path.join(base, 'skills'))) {
    const rel = path.join(base, 'skills', name, 'SKILL.md');
    if (!fs.existsSync(path.join(root, rel))) continue;
    report('skill', rel, ['name', 'description'], name);
    found++;
  }
  for (const f of files(path.join(base, 'agents'), '.md')) {
    report('agent', path.join(base, 'agents', f), ['name', 'description']);
    found++;
  }
  for (const f of files(path.join(base, 'commands'), '.md')) {
    report('command', path.join(base, 'commands', f), ['description']);
    found++;
  }
  if (!found) console.log('  (no extensions)');
}

console.log('');
if (problems.length) {
  console.log(`${problems.length} problem(s) found.`);
  process.exit(1);
}
console.log('All extensions valid.');
