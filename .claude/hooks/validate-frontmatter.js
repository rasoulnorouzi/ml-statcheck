#!/usr/bin/env node
/**
 * PostToolUse hook for Write and Edit.
 * Every extension type in this repo is a Markdown file with YAML
 * frontmatter, and a missing or misspelled key fails silently at load time
 * (the skill/agent/command simply never appears). This hook catches that
 * at write time instead.
 *
 * Exit 2 sends stderr back to Claude so it can fix the file immediately.
 */
'use strict';

const fs = require('fs');
const path = require('path');

// Which keys each kind of extension file must declare.
const RULES = [
  { kind: 'skill', match: (p) => /\/skills\/[^/]+\/SKILL\.md$/.test(p), required: ['name', 'description'] },
  { kind: 'agent', match: (p) => /\/agents\/[^/]+\.md$/.test(p), required: ['name', 'description'] },
  { kind: 'command', match: (p) => /\/commands\/[^/]+\.md$/.test(p), required: ['description'] },
];

/** Return the top-level keys of the leading YAML frontmatter block, or null. */
function frontmatterKeys(text) {
  const m = /^---\r?\n([\s\S]*?)\r?\n---(\r?\n|$)/.exec(text);
  if (!m) return null;
  return m[1]
    .split(/\r?\n/)
    .filter((line) => /^[A-Za-z_][A-Za-z0-9_-]*\s*:/.test(line)) // top level only, no indent
    .map((line) => line.slice(0, line.indexOf(':')).trim());
}

let raw = '';
process.stdin.on('data', (chunk) => (raw += chunk));
process.stdin.on('end', () => {
  let payload;
  try {
    payload = JSON.parse(raw || '{}');
  } catch {
    process.exit(0);
  }

  const target = payload?.tool_input?.file_path;
  if (!target || !target.endsWith('.md')) process.exit(0);

  const normalized = path.normalize(target).split(path.sep).join('/');
  const rule = RULES.find((r) => r.match(normalized));
  if (!rule) process.exit(0);

  let text;
  try {
    text = fs.readFileSync(target, 'utf8');
  } catch {
    process.exit(0); // File may have been moved; nothing to validate.
  }

  const keys = frontmatterKeys(text);
  if (keys === null) {
    process.stderr.write(
      `${normalized} is a ${rule.kind} file but has no YAML frontmatter block.\n` +
        `Add one starting on line 1 with: ${rule.required.join(', ')}.\n`
    );
    process.exit(2);
  }

  const missing = rule.required.filter((k) => !keys.includes(k));
  if (missing.length) {
    process.stderr.write(
      `${normalized} (${rule.kind}) is missing required frontmatter: ${missing.join(', ')}.\n` +
        `Present keys: ${keys.join(', ') || '(none)'}.\n`
    );
    process.exit(2);
  }

  process.exit(0);
});
