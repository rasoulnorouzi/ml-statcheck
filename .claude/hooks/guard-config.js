#!/usr/bin/env node
/**
 * PreToolUse hook for Write and Edit.
 * Blocks writes to paths that should never be produced by a tool call:
 * secrets and git internals. Exit code 2 blocks the call and sends stderr
 * back to Claude as the reason.
 *
 * Contract: the tool call payload arrives as JSON on stdin.
 * Paths are compared with forward slashes so the rules read the same on
 * Windows and POSIX.
 */
'use strict';

const path = require('path');

const PROTECTED = [
  { test: (p) => /(^|\/)\.env(\.|$)/.test(p), why: '.env files hold secrets' },
  { test: (p) => /\.(pem|key|p12|pfx)$/i.test(p), why: 'private key material' },
  { test: (p) => /(^|\/)\.git\//.test(p), why: 'git internals; use git commands instead' },
];

let raw = '';
process.stdin.on('data', (chunk) => (raw += chunk));
process.stdin.on('end', () => {
  let payload;
  try {
    payload = JSON.parse(raw || '{}');
  } catch {
    process.exit(0); // Malformed payload is not this hook's problem.
  }

  const target = payload?.tool_input?.file_path;
  if (!target) process.exit(0);

  const normalized = path.normalize(target).split(path.sep).join('/');
  const hit = PROTECTED.find((rule) => rule.test(normalized));
  if (!hit) process.exit(0);

  process.stderr.write(
    `Blocked by guard-config hook: ${normalized} is protected (${hit.why}).\n` +
      `Ask the user to edit this file directly if the change is genuinely needed.\n`
  );
  process.exit(2);
});
