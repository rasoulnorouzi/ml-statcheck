#!/usr/bin/env node
/**
 * SessionStart hook.
 * Lists the agents defined in this repo and points at the plan, so a fresh
 * session knows what exists without globbing for it.
 *
 * Contract: stdout is injected into the session as additional context.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const root = process.env.CLAUDE_PROJECT_DIR || process.cwd();

/** List files matching `ext` directly under `dir`, or [] if dir is absent. */
function list(dir, ext) {
  try {
    return fs.readdirSync(path.join(root, dir)).filter((f) => f.endsWith(ext)).sort();
  } catch {
    return [];
  }
}

/** List immediate subdirectories of `dir` that contain a SKILL.md. */
function skills(dir) {
  const abs = path.join(root, dir);
  try {
    return fs
      .readdirSync(abs, { withFileTypes: true })
      .filter((e) => e.isDirectory() && fs.existsSync(path.join(abs, e.name, 'SKILL.md')))
      .map((e) => e.name)
      .sort();
  } catch {
    return [];
  }
}

const agents = list('.claude/agents', '.md').map((f) => f.replace(/\.md$/, ''));
const cmds = list('.claude/commands', '.md').map((f) => f.replace(/\.md$/, ''));
const skl = skills('.claude/skills');

const lines = [
  'practice_claude — statcheck-ml project.',
  `  Agents:   ${agents.length ? agents.join(', ') : '(none)'}`,
  `  Skills:   ${skl.length ? skl.join(', ') : '(none)'}`,
  `  Commands: ${cmds.length ? cmds.join(', ') : '(none)'}`,
  '  Plan and phase status: PLAN.md. Working rules: CLAUDE.md.',
];

process.stdout.write(lines.join('\n') + '\n');
