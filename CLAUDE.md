# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in
this repository.

## What this repository is

`statcheck-ml` — a machine-learned replacement for the extraction step of the R
package `statcheck`, shipped as ONNX and run from three ports: a browser web app, R,
and Python.

Read `PLAN.md` before starting work. It holds the phase table, the current status of
each phase, and the design decisions. It is the only place progress is tracked.

## Layout

```
PLAN.md              phase table and design decisions; the single source of truth
statcheck-ml/        the project itself
.claude/agents/      one agent per phase, with its model fixed in frontmatter
.claude/hooks/       session inventory, write guard, frontmatter validator
scripts/             repository tooling, not project code
```

## Delegation

Each phase in `PLAN.md` names its agent. Use that agent for that phase. The `model:`
field in each agent is a deliberate cost choice: `haiku` for mechanical work, `sonnet`
for bounded engineering, `opus` for the four phases where a silent mistake propagates
everywhere — the label schema, the adversarial generator, the annotation protocol,
and the p-value mathematics.

Do not raise an agent's model because a task feels hard. Split the task instead.

## Design rules that are easy to violate

- **Machine learning finds results. It never judges them.** The p-value comparison is
  closed-form mathematics in all three languages. No model output touches the verdict.
- **The prefilter caps system recall.** Text it discards is unrecoverable. Tune it for
  recall alone, and treat its recall as a gate before any model tuning.
- **One spec, three thin ports.** The prefilter rules, the character vocabulary, and
  the p-value constants live in shared JSON. Each port reads them. Never write the
  same rule in three languages.
- **The processing unit is an overlapping window, not a sentence.** One result can
  cross a sentence boundary.
- **Never train on the gold set, and never call a machine annotation gold.** Label
  tiers are silver, bronze, and gold, defined in `PLAN.md`.
- **The ported regex keeps its faults.** It is the baseline improvement is measured
  against, so it must stay faithful rather than better.

## Prose style

Documentation and prose in this repository follow ASD-STE100 Simplified Technical
English: active voice, articles kept, one meaning per word, procedural sentences under
20 words. See the `doc-writer` agent for the full rule list.

This applies to documents and to chat responses about them. It does not apply to code
comments, to commit messages, or to names in the code.

## Commands

```
node scripts/list-extensions.js    # validate every agent, skill, and command loads
```

Exits non-zero when an extension has broken frontmatter, so it can gate a commit.
