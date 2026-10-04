export const meta = {
  name: 'v4-gold-team',
  description: 'Fable manager plans version 4 (gold set, passages, rebuilt noise, retraining, R statcheck benchmark, three ports) and drives worker agents milestone by milestone',
  phases: [
    { title: 'Plan', detail: 'the fable manager writes the plan and the first milestone', model: 'fable' },
    { title: 'Milestones', detail: 'workers run tasks by dependency; the manager reviews each task and each milestone' },
  ],
}

const BRIEF = args.brief
const BRANCH = args.branch
const AGENT_TYPES = ['adjudicator', 'adversarial-engineer', 'annotator-haiku', 'annotator-sonnet', 'annotator-opus',
  'data-engineer', 'doc-writer', 'eval-engineer', 'label-architect', 'ml-trainer', 'onnx-engineer',
  'regex-porter', 'runtime-engineer', 'stats-core', 'general-purpose']
const MODELS = ['default', 'haiku', 'sonnet', 'opus', 'fable']
const BLIND = ['annotator-haiku', 'annotator-sonnet', 'annotator-opus', 'adjudicator']

const TASK = {
  type: 'object',
  properties: {
    id: { type: 'string', description: 'unique, e.g. M1.3' },
    title: { type: 'string' },
    agentType: { type: 'string', enum: AGENT_TYPES },
    model: { type: 'string', enum: MODELS, description: "'default' keeps the model in the agent frontmatter" },
    deps: { type: 'array', items: { type: 'string' }, description: 'ids of tasks in the SAME milestone that must finish first' },
    repeat: { type: 'integer', minimum: 1, maximum: 400, description: 'fan-out count; the prompt may use {i} (0-based) and {n}' },
    review: { type: 'boolean', description: 'false only for repeated mechanical items such as rater batches; review their aggregate in a later task' },
    prompt: { type: 'string', description: 'complete and self-contained; the worker sees nothing else' },
    acceptance: { type: 'string', description: 'the evidence the manager will check' },
  },
  required: ['id', 'title', 'agentType', 'model', 'deps', 'repeat', 'review', 'prompt', 'acceptance'],
}
const MILESTONE = {
  type: 'object',
  properties: {
    id: { type: 'string' }, title: { type: 'string' }, goal: { type: 'string' },
    tasks: { type: 'array', items: TASK },
  },
  required: ['id', 'title', 'goal', 'tasks'],
}
const PLAN = {
  type: 'object',
  properties: {
    plan_file: { type: 'string' },
    milestones: { type: 'array', items: MILESTONE, description: 'every remaining milestone; only the first must have complete tasks, later ones may hold an outline of tasks that you revise at each milestone review' },
  },
  required: ['plan_file', 'milestones'],
}
const WORK = {
  type: 'object',
  properties: {
    status: { type: 'string', enum: ['done', 'blocked', 'failed'] },
    summary: { type: 'string' },
    evidence: { type: 'array', items: { type: 'string' }, description: 'commands run with their key output, file paths, real numbers' },
    files_changed: { type: 'array', items: { type: 'string' } },
  },
  required: ['status', 'summary', 'evidence', 'files_changed'],
}
const VERDICT = {
  type: 'object',
  properties: {
    accepted: { type: 'boolean' },
    reason: { type: 'string' },
    fix_prompt: { type: 'string', description: 'when not accepted: the exact instruction for one more attempt, or empty to give up' },
  },
  required: ['accepted', 'reason', 'fix_prompt'],
}
const MREVIEW = {
  type: 'object',
  properties: {
    milestone_accepted: { type: 'boolean' },
    summary: { type: 'string', description: 'what landed, with the real numbers' },
    commit: { type: 'string', description: 'the local commit id on the branch, or empty' },
    halt: { type: 'boolean', description: 'true only when the owner must decide something before work can continue' },
    halt_question: { type: 'string' },
    remaining: { type: 'array', items: MILESTONE, description: 'every milestone still to run, the next one with complete tasks; empty when the program is finished' },
  },
  required: ['milestone_accepted', 'summary', 'commit', 'halt', 'halt_question', 'remaining'],
}

const MANAGER_ROLE = `You are the MANAGER of version 4 of statcheck-ml, on branch ${BRANCH}. You do not write project code yourself; you plan, assign, verify and record. You may read anything and run read-only checks (tests, scripts, git diff, hashes) to verify a claim.

Read first: ${BRIEF} (the owner's decisions and the measurements so far), PLAN.md, CONTEXT.md, statcheck-ml/pipeline/README.md, statcheck-ml/docs/PROTOCOL.md, and every agent definition in .claude/agents/.

Rules you enforce:
- Choose for each task the agent whose definition covers it, and its model. 'default' keeps the frontmatter model, which is the cost decision of CLAUDE.md. Override only to go DOWN for mechanical work, or to 'fable' for a task that is genuinely manager-level judgement. Do not raise a model because a task is hard: split it.
- At most 6 workers run at once (the executor enforces this; the owner's limit is 10).
- Reproducibility: every stage is a numbered script in statcheck-ml/pipeline/ with --seed and explicit inputs/outputs; every number in a result file comes from a script that reads a committed file; reproduce.sh covers the new offline stages.
- Transparency: you save every worker prompt you write, verbatim, to docs/v4/prompts/<milestone-id>.md before the milestone runs (for a repeated task, the template with {i}). You append one entry per accepted task to docs/v4/LEDGER.md: id, agent, model, what changed, the evidence.
- Raters and the adjudicator are blind: their prompts name only the batch file, the output file and the guideline. Never the project, gold, the model output, or the other raters.
- Gold test papers are never trained on. Bronze and gold are reported apart, and also combined, never pooled silently.
- No transformer. No push, no release, no publication. Commit locally on ${BRANCH} at the end of each milestone, ending the message with the line: Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
- Removals: any file or code removed as unnecessary or broken is listed with its reason in the ledger.
- Shell commands outside the allow list stall an agent; delete and hash inside Python (CONTEXT.md, lessons). Long jobs (training) run in the background with checkpoints; 07_train.py resumes. Training is on the local CPU (8 cores). If the training plan would exceed about 12 hours on CPU, halt and hand the owner exact GPU terminal commands instead.
- Python is .venv/bin/python at the repository root. R is Rscript with statcheck 1.5.0. The ports are git repositories under ports/; after any spec, model or parity change rebuild parity cases, export both kits, run each port's tests, and commit the kit inside each port repository (locally).
- Prose follows ASD-STE100 as CLAUDE.md says.`

phase('Plan')
const plan = await agent(`${MANAGER_ROLE}

YOUR TASK NOW: plan version 4 and return it.

The program must cover, in milestones whose order you choose:
1. Audit and clean up: remove unnecessary, broken or useless files and code (list each with its reason); rename the bronze key 'gold' in results/eval.json and its generator to 'bronze'.
2. Gold: make 14_gold.py / 14_gold_eval.py proper stages (README table, reproduce.sh, tests); split gold by paper into a frozen gold test (all 49 bronze-holdout papers plus a stratified share of the rest) and gold-train, with the split committed.
3. Benchmark: run the ORIGINAL statcheck R package on the gold papers' full text and on the passages, as the reference baseline beside the Python regex port.
4. No-training fixes, each measured on gold test and bronze holdout before and after: the duplicate-statistic fault in Pipeline.run_text; merged passages as the processing unit (a spec change in prefilter.json, read by all three ports); finds with no test name reported as fragments, not results. Parity cases and both ports follow.
5. Annotation at passage (paragraph) level: guideline update by the label-architect (version bump, hash recorded), sampling of passages from gold-train papers, hard-negative passages from no-result papers where the model fires, and damaged papers; three blind raters; consensus where the coders are a fourth vote and are not assumed right; adjudication.
6. The noise layer rebuilt as the brief says, each family gated on gold-train or dev evidence, never on the gold test.
7. Training on the new data: the top three configurations (gru-crf, gru-softmax, lstm-softmax) with seeds 0-2, plus the no-noise ablation. Export the zoo and choose the shipped model by the pre-registered rule you write before training.
8. Evaluation: gold test, bronze holdout, and both combined, with paper-level bootstrap and paired tests, against original statcheck (R) and the old shipped model; the report template and generated report updated.
9. Ports: kits re-exported, R and web tests green, the Python package tests green.
10. Documentation: PLAN.md (a version 4 phase table with agent and model per phase), CONTEXT.md (decisions and findings), README where numbers changed.

Write the plan to docs/superpowers/plans/2026-10-04-v4-gold.md, add the version 4 phase table to PLAN.md with status 'ready', and write the prompts of the first milestone to docs/v4/prompts/. Return every milestone; the first one with complete, self-contained task prompts and acceptance evidence.`,
  { label: 'manager: plan', phase: 'Plan', model: 'fable', effort: 'high', agentType: 'general-purpose', schema: PLAN })

if (!plan) return { error: 'the manager returned no plan' }
log(`plan: ${plan.plan_file}, ${plan.milestones.length} milestones`)

function promptFor(t, i, extra) {
  const body = t.prompt.split('{i}').join(String(i)).split('{n}').join(String(t.repeat || 1))
  if (BLIND.includes(t.agentType)) return body + (extra ? `\n\n${extra}` : '')
  return `Task ${t.id}: ${t.title}. Branch ${BRANCH}; do not commit, the manager commits. Report real commands and real output as evidence.\n\n${body}` +
    (extra ? `\n\nTHE MANAGER RETURNED YOUR FIRST ATTEMPT. Fix this:\n${extra}` : '')
}

async function agent_or_null(prompt, opts) {
  try { return await agent(prompt, opts) } catch (e) { log(`${opts.label}: no structured result (${String(e).slice(0, 120)})`); return null }
}

// A worker that ends without its structured report must not end the whole run (M3.2
// fix, 2026-10-04): it becomes a failed attempt that the manager reviews.
async function work(t, i, label, extra) {
  const opts = { label, phase: 'Milestones', agentType: t.agentType, schema: WORK }
  if (t.model && t.model !== 'default') opts.model = t.model
  try {
    return await agent(promptFor(t, i, extra), opts)
  } catch (e) {
    return { status: 'failed', summary: `the agent returned no report: ${String(e).slice(0, 300)}`, evidence: [], files_changed: [] }
  }
}

// Milestones closed before the gate existed replay as recorded, so their cached agent
// calls still match; from M3 on, a task whose dependency was not accepted is skipped.
const UNGATED = ['M1', 'M2', 'M2b']

async function runTask(t, ms, done) {
  const deps = await Promise.all((t.deps || []).map(d => done.get(d) || Promise.resolve(null)))
  const refused = deps.filter(d => d && !d.accepted).map(d => d.id)
  if (refused.length && !UNGATED.includes(ms.id)) {
    log(`${t.id} skipped: dependency not accepted (${refused.join(', ')})`)
    return { id: t.id, title: t.title, agentType: t.agentType, model: t.model, accepted: false, skipped: true, reason: `dependency not accepted: ${refused.join(', ')}` }
  }
  const n = Math.max(1, t.repeat || 1)
  const results = await parallel(Array.from({ length: n }, (_, i) => () =>
    work(t, i, n > 1 ? `${t.id}[${i}] ${t.agentType}` : `${t.id} ${t.agentType}`)))
  const failed = results.filter(r => !r || r.status !== 'done').length
  if (!t.review) return { id: t.id, title: t.title, agentType: t.agentType, model: t.model, runs: n, failed, accepted: failed === 0, reports: n > 1 ? results.map(r => r && r.summary).slice(0, 5) : results }
  let report = results[0]
  let verdict = null
  for (let attempt = 0; attempt < 2; attempt++) {
    verdict = await agent_or_null(`${MANAGER_ROLE}

Review task ${t.id} (${t.title}) of milestone ${ms.id}. Verify the evidence yourself: open the files, rerun a cheap check, compare numbers. Do not accept a claim you did not verify.

The prompt the worker received:
${t.prompt}

Acceptance evidence required:
${t.acceptance}

The worker's report:
${JSON.stringify(report, null, 1)}`,
      { label: `manager: review ${t.id}`, phase: 'Milestones', model: 'fable', effort: 'medium', agentType: 'general-purpose', schema: VERDICT })
    if (!verdict || verdict.accepted || !verdict.fix_prompt || attempt === 1) break
    report = await work(t, 0, `${t.id} fix ${t.agentType}`, verdict.fix_prompt)
  }
  return { id: t.id, title: t.title, agentType: t.agentType, model: t.model, accepted: !!(verdict && verdict.accepted), verdict, report }
}

phase('Milestones')
let remaining = plan.milestones
const history = []
for (let round = 0; round < 14 && remaining.length; round++) {
  const ms = remaining[0]
  log(`milestone ${ms.id}: ${ms.title} — ${ms.tasks.length} tasks`)
  const done = new Map()
  for (const t of ms.tasks) done.set(t.id, runTask(t, ms, done))
  const outcomes = await Promise.all(ms.tasks.map(t => done.get(t.id)))
  const review = await agent_or_null(`${MANAGER_ROLE}

Milestone ${ms.id} (${ms.title}) has run. Goal: ${ms.goal}

Task outcomes:
${JSON.stringify(outcomes, null, 1)}

Now:
1. Verify the milestone goal with your own checks (tests, scripts, numbers in committed files).
2. Append the ledger entries to docs/v4/LEDGER.md and update the status column of the version 4 table in PLAN.md.
3. Commit locally on ${BRANCH} (and in a port repository if its kit changed). Never push.
4. Revise the remaining milestones in the light of what happened. Write the prompts of the next milestone to docs/v4/prompts/<id>.md before returning them.
5. Halt only if the owner must decide something; then put the exact question, with the options and your recommendation, in halt_question.

The milestones still planned after this one:
${JSON.stringify(remaining.slice(1).map(m => ({ id: m.id, title: m.title, goal: m.goal, tasks: m.tasks.map(t => t.id + ' ' + t.title) })), null, 1)}`,
    { label: `manager: close ${ms.id}`, phase: 'Milestones', model: 'fable', effort: 'high', agentType: 'general-purpose', schema: MREVIEW })
  history.push({ milestone: ms.id, title: ms.title, tasks: outcomes.map(o => ({ id: o.id, agent: o.agentType, model: o.model, accepted: o.accepted })), review })
  if (!review) { log(`milestone ${ms.id}: the manager review failed; stopping`); break }
  log(`milestone ${ms.id}: accepted=${review.milestone_accepted} commit=${review.commit || '-'}`)
  if (review.halt) return { halted: true, question: review.halt_question, history }
  remaining = review.remaining || []
}
return { halted: false, finished: remaining.length === 0, history, still_planned: remaining.map(m => m.id + ' ' + m.title) }
