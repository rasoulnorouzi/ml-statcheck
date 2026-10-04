export const meta = {
  name: 'v4-adjudicate-disputes',
  description: 'The blind adjudicator decides every dispute batch of version 4, six at a time',
  phases: [{ title: 'Adjudicate', detail: 'one general-purpose agent per dispute batch, guideline 3.0 and the rules from the agent file' }],
}

// The adjudicator agent type was loaded when the session started, before guideline
// 3.0, so the adjudicator is a general-purpose agent that reads its instructions.
const ROOT = '/home/rass/Desktop/projects/ml-statcheck'
const batches = Array.from({ length: args.n }, (_, i) => i).filter(i => !args.only || args.only.includes(i))
if (args.reverse) batches.reverse()

function prompt(i) {
  return `You are an adjudicator. Your complete instructions are the text of the file ${ROOT}/.claude/agents/adjudicator.md below its frontmatter (the block between the two --- lines). Read that file first and follow it exactly; it is your only source of instructions.

Then read exactly one more file, the dispute batch: ${ROOT}/statcheck-ml/dataset/passages/disputes/batch_${i}.json

Write exactly one file, the output: ${ROOT}/statcheck-ml/data/annotation/passages/adjudicator/batch_${i}.json

Read nothing else. When the file is written, reply with one line: the number of objects you wrote.`
}

phase('Adjudicate')
const results = await parallel(batches.map(i => () =>
  agent(prompt(i), { label: `adjudicator batch_${i}`, phase: 'Adjudicate', agentType: 'general-purpose', model: args.model })
    .then(r => ({ i, reply: String(r || '').slice(0, 80) }))
    .catch(e => ({ i, error: String(e).slice(0, 200) }))))
const failed = results.filter(r => !r || r.error).map(r => r && r.i)
log(`adjudicator: ${batches.length - failed.length} of ${batches.length} batches returned`)
return { returned: batches.length - failed.length, failed }
