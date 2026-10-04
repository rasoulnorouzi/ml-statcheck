export const meta = {
  name: 'v4-rate-passages',
  description: 'One blind rater model rates every passage batch of version 4, six batches at a time',
  phases: [{ title: 'Rate', detail: 'one general-purpose agent per batch, guideline 3.0 from the agent file' }],
}

// The rater agent types were loaded when the session started, before guideline 3.0,
// so each rater is a general-purpose agent that reads its instructions from the file.
const ROOT = '/home/rass/Desktop/projects/ml-statcheck'
const rater = args.rater
const batches = Array.from({ length: args.n }, (_, i) => i).filter(i => !args.only || args.only.includes(i))
if (args.reverse) batches.reverse()

function prompt(i) {
  return `You are a rater. Your complete instructions are the text of the file ${ROOT}/.claude/agents/annotator-${rater}.md below its frontmatter (the block between the two --- lines). Read that file first and follow it exactly; it is your only source of instructions.

Then read exactly one more file, the batch: ${ROOT}/statcheck-ml/dataset/passages/batches/batch_${i}.json

Write exactly one file, the output: ${ROOT}/statcheck-ml/data/annotation/passages/${rater}/batch_${i}.json

Read nothing else. When the file is written, reply with one line: the number of objects you wrote.`
}

phase('Rate')
const results = await parallel(batches.map(i => () =>
  agent(prompt(i), { label: `${rater} batch_${i}`, phase: 'Rate', agentType: 'general-purpose', model: args.model })
    .then(r => ({ i, reply: String(r || '').slice(0, 80) }))
    .catch(e => ({ i, error: String(e).slice(0, 200) }))))
const failed = results.filter(r => !r || r.error).map(r => r && r.i)
log(`${rater}: ${batches.length - failed.length} of ${batches.length} batches returned`)
return { rater, returned: batches.length - failed.length, failed }
