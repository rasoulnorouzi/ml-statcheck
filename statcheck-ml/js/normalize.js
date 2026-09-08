// Make the text look the same whatever engine read the PDF.
//
// This is the JavaScript port of `statcheck_ml/normalize.py`. It reads the same
// `spec/normalize.json`, so the rules exist once and this file only applies
// them. Never write a rule here that is not in the spec.
//
// A browser gets PDF.js, which returns positioned pieces rather than lines, and
// a port must join them before this stage runs. Measured on 198 holdout
// documents, PDF.js holds 0.926 of the results and keeps 0.907 after the
// prefilter, against 0.929 for PyMuPDF, which is what the models were trained
// on.

/**
 * Build a normaliser from the shared spec.
 *
 * @param {object} spec Parsed `spec/normalize.json`.
 * @param {object} [charmap] Parsed `spec/charmap.json`. Every character the
 *   model can read. A character in it is never renamed, whatever position it
 *   stands in, because renaming one can only move the input away from the text
 *   the model was trained on. Without it the stage renames the footnote marker,
 *   the multiplication sign and the significance star, and costs recall.
 */
export function createNormalizer(spec, charmap) {
  const known = new Set(charmap ? Object.keys(charmap.chars) : []);
  const targetWidth = spec.target_line_width;
  const reflowTrigger = spec.reflow_trigger_width;
  const slots = spec.canonical_operator_slots;
  const operatorSite = new RegExp(spec.operator_site_pattern, 'gu');
  const keep = new Set([
    ...spec.text_characters,
    ...spec.protected_symbols,
  ]);

  /**
   * Cut joined columns back into lines of a normal width.
   *
   * A line shorter than the trigger is never touched, so text that already
   * breaks per page line passes through almost unchanged.
   */
  function reflow(text) {
    const normalised = text.replace(/\r\n/g, '\n').replace(/\r/g, '\n');
    const out = [];
    for (let line of normalised.split('\n')) {
      while (line.length > reflowTrigger) {
        // The search window is [targetWidth / 2, targetWidth), matching
        // Python's `rfind(" ", start, end)`, whose end is exclusive. Searching
        // one character wider picks a different cut and the ports disagree.
        let cut = line.lastIndexOf(' ', targetWidth - 1);
        if (cut < Math.floor(targetWidth / 2)) cut = targetWidth;
        out.push(line.slice(0, cut));
        line = line.slice(cut).replace(/^\s+/, '');
      }
      out.push(line);
    }
    return out.join('\n');
  }

  /**
   * Rename damaged operator characters to the alphabet the model knows.
   *
   * A character counts as a damaged operator when the model cannot read it AND
   * it stands where an operator belongs. Both halves are needed. The position
   * alone is not enough, because the copyright sign, the multiplication sign
   * and the significance star all stand between a letter and a digit, and the
   * model reads all three well.
   *
   * @returns {{text: string, mapping: Map<string, string>}}
   */
  function canonicalise(text) {
    const damaged = new Set();
    operatorSite.lastIndex = 0;
    let match;
    while ((match = operatorSite.exec(text)) !== null) {
      const ch = match[1];
      if (!known.has(ch) && !keep.has(ch) && !slots.includes(ch)) damaged.add(ch);
    }
    if (damaged.size === 0) return { text, mapping: new Map() };

    // A slot already used in this document keeps its meaning, so the rename
    // must not take it. Inside one document the correspondence between two
    // engines is one to one, so the free slots are enough.
    const free = slots.filter((c) => !text.includes(c));
    if (free.length === 0) return { text, mapping: new Map() };

    const count = (c) => text.split(c).length - 1;
    const order = [...damaged]
      .sort((a, b) => count(b) - count(a) || (a < b ? -1 : 1))
      .slice(0, free.length);

    const mapping = new Map();
    order.forEach((ch, i) => mapping.set(ch, free[i]));

    let out = '';
    for (const ch of text) out += mapping.get(ch) ?? ch;
    return { text: out, mapping };
  }

  /**
   * Apply every rule that makes the text engine independent.
   */
  function normalize(text) {
    const linesBefore = text.split('\n').length;
    const flowed = reflow(text);
    const { text: fixed, mapping } = canonicalise(flowed);
    const renamed = {};
    for (const [k, v] of mapping) renamed[k] = v.codePointAt(0);
    return {
      text: fixed,
      info: {
        lines_before: linesBefore,
        lines_after: fixed.split('\n').length,
        operators_renamed: renamed,
      },
    };
  }

  return { reflow, canonicalise, normalize };
}

/**
 * Join what PDF.js returns into lines, before the normaliser runs.
 *
 * PDF.js reports text as pieces with a position and no lines. A port must
 * decide where a line ends, and the rule must match this one, or the port reads
 * a different document from the one measured.
 *
 * @param {object} textContent The value of `page.getTextContent()`.
 */
export function joinPdfJsItems(textContent) {
  const parts = [];
  let lastY = null;
  for (const item of textContent.items) {
    if (item.str === undefined) continue;
    const y = item.transform ? Math.round(item.transform[5]) : null;
    // A change of vertical position of more than two units is a new line.
    if (lastY !== null && y !== null && Math.abs(y - lastY) > 2) parts.push('\n');
    parts.push(item.str);
    if (item.hasEOL) parts.push('\n');
    lastY = y;
  }
  return parts.join('');
}
