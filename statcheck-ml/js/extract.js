// Read a PDF in the browser, and hand back text the rest of the system can use.
//
// The browser port uses PDF.js. It is the engine a browser can use without
// downloading a second binary, it is under Apache-2.0, and it reads almost as
// much as the engine the models were trained on. Measured on 198 holdout
// documents with 323 gold results, PDF.js holds 0.926 of them against 0.929 for
// PyMuPDF, and keeps 0.923 after the prefilter against 0.929.
//
// PDF.js reports text as pieces with a position and no lines at all. Where a
// line ends is therefore a decision this file makes, and every port must make
// the same one, or each port reads a different document. `joinItems` holds that
// decision and `spec/normalize.json` holds everything after it.
//
// Usage:
//   import { extractText } from './extract.js';
//   const text = await extractText(pdfjs, arrayBuffer, spec);

import { createNormalizer } from './normalize.js';

/** The vertical distance, in PDF units, that starts a new line. */
const NEW_LINE_AT = 2;

/**
 * Join the pieces PDF.js returns into lines.
 *
 * @param {object} textContent The value of `page.getTextContent()`.
 * @returns {string}
 */
export function joinItems(textContent) {
  const parts = [];
  let lastY = null;
  for (const item of textContent.items) {
    if (item.str === undefined) continue;
    const y = item.transform ? Math.round(item.transform[5]) : null;
    if (lastY !== null && y !== null && Math.abs(y - lastY) > NEW_LINE_AT) {
      parts.push('\n');
    }
    parts.push(item.str);
    if (item.hasEOL) parts.push('\n');
    lastY = y;
  }
  return parts.join('');
}

/**
 * Read every page of a PDF and return normalised text.
 *
 * The normalisation stage runs here, so the caller receives text that does not
 * depend on which engine read the file.
 *
 * @param {object} pdfjs The `pdfjs-dist` module.
 * @param {ArrayBuffer|Uint8Array} data The PDF itself.
 * @param {object} spec Parsed `spec/normalize.json`.
 * @param {(page: number, total: number) => void} [onProgress]
 * @returns {Promise<{text: string, pages: number, info: object}>}
 */
export async function extractText(pdfjs, data, spec, onProgress) {
  const { normalize } = createNormalizer(spec);
  const bytes = data instanceof Uint8Array ? data : new Uint8Array(data);
  const doc = await pdfjs.getDocument({ data: bytes, verbosity: 0 }).promise;

  let raw = '';
  try {
    for (let i = 1; i <= doc.numPages; i++) {
      const page = await doc.getPage(i);
      raw += joinItems(await page.getTextContent()) + '\n';
      page.cleanup();
      if (onProgress) onProgress(i, doc.numPages);
    }
  } finally {
    // Releasing the document is optional, and its name has changed between
    // versions. A failure here must never discard the text already read.
    try {
      if (typeof doc.destroy === 'function') await doc.destroy();
      else if (typeof doc.cleanup === 'function') await doc.cleanup();
    } catch { /* nothing to release */ }
  }

  const { text, info } = normalize(raw);
  return { text, pages: doc.numPages, info };
}
