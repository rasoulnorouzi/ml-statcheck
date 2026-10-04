"""Build an inventory of passages from gold-train papers excluding the dev split.

Passages are the units from the prefilter, split at line boundaries when they exceed
max_chars. Each passage is one JSONL record with its coverage of gold results, cascade
finds, and structural metadata for training.

Usage: python pipeline/17_passage_inventory.py data/clean/ --gold dataset/gold/gold.jsonl \\
    --split dataset/gold/split.json --dev dataset/gold/dev.json --max-chars 1500 \\
    --seed 0 --out dataset/passages/inventory.jsonl \\
    --summary dataset/passages/inventory_summary.json
"""
import argparse
import collections
import hashlib
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, os.path.realpath(os.path.join(os.path.dirname(__file__), "..", "src")))

from statcheck_ml.gold import assert_not_gold_test, load_split
from statcheck_ml.normalize import normalize
from statcheck_ml.prefilter import Prefilter
from statcheck_ml.pipeline import Pipeline
from statcheck_ml.repair_validated import repair_validated

ROOT = Path(__file__).resolve().parents[1]
NUM = re.compile(r"\d*\.?\d+")
IN_SCOPE = {"t", "F", "r", "z", "chi2", "Q"}


@dataclass
class PassageRecord:
    passage_id: str
    doc: str
    journal: str
    paper_has_results: bool
    paper_damaged: bool
    start_line: int
    end_line: int
    char_start: int
    n_chars: int
    n_lines: int
    text_sha256: str
    paper_text_sha256: str
    piece: int
    pieces: int
    coder_results: list
    cascade: dict
    flags: dict


def check_span(piece_text, anchor_s, anchor_e, orig_text, orig_anchor_s, orig_anchor_e):
    try:
        orig_nums = [float(m.group()) for m in NUM.finditer(orig_text[orig_anchor_s:orig_anchor_e])]
        piece_nums = [float(m.group()) for m in NUM.finditer(piece_text[anchor_s:anchor_e])]
        return orig_nums == piece_nums
    except (ValueError, IndexError):
        return False


def paper_pieces(text, max_chars):
    # A piece ends at a line end so a result that a line break splits stays whole.
    prefilter = Prefilter()
    units = list(prefilter.units(text))

    pieces = []

    for unit_idx, unit in enumerate(units):
        unit_text = unit.text
        unit_lines = unit_text.split("\n")

        # Split the unit into pieces if it exceeds max_chars
        pieces_list = []
        current_piece_start = 0
        current_char_count = 0

        for line_idx, line in enumerate(unit_lines):
            line_len = len(line) + 1  # +1 for newline

            if current_char_count > 0 and current_char_count + line_len > max_chars:
                # Start a new piece
                pieces_list.append((current_piece_start, line_idx - 1))
                current_piece_start = line_idx
                current_char_count = line_len
            else:
                current_char_count += line_len

        # Add the final piece
        pieces_list.append((current_piece_start, len(unit_lines) - 1))

        # Create records for each piece
        for piece_idx, (piece_start_idx, piece_end_idx) in enumerate(pieces_list):
            piece_lines = unit_lines[piece_start_idx:piece_end_idx + 1]
            piece_text = "\n".join(piece_lines)

            # Calculate positions in repaired text
            start_line = unit.start_line + piece_start_idx
            end_line = unit.start_line + piece_end_idx

            # Calculate char_start: unit.char_start + offset within unit
            piece_char_offset = sum(len(unit_lines[i]) + 1 for i in range(piece_start_idx))
            piece_char_start = unit.char_start + piece_char_offset

            pieces.append({
                "unit": unit_idx,
                "start_line": start_line,
                "end_line": end_line,
                "piece": piece_idx + 1,
                "pieces": len(pieces_list),
                "char_start": piece_char_start,
                "text": piece_text,
            })

    return pieces


def build_passages(job):
    doc, paper, clean_dir, max_chars = job

    # Read and process text
    text_path = os.path.join(clean_dir, doc)
    raw_text = open(text_path, encoding="utf-8").read()

    text, _ = normalize(raw_text)
    text, _ = repair_validated(text)
    paper_text_sha256 = hashlib.sha256(text.encode("utf-8")).hexdigest()

    # Get units from prefilter and build pieces
    pieces = paper_pieces(text, max_chars)

    # Get units for line range checking
    prefilter = Prefilter()
    units = list(prefilter.units(text))

    # Run pipeline to get cascade results and fragments, once per paper on raw text
    pipeline = Pipeline()
    cascade_report = pipeline.run_text(raw_text)
    cascade_results = cascade_report.get("results", [])
    cascade_fragments = cascade_report.get("fragments", [])

    # Build cascade dict mapping line -> (results, fragments)
    cascade_by_line = {}
    for result in cascade_results:
        line_num = result.get("line", -1)
        if line_num not in cascade_by_line:
            cascade_by_line[line_num] = {"results": [], "fragments": []}
        cascade_by_line[line_num]["results"].append(result)
    for fragment in cascade_fragments:
        line_num = fragment.get("line", -1)
        if line_num not in cascade_by_line:
            cascade_by_line[line_num] = {"results": [], "fragments": []}
        cascade_by_line[line_num]["fragments"].append(fragment)

    # Build line offsets for position tracking in repaired text
    lines = text.split("\n")
    offsets = [0]
    for ln in lines:
        offsets.append(offsets[-1] + len(ln) + 1)

    # Check which results are damaged
    paper_damaged = any(r.get("damaged") for r in paper.get("results", []))

    # Build gold results by line
    gold_by_line = {}
    unlocated_results = []
    for r in paper.get("results", []):
        line_num = r.get("line")
        if not r.get("located"):
            unlocated_results.append(r)
            continue
        if line_num is not None:
            if line_num not in gold_by_line:
                gold_by_line[line_num] = []
            gold_by_line[line_num].append(r)

    # Track results not covered by any unit, and unit statistics
    results_in_units = set()  # row numbers of processed results
    results_outside_units = set()  # row numbers of located results not in any unit
    units_split_count = 0
    largest_unit = 0

    # Build unit line ranges for later checking
    unit_line_ranges = [(unit.start_line, unit.start_line + len(unit.text.split("\n")) - 1) for unit in units]

    # Count split units
    for unit in units:
        largest_unit = max(largest_unit, len(unit.text))
        piece_count = sum(1 for p in pieces if p["unit"] == units.index(unit))
        if piece_count > 1:
            units_split_count += 1

    # Split units into passages
    passages = []

    for piece in pieces:
        unit_idx = piece["unit"]
        unit = units[unit_idx]
        unit_text = unit.text
        unit_lines = unit_text.split("\n")

        start_line = piece["start_line"]
        end_line = piece["end_line"]
        piece_char_start = piece["char_start"]
        piece_text = piece["text"]
        piece_lines = piece_text.split("\n")
        piece_idx = piece["piece"] - 1
        pieces_count = piece["pieces"]

        # Build passage_id
        passage_id = f"gp{hashlib.sha1(doc.encode()).hexdigest()[:8]}:{start_line}:{end_line}"

        # Calculate piece text hash
        text_sha256 = hashlib.sha256(piece_text.encode("utf-8")).hexdigest()

        # Collect coder results for this piece
        coder_results = []
        for line_num in range(start_line, end_line + 1):
            if line_num in gold_by_line:
                for r in gold_by_line[line_num]:
                    results_in_units.add(r.get("row"))
                    # Convert to coder_results format
                    anchor = r.get("anchor")
                    span = r.get("span")

                    # Map gold offsets to piece coordinates
                    span_in_piece = None
                    anchor_in_piece = None
                    on_blanked_line = False

                    if anchor is not None and span is not None:
                        anchor_line = line_num
                        if anchor_line < len(offsets):
                            piece_offset_for_anchor_line = sum(len(piece_lines[i]) + 1 for i in range(anchor_line - start_line))
                            try:
                                anchor_s_piece = piece_offset_for_anchor_line + (anchor[0] - offsets[anchor_line])
                                anchor_e_piece = piece_offset_for_anchor_line + (anchor[1] - offsets[anchor_line])

                                # Find span_line (the line where span starts)
                                span_line = anchor_line
                                for check_line in range(start_line, end_line + 1):
                                    if check_line + 1 < len(offsets) and offsets[check_line] <= span[0] < offsets[check_line + 1]:
                                        span_line = check_line
                                        break
                                    elif check_line + 1 >= len(offsets) and offsets[check_line] <= span[0]:
                                        span_line = check_line
                                        break

                                piece_offset_for_span_line = sum(len(piece_lines[i]) + 1 for i in range(span_line - start_line))
                                span_s_piece = piece_offset_for_span_line + (span[0] - offsets[span_line])
                                span_e_piece = piece_offset_for_span_line + (span[1] - offsets[span_line])

                                # Check if on blanked line (line length would be 0)
                                if len(unit_lines[anchor_line - unit.start_line]) == 0:
                                    on_blanked_line = True
                                    anchor_in_piece = None
                                    span_in_piece = None
                                else:
                                    anchor_in_piece = [anchor_s_piece, anchor_e_piece]
                                    span_in_piece = [span_s_piece, span_e_piece]

                                    # Verify numbers match
                                    if not check_span(piece_text, anchor_in_piece[0], anchor_in_piece[1],
                                                      text, anchor[0], anchor[1]):
                                        span_in_piece = None
                                        anchor_in_piece = None
                            except (IndexError, ValueError):
                                pass

                    # Format df1/df2: remove ONE trailing ".0"
                    df1_str = None
                    if r.get("df1") is not None:
                        s = str(r.get("df1"))
                        df1_str = s[:-2] if s.endswith(".0") else s

                    df2_str = None
                    if r.get("df2") is not None:
                        s = str(r.get("df2"))
                        df2_str = s[:-2] if s.endswith(".0") else s

                    # Format p_operator: "=" "<" ">" kept; "≤" -> "<"; "≥" -> ">"; else null
                    p_op = r.get("p_operator")
                    if p_op == "≤":
                        p_op = "<"
                    elif p_op == "≥":
                        p_op = ">"
                    if p_op not in ("=", "<", ">"):
                        p_op = None

                    # Format p_value: str(p) if p is not None else null
                    p_val = None
                    if r.get("p") is not None:
                        p_val = str(r.get("p"))

                    # Statistic: text[anchor] when located, else str(stat)
                    statistic = None
                    if anchor_in_piece is not None:
                        statistic = piece_text[anchor_in_piece[0]:anchor_in_piece[1]]
                    else:
                        statistic = str(r.get("stat"))

                    coder_result = {
                        "test_type": r.get("test_type") if r.get("test_type") in IN_SCOPE else None,
                        "test_type_coded": r.get("test_type_coded"),
                        "in_scope": r.get("test_type") in IN_SCOPE,
                        "statistic": statistic,
                        "df1": df1_str,
                        "df2": df2_str,
                        "n": None,
                        "p_operator": p_op,
                        "p_value": p_val,
                        "quote": r.get("raw"),
                        "damaged": r.get("damaged", False),
                        "confidence": "high",
                        "coder": r.get("coder"),
                        "row": r.get("row"),
                        "located": r.get("located", False),
                        "span_in_passage": span_in_piece,
                        "statistic_span_in_passage": anchor_in_piece,
                        "on_blanked_line": on_blanked_line,
                    }
                    coder_results.append(coder_result)

        # Collect cascade results and fragments for this piece
        cascade_in_piece = {"results": 0, "fragments": 0}
        for line_num in range(start_line, end_line + 1):
            if line_num in cascade_by_line:
                cascade_in_piece["results"] += len(cascade_by_line[line_num]["results"])
                cascade_in_piece["fragments"] += len(cascade_by_line[line_num]["fragments"])

        # Build flags
        has_gold = len(coder_results) > 0
        has_gold_in_scope = any(cr["in_scope"] for cr in coder_results)
        gold_damaged = any(cr["damaged"] for cr in coder_results)
        cascade_fires = cascade_in_piece["results"] > 0 or cascade_in_piece["fragments"] > 0

        record = PassageRecord(
            passage_id=passage_id,
            doc=doc,
            journal=doc.split("/", 1)[0],
            paper_has_results=paper.get("has_results", False),
            paper_damaged=paper_damaged,
            start_line=start_line,
            end_line=end_line,
            char_start=piece_char_start,
            n_chars=len(piece_text),
            n_lines=len(piece_lines),
            text_sha256=text_sha256,
            paper_text_sha256=paper_text_sha256,
            piece=piece_idx + 1,
            pieces=pieces_count,
            coder_results=coder_results,
            cascade=cascade_in_piece,
            flags={
                "has_gold": has_gold,
                "has_gold_in_scope": has_gold_in_scope,
                "gold_damaged": gold_damaged,
                "cascade_fires": cascade_fires,
            },
        )

        passages.append(record)

    # Mark located results that don't fall in any unit
    for r in paper.get("results", []):
        if not r.get("located"):
            continue
        line_num = r.get("line")
        if line_num is not None:
            # Check if this line falls in any unit range
            in_unit = False
            for start, end in unit_line_ranges:
                if start <= line_num <= end:
                    in_unit = True
                    break
            if not in_unit:
                results_outside_units.add(r.get("row"))

    # Return passages and metadata for summary
    return passages, {
        "units_split_count": units_split_count,
        "largest_unit": largest_unit,
        "unlocated_results": unlocated_results,
        "results_in_units": results_in_units,
        "results_outside_units": results_outside_units,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clean_text_dir")
    ap.add_argument("--gold", default=str(ROOT / "dataset/gold/gold.jsonl"))
    ap.add_argument("--split", default=str(ROOT / "dataset/gold/split.json"))
    ap.add_argument("--dev", default=str(ROOT / "dataset/gold/dev.json"))
    ap.add_argument("--max-chars", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0, help="unused: this stage does not sample")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "dataset/passages/inventory.jsonl"))
    ap.add_argument("--summary", default=str(ROOT / "dataset/passages/inventory_summary.json"))
    args = ap.parse_args()

    # Load data
    papers = [json.loads(l) for l in open(args.gold) if l.strip()]
    split_data = load_split(args.split)
    with open(args.dev) as f:
        dev_data = json.load(f)

    dev_set = set(dev_data["dev"])
    gold_train_set = set(split_data["gold_train"])

    # Select papers: gold_train minus dev
    selected_papers = [p for p in papers if p["doc"] in gold_train_set and p["doc"] not in dev_set]
    assert_not_gold_test(selected_papers, split_data)

    # Sort by doc
    selected_papers.sort(key=lambda p: p["doc"])

    # Prepare jobs
    jobs = [(p["doc"], p, args.clean_text_dir, args.max_chars)
            for p in selected_papers]

    # Process in parallel
    started = time.time()
    all_passages = []
    all_units_split_count = 0
    all_largest_unit = 0
    all_unlocated_results = []
    all_results_in_units = set()
    all_results_outside_units = set()

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for passages, metadata in executor.map(build_passages, jobs):
            all_passages.extend(passages)
            all_units_split_count += metadata["units_split_count"]
            all_largest_unit = max(all_largest_unit, metadata["largest_unit"])
            all_unlocated_results.extend(metadata["unlocated_results"])
            all_results_in_units.update(metadata["results_in_units"])
            all_results_outside_units.update(metadata["results_outside_units"])

    # Sort by doc, then start_line, then piece
    all_passages.sort(key=lambda p: (p.doc, p.start_line, p.piece))

    # Write inventory
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for passage in all_passages:
            record = {
                "passage_id": passage.passage_id,
                "doc": passage.doc,
                "journal": passage.journal,
                "paper_has_results": passage.paper_has_results,
                "paper_damaged": passage.paper_damaged,
                "start_line": passage.start_line,
                "end_line": passage.end_line,
                "char_start": passage.char_start,
                "n_chars": passage.n_chars,
                "n_lines": passage.n_lines,
                "text_sha256": passage.text_sha256,
                "paper_text_sha256": passage.paper_text_sha256,
                "piece": passage.piece,
                "pieces": passage.pieces,
                "coder_results": passage.coder_results,
                "cascade": passage.cascade,
                "flags": passage.flags,
            }
            f.write(json.dumps(record) + "\n")

    # Verify counts
    total_located_gold = len([r for p in selected_papers for r in p.get("results", []) if r.get("located")])
    results_not_in_units = len(all_results_outside_units)

    # Count span_check_failures and on_blanked_line properly
    span_check_failures = sum(
        sum(1 for cr in p.coder_results if cr["span_in_passage"] is None and cr["located"] and not cr.get("on_blanked_line", False))
        for p in all_passages
    )
    on_blanked_line = sum(
        sum(1 for cr in p.coder_results if cr.get("on_blanked_line", False))
        for p in all_passages
    )

    # Build summary
    summary = {
        "papers": len(selected_papers),
        "pieces": len(all_passages),
        "units_split": {
            "count": all_units_split_count,
            "largest_unit": all_largest_unit,
        },
        "passages_by_flag": {
            "has_gold": sum(1 for p in all_passages if p.flags["has_gold"]),
            "has_gold_in_scope": sum(1 for p in all_passages if p.flags["has_gold_in_scope"]),
            "gold_damaged": sum(1 for p in all_passages if p.flags["gold_damaged"]),
            "cascade_fires": sum(1 for p in all_passages if p.flags["cascade_fires"]),
        },
        "hard_negatives": sum(1 for p in all_passages if not p.paper_has_results and p.flags["cascade_fires"]),
        "damaged_no_gold": sum(1 for p in all_passages if p.paper_damaged and not p.flags["has_gold"]),
        "clean_negatives": {},
        "coder_rows": {
            "total_in_gold": total_located_gold,
            "in_scope": sum(sum(1 for cr in p.coder_results if cr["in_scope"]) for p in all_passages),
            "located_in_passage": sum(sum(1 for cr in p.coder_results if cr["span_in_passage"] is not None) for p in all_passages),
            "span_check_failures": span_check_failures,
            "on_blanked_line": on_blanked_line,
            "not_inside_any_unit": results_not_in_units,
            "unlocated": len(all_unlocated_results),
        },
    }

    # Build clean_negatives by journal
    journal_clean_negatives = collections.defaultdict(int)
    for p in all_passages:
        if not p.flags["has_gold"] and not p.flags["cascade_fires"]:
            journal_clean_negatives[p.journal] += 1
    summary["clean_negatives"] = dict(sorted(journal_clean_negatives.items()))

    # Write summary
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, sort_keys=True, indent=1) + "\n")

    # Print run time to stderr and summary to stdout
    run_seconds = round(time.time() - started, 2)
    print(json.dumps(summary, sort_keys=True, indent=1), file=sys.stdout)
    print(f"run_seconds: {run_seconds}", file=sys.stderr)


if __name__ == "__main__":
    main()
