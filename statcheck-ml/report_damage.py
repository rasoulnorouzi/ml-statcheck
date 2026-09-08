"""Report performance for each kind of damage, across every system.

The overall score hides the thing that matters. More than half of the results
in the holdout sit in text whose symbols the conversion destroyed, and the
kinds of destruction are not equally hard.

An operator replaced by another character is recoverable, because the value is
still there. A lost decimal point is different: the number itself is wrong, so
no extractor can read it correctly and only the arithmetic can decide what it
should have been.

Usage: python report_damage.py <eval.json> [out.md]
"""
from __future__ import annotations

import json
import sys


def main(path: str, out_path: str | None = None):
    report = json.load(open(path, encoding="utf-8"))
    systems = report["systems"]
    names = list(systems)

    families = sorted({k[len("damage:"):] for s in systems.values()
                       for k in s if k.startswith("damage:")})

    def support(fam):
        key = f"damage:{fam}"
        return max((systems[n].get(key, {}).get("tp", 0) +
                    systems[n].get(key, {}).get("fn", 0)) for n in names)

    families.sort(key=support, reverse=True)

    lines = []
    lines.append(f"Holdout: {report['windows']} windows, {report['gold_results']} results, "
                 f"{report['gold_damaged']} damaged\n")
    lines.append("## Recall by kind of damage\n")

    head = "| kind of damage | n | " + " | ".join(names) + " |"
    rule = "|---|---:|" + "---|" * len(names)
    lines += [head, rule]

    for fam in families:
        key = f"damage:{fam}"
        row = [f"| {fam} | {support(fam)} "]
        for n in names:
            m = systems[n].get(key)
            row.append(f"| {m['recall']:.3f} " if m else "| - ")
        lines.append("".join(row) + "|")

    lines.append("\n## The same, as counts found of total\n")
    lines += [head, rule]
    for fam in families:
        key = f"damage:{fam}"
        row = [f"| {fam} | {support(fam)} "]
        for n in names:
            m = systems[n].get(key)
            row.append(f"| {m['tp']}/{m['tp'] + m['fn']} " if m else "| - ")
        lines.append("".join(row) + "|")

    text = "\n".join(lines)
    print(text)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
        print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
