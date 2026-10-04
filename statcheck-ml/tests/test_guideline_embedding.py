"""Each rater agent and the adjudicator embed docs/GUIDELINE.md verbatim.

The collection stage stamps the sha256 of docs/GUIDELINE.md on every record,
but an agent reads only its own body. A record is only honest about its
guideline when the body holds the same text, so a drift fails here.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENTS = ROOT.parent / ".claude" / "agents"
GUIDELINE = ROOT / "docs" / "GUIDELINE.md"
EMBEDDING_AGENTS = ["annotator-haiku", "annotator-sonnet", "annotator-opus", "adjudicator"]


def _read(path):
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _version(guideline):
    first = guideline.split("\n", 1)[0]
    m = re.search(r", version (\d+\.\d+) \(\d{4}-\d{2}-\d{2}\)$", first)
    assert m, f"no version and date in the header line: {first!r}"
    return m.group(1)


def test_version_only_in_header_line():
    guideline = _read(GUIDELINE)
    version = _version(guideline)
    rest = guideline.split("\n", 1)[1]
    assert f"version {version}" not in rest
    assert "Guideline version:" not in rest


@pytest.mark.parametrize("agent", EMBEDDING_AGENTS)
def test_agent_embeds_guideline_verbatim(agent):
    guideline = _read(GUIDELINE)
    heading = f"## Guideline — docs/GUIDELINE.md, version {_version(guideline)}, verbatim\n\n"
    body = _read(AGENTS / f"{agent}.md")
    assert body.count(heading) == 1, f"{agent}: heading {heading.strip()!r} not found once"
    after = body.split(heading, 1)[1]
    assert after.startswith(guideline), f"{agent}: the embedded text differs from docs/GUIDELINE.md"
    assert after[len(guideline):].startswith("\n## Your output contract\n")
