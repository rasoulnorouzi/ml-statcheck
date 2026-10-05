import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "ports_stage", str(Path(__file__).parent.parent / "pipeline" / "13_ports.py"))
ports_stage = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ports_stage)


@pytest.mark.parametrize("summary, expected", [
    ("....\n9 passed in 1.20s\n", (9, 0)),
    ("..F\n1 failed, 2 passed in 0.5s\n", (2, 1)),
    ("1 failed, 2 passed, 1 error in 0.5s\n\n", (2, 2)),
    ("3 passed, 1 skipped, 2 warnings in 4.00s", (3, 0)),
])
def test_pytest_counts(summary, expected):
    assert ports_stage.pytest_counts(summary) == expected


def test_pytest_counts_rejects_unreadable_output():
    with pytest.raises(SystemExit):
        ports_stage.pytest_counts("no tests ran in 0.01s")
