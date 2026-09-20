# statcheck-ml

Find reported statistical results with a learned model, then check them with
exact mathematics. A machine-learned replacement for the extraction step of
the R package `statcheck`.

## Install

    pip install "statcheck-ml[pdf] @ git+https://github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml"

The repository is private, so pip needs a credential: `gh auth login`, a
personal access token with `repo` scope, or the `git+ssh://git@github.com/...`
form. A PyPI release follows; then `pip install "statcheck-ml[pdf]"`.

## Use

    statcheck-ml check paper.pdf

Full method and setup: [root README](https://github.com/rasoulnorouzi/ml-statcheck#readme).
Every number: [results/REPORT.md](https://github.com/rasoulnorouzi/ml-statcheck/blob/main/statcheck-ml/results/REPORT.md).
