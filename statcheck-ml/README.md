# statcheck-ml

Find reported statistical results with a learned model, then check them with
exact mathematics. A machine-learned replacement for the extraction step of
the R package `statcheck`.

## Install

    pip install "statcheck-ml[pdf] @ git+https://github.com/rasoulnorouzi/ml-statcheck.git#subdirectory=statcheck-ml"

A PyPI release follows; then `pip install "statcheck-ml[pdf]"`.

## Use

On the command line:

    statcheck-ml check paper.pdf

In Python:

```python
from statcheck_ml import Pipeline, summarise

report = Pipeline().run_pdf("paper.pdf")
print(summarise(report))
```

`Pipeline()` uses the model packaged with the install, so there is no path to
set and no checkout to make.

Full method and setup: [root README](https://github.com/rasoulnorouzi/ml-statcheck#readme).
Every number: [results/REPORT.md](https://github.com/rasoulnorouzi/ml-statcheck/blob/main/statcheck-ml/results/REPORT.md).
