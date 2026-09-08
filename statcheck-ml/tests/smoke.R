# Check that the R port loads and runs.
#
# `parity.R` already executes `normalize.R`, so a fault there fails loudly.
# `extract.R` had no test at all, and it is the file the R package calls first.
# This runs it without reading a PDF, so the test needs no corpus.
#
# Usage: Rscript tests/smoke.R

args <- commandArgs(trailingOnly = FALSE)
this <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(this)) dirname(normalizePath(this)) else getwd()
root <- normalizePath(file.path(here, ".."))

failures <- 0
check <- function(name, ok, detail = "") {
  if (isTRUE(ok)) {
    cat(sprintf("  ok    %s\n", name))
  } else {
    cat(sprintf("  FAIL  %s%s\n", name,
                if (nzchar(detail)) paste0(" - ", detail) else ""))
    failures <<- failures + 1
  }
}

cat("normalize.R\n")
source(file.path(root, "r", "normalize.R"))
check("functions exist",
      all(vapply(c("load_normalize_spec", "reflow", "canonicalise",
                   "normalize_text"), exists, logical(1), mode = "function")))

spec <- load_normalize_spec(file.path(root, "src", "statcheck_ml", "spec",
                                      "normalize.json"))
check("the spec loads", is.list(spec) && length(spec$known) > 100,
      paste("known characters:", length(spec$known)))
check("plain text is unchanged",
      normalize_text("F(1, 17) = 3.5, p = .04", spec) ==
        "F(1, 17) = 3.5, p = .04")

star <- normalize_text("significant, * p < .05", spec)
check("a character the model reads is left alone", grepl("*", star, fixed = TRUE),
      star)

damaged <- normalize_text("F(1, 17) ϭ 35.72", spec)
check("a character the model cannot read is renamed",
      !grepl("ϭ", damaged, fixed = TRUE), damaged)

check("empty input is safe", normalize_text("", spec) == "")

cat("extract.R\n")
source(file.path(root, "r", "extract.R"))
check("functions exist",
      all(vapply(c("read_pdf_text", "has_timeout"), exists, logical(1),
                 mode = "function")))
check("the time limit reports itself", is.logical(has_timeout()))
check("a missing file returns empty text, and does not stop the run",
      read_pdf_text(file.path(tempdir(), "no-such-file.pdf"), spec) == "")

if (failures > 0) {
  cat(sprintf("\nFAIL: %d check(s)\n", failures))
  quit(status = 1)
}
cat("\nPASS\n")
