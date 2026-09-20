# Check that the R port of the normalisation stage matches Python.
#
# The cases and the answers come from `parity_cases.json`, which
# `make_parity_cases.py` writes. This check needs no corpus and no Python, so it
# can gate a commit.
#
# Usage: Rscript tests/parity.R

suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
this <- sub("^--file=", "", args[grep("^--file=", args)])
here <- if (length(this)) dirname(normalizePath(this)) else getwd()
root <- normalizePath(file.path(here, ".."))

source(file.path(root, "r", "normalize.R"))

spec <- load_normalize_spec(file.path(root, "src", "statcheck_ml", "spec",
                                      "normalize.json"))
cases <- jsonlite::fromJSON(file.path(here, "parity_cases.json"),
                            simplifyVector = FALSE)$sections$normalise

first_difference <- function(a, b) {
  ca <- strsplit(a, "")[[1]]
  cb <- strsplit(b, "")[[1]]
  n <- min(length(ca), length(cb))
  if (n > 0) {
    differs <- which(ca[seq_len(n)] != cb[seq_len(n)])
    if (length(differs)) return(differs[1])
  }
  if (length(ca) == length(cb)) NA else n + 1
}

pass <- 0
fails <- list()
for (case in cases) {
  got <- normalize_text(case$text, spec)
  if (identical(got, case$expected)) {
    pass <- pass + 1
    next
  }
  at <- first_difference(got, case$expected)
  fails[[length(fails) + 1]] <- list(
    name = paste(case$engine, case$name, sep = "/"),
    at = at,
    want = if (is.na(at)) "" else substr(case$expected, at, at + 29),
    got = if (is.na(at)) "" else substr(got, at, at + 29)
  )
}

cat(sprintf("R port: %d/%d cases match Python\n", pass, length(cases)))
for (f in head(fails, 5)) {
  cat(sprintf("  %s  at %s\n    want %s\n    got  %s\n",
              f$name, as.character(f$at),
              encodeString(f$want), encodeString(f$got)))
}
if (length(fails) > 0) {
  cat("\nFAIL: the R port does not match the Python port.\n")
  quit(status = 1)
}
cat("PASS\n")
