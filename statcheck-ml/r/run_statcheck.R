# Run the real statcheck package over a set of passages.
#
# This produces the baseline the project is measured against. The Python port
# in extract.py is a convenience, not the reference: only the R package can say
# what statcheck actually finds.
#
# Usage:
#   Rscript run_statcheck.R <windows.json> <out.csv>
#
# The input is a JSON array of objects with `window_id` and `text`.

suppressPackageStartupMessages({
  library(jsonlite)
  library(statcheck)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_statcheck.R <windows.json> <out.csv>")

windows <- fromJSON(args[1], simplifyDataFrame = FALSE)
texts <- vapply(windows, function(w) w$text, character(1))
ids <- vapply(windows, function(w) w$window_id, character(1))
names(texts) <- ids

# Run one passage at a time. statcheck raises an error on some inputs, and a
# single bad passage would otherwise end the whole run and lose every result
# found before it.
pieces <- list()
failed <- 0L
for (i in seq_along(texts)) {
  one <- texts[i]
  names(one) <- ids[i]
  res <- tryCatch(
    suppressWarnings(statcheck(one, messages = FALSE)),
    error = function(e) NULL
  )
  if (is.null(res)) {
    failed <- failed + 1L
  } else if (nrow(res) > 0) {
    pieces[[length(pieces) + 1L]] <- res
  }
}
cat("passages statcheck could not parse:", failed, "\n")

found <- if (length(pieces)) do.call(rbind, pieces) else NULL

if (is.null(found) || nrow(found) == 0) {
  empty <- data.frame(
    window_id = character(0), test_type = character(0),
    df1 = numeric(0), df2 = numeric(0), test_value = numeric(0),
    p_comp = character(0), reported_p = numeric(0), computed_p = numeric(0),
    error = logical(0), decision_error = logical(0), raw = character(0)
  )
  write.csv(empty, args[2], row.names = FALSE, fileEncoding = "UTF-8")
  cat("statcheck found 0 results in", length(texts), "passages\n")
  quit(status = 0)
}

out <- data.frame(
  window_id = as.character(found$source),
  test_type = as.character(found$test_type),
  df1 = found$df1,
  df2 = found$df2,
  test_value = found$test_value,
  p_comp = as.character(found$p_comp),   # the p operator, not test_comp
  reported_p = found$reported_p,
  computed_p = found$computed_p,
  error = found$error,
  decision_error = found$decision_error,
  raw = as.character(found$raw),
  stringsAsFactors = FALSE
)

write.csv(out, args[2], row.names = FALSE, fileEncoding = "UTF-8")
cat("statcheck found", nrow(out), "results in", length(texts), "passages\n")
cat("  passages with at least one result:", length(unique(out$window_id)), "\n")
cat("  inconsistencies:", sum(out$error, na.rm = TRUE), "\n")
cat("  decision errors:", sum(out$decision_error, na.rm = TRUE), "\n")
