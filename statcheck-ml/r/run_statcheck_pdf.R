# Run the real statcheck on PDF files with checkPDF, which reads them with
# pdftools. This is the path a user without our text extraction takes.
#
# Usage:
#   Rscript run_statcheck_pdf.R <pdfs.json> <out.csv>
#
# The input is a JSON array of objects with `window_id` and `path`.
# The output has the columns of run_statcheck.R.

suppressPackageStartupMessages({
  library(jsonlite)
  library(statcheck)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_statcheck_pdf.R <pdfs.json> <out.csv>")

items <- fromJSON(args[1], simplifyDataFrame = FALSE)

pieces <- list()
failed <- 0L
for (it in items) {
  # statcheck returns NULL, without an error, when a text holds no result.
  # Only an error counts as a failure.
  res <- tryCatch(
    suppressWarnings(suppressMessages(checkPDF(it$path, messages = FALSE))),
    error = function(e) e
  )
  if (inherits(res, "error")) {
    failed <- failed + 1L
    cat("error:", conditionMessage(res), "\n", file = stderr())
  } else if (!is.null(res) && nrow(res) > 0) {
    res$source <- it$window_id
    pieces[[length(pieces) + 1L]] <- res
  }
}
cat("pdfs that raised an error:", failed, "\n")

found <- if (length(pieces)) do.call(rbind, pieces) else NULL

if (is.null(found) || nrow(found) == 0) {
  empty <- data.frame(
    window_id = character(0), test_type = character(0),
    df1 = numeric(0), df2 = numeric(0), test_value = numeric(0),
    p_comp = character(0), reported_p = numeric(0), computed_p = numeric(0),
    error = logical(0), decision_error = logical(0), raw = character(0)
  )
  write.csv(empty, args[2], row.names = FALSE, fileEncoding = "UTF-8")
  cat("statcheck found 0 results in", length(items), "pdfs\n")
  quit(status = if (failed > 0) 3 else 0)
}

out <- data.frame(
  window_id = as.character(found$source),
  test_type = as.character(found$test_type),
  df1 = found$df1,
  df2 = found$df2,
  test_value = found$test_value,
  p_comp = as.character(found$p_comp),
  reported_p = found$reported_p,
  computed_p = found$computed_p,
  error = found$error,
  decision_error = found$decision_error,
  raw = as.character(found$raw),
  stringsAsFactors = FALSE
)
write.csv(out, args[2], row.names = FALSE, fileEncoding = "UTF-8")
cat("statcheck found", nrow(out), "results in", length(items), "pdfs\n")
quit(status = if (failed > 0) 3 else 0)
