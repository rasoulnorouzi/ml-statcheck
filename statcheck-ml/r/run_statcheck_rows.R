# Run statcheck over one short APA string per input row and say what happened
# to each row.
#
# pipeline/25_verdicts_v4.py builds one string per gold result from the coded
# values and needs statcheck's verdict on each, or the reason there is none.
# run_statcheck.R only counts the inputs statcheck fails on; this script names
# them, so a row with no reference can be counted by cause.
#
# Usage:
#   Rscript run_statcheck_rows.R <rows.json> <out.json>
#
# The input is a JSON array of objects with `id` and `text`. The output is a
# JSON array with one object per input, in input order: `id`, `status` ("ok",
# "none" when statcheck found no result, "error" when it raised one, "several"
# when it found more than one), `message`, and for "ok" the statcheck columns.

suppressPackageStartupMessages({
  library(jsonlite)
  library(statcheck)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 2) stop("usage: Rscript run_statcheck_rows.R <rows.json> <out.json>")

rows <- fromJSON(args[1], simplifyDataFrame = FALSE)

# One row at a time: statcheck raises an error on some inputs, and inside one
# vector call that error would lose every other row.
one_row <- function(r) {
  text <- r$text
  names(text) <- r$id
  res <- tryCatch(suppressWarnings(statcheck(text, messages = FALSE)),
                  error = function(e) conditionMessage(e))
  if (is.character(res)) {
    return(list(id = r$id, status = "error", message = res))
  }
  if (is.null(res) || nrow(res) == 0) {
    return(list(id = r$id, status = "none", message = ""))
  }
  if (nrow(res) > 1) {
    return(list(id = r$id, status = "several", message = as.character(nrow(res))))
  }
  list(id = r$id, status = "ok", message = "",
       test_type = as.character(res$test_type), df1 = res$df1, df2 = res$df2,
       test_value = res$test_value, p_comp = as.character(res$p_comp),
       reported_p = res$reported_p, computed_p = res$computed_p,
       error = res$error, decision_error = res$decision_error,
       raw = as.character(res$raw))
}

out <- lapply(rows, one_row)
# digits = NA writes every double at full precision, so the Python side
# compares computed p-values without a loss from printing.
writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, na = "null", null = "null"),
           args[2], useBytes = TRUE)
cat("rows:", length(out), "\n")
