# Read a PDF the way the R port must, and hand back text the rest can use.
#
# The R port uses pdftools, which bundles poppler. R has no other maintained
# PDF reader, so this is a constraint and not a choice. Measured on 198 holdout
# documents with 323 gold results, pdftools holds 0.916 of them and keeps 0.912
# after the prefilter, against 0.929 for the Python port.
#
# pdf_text() is not the same as the pdftotext command line. It reads more. Never
# measure the R port by borrowing a pdftotext number.
#
# pdftools hangs on some files, twice in the 198 measured, where the command
# line reads the same files without trouble. Each document therefore gets a time
# limit, and a document that passes it returns empty text so the caller can
# continue with the rest of the corpus.
#
# R.utils supplies the time limit. It belongs in Suggests, not Imports: without
# it the reader still works and only loses the limit.

# `normalize.R` must be sourced first. Finding it from inside a sourced file is
# unreliable in R, and guessing wrongly is worse than asking the caller.
if (!exists("normalize_text", mode = "function")) {
  stop("source normalize.R before extract.R", call. = FALSE)
}

#' Is the time limit available?
has_timeout <- function() {
  requireNamespace("R.utils", quietly = TRUE)
}

#' Read one PDF and return normalised text.
#'
#' @param path The PDF.
#' @param spec The parsed normalisation spec.
#' @param timeout Seconds to allow one document. Ignored when R.utils is absent.
#' @return Normalised text, or "" when the document could not be read.
read_pdf_text <- function(path, spec, timeout = 60) {
  read_it <- function() paste(pdftools::pdf_text(path), collapse = "")

  txt <- tryCatch({
    if (has_timeout()) {
      R.utils::withTimeout(read_it(), timeout = timeout, onTimeout = "silent")
    } else {
      # Without R.utils there is no limit, and a bad document can stop the run.
      # Warn once so the cause is visible rather than mysterious.
      if (!isTRUE(getOption("statcheckml.timeout.warned"))) {
        warning("R.utils is not installed, so no time limit applies to ",
                "pdftools. Install R.utils to stop one document halting a run.",
                call. = FALSE)
        options(statcheckml.timeout.warned = TRUE)
      }
      read_it()
    }
  }, error = function(e) "")

  if (is.null(txt) || length(txt) == 0 || is.na(txt[1])) return("")
  normalize_text(txt, spec)
}
