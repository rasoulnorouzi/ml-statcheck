# Make the text look the same whatever engine read the PDF.
#
# This is the R port of `statcheck_ml/normalize.py`. It reads the same
# `spec/normalize.json`, so the rules exist once and this file only applies
# them. Never write a rule here that is not in the spec.
#
# R reads PDF files with pdftools, which bundles poppler. Poppler returns a
# whole column as one line, and every prefilter rule measures one line. Without
# this stage the R port keeps 0.793 of the holdout results against 0.929 for the
# Python port. With it the R port keeps 0.879.

library(jsonlite)

#' Load the shared normalisation spec.
#'
#' @param path Path to `spec/normalize.json`.
#' @param charmap_path Path to `spec/charmap.json`. Every character the model
#'   can read. A character in it is never renamed, whatever position it stands
#'   in, because renaming one can only move the input away from the text the
#'   model was trained on. Without it the stage renames the footnote marker, the
#'   multiplication sign and the significance star, and costs recall.
load_normalize_spec <- function(path, charmap_path = NULL) {
  spec <- jsonlite::fromJSON(path, simplifyVector = TRUE)
  spec$keep <- unique(c(
    strsplit(spec$text_characters, "")[[1]],
    strsplit(spec$protected_symbols, "")[[1]]
  ))
  if (is.null(charmap_path)) {
    charmap_path <- file.path(dirname(path), "charmap.json")
  }
  spec$known <- if (file.exists(charmap_path)) {
    names(jsonlite::fromJSON(charmap_path, simplifyVector = FALSE)$chars)
  } else {
    character(0)
  }
  spec
}

#' Cut joined columns back into lines of a normal width.
#'
#' A line shorter than the trigger is never touched, so text from an engine that
#' already breaks lines per page line passes through almost unchanged.
reflow <- function(text, spec) {
  text <- gsub("\r\n", "\n", text, fixed = TRUE)
  text <- gsub("\r", "\n", text, fixed = TRUE)
  target <- spec$target_line_width
  trigger <- spec$reflow_trigger_width

  # `strsplit` drops the empty piece after a final separator and Python's
  # `split` keeps it. Without this the R port loses the last newline of every
  # document, and the three ports return texts of different length.
  lines <- strsplit(text, "\n", fixed = TRUE)[[1]]
  if (grepl("\n$", text)) lines <- c(lines, "")

  out <- character(0)
  for (line in lines) {
    while (nchar(line) > trigger) {
      # `cut` counts from zero, so that this matches Python's
      # `line.rfind(" ", target %/% 2, target)` exactly. R counts from one, and
      # mixing the two conventions keeps the space the cut lands on, which made
      # the R port disagree with the other two.
      window <- substr(line, 1, target)          # positions 1..target
      spaces <- gregexpr(" ", window, fixed = TRUE)[[1]]
      spaces <- spaces[spaces > 0] - 1           # now counted from zero
      spaces <- spaces[spaces >= target %/% 2]   # Python's search starts here
      cut <- if (length(spaces) == 0) target else max(spaces)
      out <- c(out, substr(line, 1, cut))        # Python's line[:cut]
      line <- sub("^\\s+", "", substr(line, cut + 1, nchar(line)))
    }
    out <- c(out, line)
  }
  paste(out, collapse = "\n")
}

#' Rename damaged operator characters to the alphabet the model knows.
#'
#' A character counts as a damaged operator when the model cannot read it AND it
#' stands where an operator belongs. Both halves are needed. The position alone
#' is not enough, because the copyright sign, the multiplication sign and the
#' significance star all stand between a letter and a digit, and the model reads
#' all three well.
canonicalise <- function(text, spec) {
  slots <- spec$canonical_operator_slots
  matches <- gregexpr(spec$operator_site_pattern, text, perl = TRUE)[[1]]
  if (matches[1] == -1) return(list(text = text, mapping = character(0)))

  starts <- attr(matches, "capture.start")
  lengths <- attr(matches, "capture.length")
  chars <- substring(text, starts, starts + lengths - 1)
  chars <- chars[nzchar(chars)]

  damaged <- unique(chars[!(chars %in% spec$known) & !(chars %in% spec$keep) &
                            !(chars %in% slots)])
  if (length(damaged) == 0) return(list(text = text, mapping = character(0)))

  # A slot already used in this document keeps its meaning, so the rename must
  # not take it. Inside one document the correspondence between two engines is
  # one to one, so the free slots are enough.
  used <- vapply(slots, function(s) grepl(s, text, fixed = TRUE), logical(1))
  free <- slots[!used]
  if (length(free) == 0) return(list(text = text, mapping = character(0)))

  counts <- vapply(damaged,
                   function(c) lengths(regmatches(text, gregexpr(c, text, fixed = TRUE))),
                   numeric(1))
  # The tie is broken by the code point, never by `order` on the characters
  # themselves. String order in R follows the locale, so the same input would
  # assign different slots on different machines and the ports would disagree.
  code_points <- vapply(damaged, utf8ToInt, integer(1))
  order_by_count <- damaged[order(-counts, code_points)]
  order_by_count <- head(order_by_count, length(free))

  mapping <- setNames(free[seq_along(order_by_count)], order_by_count)
  for (from in names(mapping)) {
    text <- gsub(from, mapping[[from]], text, fixed = TRUE)
  }
  list(text = text, mapping = mapping)
}

#' Apply every rule that makes the text engine independent.
normalize_text <- function(text, spec) {
  flowed <- reflow(text, spec)
  canonicalise(flowed, spec)$text
}
