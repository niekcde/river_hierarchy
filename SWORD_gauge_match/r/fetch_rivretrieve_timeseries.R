#!/usr/bin/env Rscript

default_function_map <- c(
  AU = "australia",
  BR = "brazil",
  CA = "canada",
  CL = "chile",
  FR = "france",
  GB = "uk",
  IE = "ireland",
  JP = "japan",
  UK = "uk",
  US = "usa",
  ZA = "southAfrica"
)

parse_args <- function(args) {
  parsed <- list()
  i <- 1
  while (i <= length(args)) {
    key <- args[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    key <- substring(key, 3)
    if (i == length(args) || startsWith(args[[i + 1]], "--")) {
      parsed[[key]] <- TRUE
      i <- i + 1
    } else {
      parsed[[key]] <- args[[i + 1]]
      i <- i + 2
    }
  }
  parsed
}

parse_function_map <- function(text) {
  if (is.null(text) || !nzchar(text)) {
    return(default_function_map)
  }
  merged <- default_function_map
  parts <- strsplit(text, ",", fixed = TRUE)[[1]]
  for (part in parts) {
    if (!nzchar(part) || !grepl("=", part, fixed = TRUE)) {
      next
    }
    bits <- strsplit(part, "=", fixed = TRUE)[[1]]
    merged[[toupper(trimws(bits[[1]]))]] <- trimws(bits[[2]])
  }
  merged
}

parse_integer_arg <- function(value, default = 0L) {
  if (is.null(value) || !nzchar(value)) {
    return(as.integer(default))
  }
  parsed <- suppressWarnings(as.integer(value))
  if (is.na(parsed)) {
    return(as.integer(default))
  }
  max(0L, parsed)
}

parse_numeric_arg <- function(value, default = 0) {
  if (is.null(value) || !nzchar(value)) {
    return(as.numeric(default))
  }
  parsed <- suppressWarnings(as.numeric(value))
  if (is.na(parsed)) {
    return(as.numeric(default))
  }
  max(0, parsed)
}

build_usgs_daily_url <- function(site, start_date, end_date) {
  if (is.null(start_date) || !nzchar(start_date) || is.null(end_date) || !nzchar(end_date)) {
    stop("USGS direct fallback requires both start_date and end_date", call. = FALSE)
  }
  monitoring_location_id <- sprintf("USGS-%s", as.character(site))
  time_value <- utils::URLencode(sprintf("%s/%s", start_date, end_date), reserved = TRUE)
  sprintf(
    paste0(
      "https://api.waterdata.usgs.gov/ogcapi/v0/collections/daily/items?",
      "f=json&lang=en-US&skipGeometry=TRUE&monitoring_location_id=%s&",
      "parameter_code=00060&statistic_id=00003&time=%s&limit=50000"
    ),
    monitoring_location_id,
    time_value
  )
}

as_scalar_character <- function(value, default = NA_character_) {
  if (is.null(value) || length(value) == 0) {
    return(default)
  }
  as.character(value[[1]])
}

parse_usgs_daily_payload <- function(payload) {
  features <- payload[["features"]]
  if (is.null(features) || length(features) == 0) {
    return(data.frame(
      Date = as.Date(character()),
      Q = numeric(),
      stringsAsFactors = FALSE
    ))
  }

  records <- lapply(features, function(feature) {
    properties <- feature[["properties"]]
    if (is.null(properties)) {
      return(NULL)
    }
    data.frame(
      Date = as.Date(as_scalar_character(properties[["time"]])),
      Q = suppressWarnings(as.numeric(as_scalar_character(properties[["value"]]))),
      stringsAsFactors = FALSE
    )
  })

  records <- Filter(Negate(is.null), records)
  if (length(records) == 0) {
    return(data.frame(
      Date = as.Date(character()),
      Q = numeric(),
      stringsAsFactors = FALSE
    ))
  }

  df <- do.call(rbind, records)
  df <- df[!is.na(df$Date) & !is.na(df$Q), , drop = FALSE]
  df[order(df$Date), , drop = FALSE]
}

bind_rows_fill <- function(tables) {
  if (length(tables) == 0) {
    return(data.frame())
  }
  all_names <- unique(unlist(lapply(tables, names), use.names = FALSE))
  aligned <- lapply(tables, function(tbl) {
    missing <- setdiff(all_names, names(tbl))
    for (name in missing) {
      tbl[[name]] <- NA
    }
    tbl[, all_names, drop = FALSE]
  })
  do.call(rbind, aligned)
}

read_table <- function(path) {
  if (grepl("\\.parquet$", path, ignore.case = TRUE)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("The R package 'arrow' is required to read Parquet input. Use a .csv path or install arrow.", call. = FALSE)
    }
    return(as.data.frame(arrow::read_parquet(path), stringsAsFactors = FALSE))
  }
  utils::read.csv(path, stringsAsFactors = FALSE)
}

write_table <- function(df, output_path) {
  dir.create(dirname(output_path), recursive = TRUE, showWarnings = FALSE)
  if (grepl("\\.parquet$", output_path, ignore.case = TRUE)) {
    if (!requireNamespace("arrow", quietly = TRUE)) {
      stop("The R package 'arrow' is required to write Parquet output. Use a .csv path or install arrow.", call. = FALSE)
    }
    arrow::write_parquet(df, output_path)
    return(invisible(output_path))
  }
  utils::write.csv(df, output_path, row.names = FALSE)
  invisible(output_path)
}

resolve_function_name <- function(row, function_map) {
  if ("country_function" %in% names(row) && !is.na(row[["country_function"]]) && nzchar(row[["country_function"]])) {
    return(as.character(row[["country_function"]]))
  }
  if ("source_function" %in% names(row) && !is.na(row[["source_function"]]) && nzchar(row[["source_function"]])) {
    return(as.character(row[["source_function"]]))
  }
  country_code <- toupper(as.character(row[["country"]]))
  function_map[[country_code]]
}

is_retryable_fetch_error <- function(fetch_status, fetch_error) {
  if (!identical(fetch_status, "error")) {
    return(FALSE)
  }
  if (is.null(fetch_error) || !nzchar(fetch_error)) {
    return(TRUE)
  }
  message_text <- tolower(fetch_error)
  if (grepl("does not have a record associated", message_text, fixed = TRUE)) {
    return(FALSE)
  }
  retry_patterns <- c(
    "argument is of length zero",
    "cannot open url",
    "timeout",
    "timed out",
    "http status was '500",
    "http status was '502",
    "http status was '503",
    "http status was '504",
    "too many requests",
    "http status was '429",
    "rate limit",
    "connection reset",
    "server returned nothing",
    "temporary failure"
  )
  any(vapply(retry_patterns, function(pattern) grepl(pattern, message_text, fixed = TRUE), logical(1)))
}

annotate_fetch_result <- function(result, attempt, retryable) {
  result$fetch_attempts <- as.integer(attempt)
  result$fetch_retryable <- as.logical(retryable)
  result
}

should_try_usgs_fallback <- function(row, function_name, result, variable, start_date, end_date) {
  identical(toupper(as.character(row[["country"]])), "US") &&
    identical(function_name, "usa") &&
    identical(tolower(variable), "discharge") &&
    !is.null(start_date) && nzchar(start_date) &&
    !is.null(end_date) && nzchar(end_date) &&
    !identical(result$fetch_status[[1]], "ok") &&
    !identical(result$fetch_status[[1]], "missing_mapping") &&
    !identical(result$fetch_status[[1]], "missing_function")
}

fetch_usgs_daily_direct <- function(row, variable, start_date, end_date, function_name) {
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    return(build_status_row(
      row = row,
      variable = variable,
      function_name = function_name,
      fetch_status = "error",
      fetch_error = "USGS direct fallback requires the R package 'jsonlite'",
      fetch_source = "usgs_direct",
      fallback_used = TRUE
    ))
  }

  url <- build_usgs_daily_url(row[["station_id"]], start_date, end_date)
  result <- tryCatch(
    {
      payload <- jsonlite::fromJSON(url, simplifyVector = FALSE)
      parse_usgs_daily_payload(payload)
    },
    error = function(err) {
      build_status_row(
        row = row,
        variable = variable,
        function_name = function_name,
        fetch_status = "error",
        fetch_error = conditionMessage(err),
        fetch_source = "usgs_direct",
        fallback_used = TRUE
      )
    }
  )

  if (is.data.frame(result) && "fetch_status" %in% names(result)) {
    return(result)
  }
  finalize_station_result(
    result,
    row,
    variable,
    function_name,
    fetch_source = "usgs_direct",
    fallback_used = TRUE
  )
}

fetch_station_timeseries <- function(
  row,
  variable,
  start_date,
  end_date,
  function_map,
  max_retries = 0L,
  retry_backoff_seconds = 0,
  station_pause_seconds = 0
) {
  function_name <- resolve_function_name(row, function_map)
  if (is.null(function_name) || !nzchar(function_name)) {
    return(build_status_row(
      row = row,
      variable = variable,
      function_name = NA_character_,
      fetch_status = "missing_mapping",
      fetch_error = "No RivRetrieve function mapping for station country"
    ))
  }
  if (!exists(function_name, where = asNamespace("RivRetrieve"), mode = "function")) {
    return(build_status_row(
      row = row,
      variable = variable,
      function_name = function_name,
      fetch_status = "missing_function",
      fetch_error = sprintf("RivRetrieve function '%s' not found", function_name)
    ))
  }

  fun <- get(function_name, envir = asNamespace("RivRetrieve"))
  call_args <- list(site = as.character(row[["station_id"]]), variable = variable)
  if (!is.null(start_date) && nzchar(start_date)) {
    call_args$start_date <- start_date
  }
  if (!is.null(end_date) && nzchar(end_date)) {
    call_args$end_date <- end_date
  }

  total_attempts <- max(1L, as.integer(max_retries) + 1L)
  for (attempt in seq_len(total_attempts)) {
    result <- tryCatch(
      do.call(fun, call_args),
      error = function(err) {
        return(build_status_row(
          row = row,
          variable = variable,
          function_name = function_name,
          fetch_status = "error",
          fetch_error = conditionMessage(err)
        ))
      }
    )

    finalized <- finalize_station_result(result, row, variable, function_name)
    retryable <- is_retryable_fetch_error(finalized$fetch_status[[1]], finalized$fetch_error[[1]])
    finalized <- annotate_fetch_result(finalized, attempt, retryable)

    if (!retryable || attempt >= total_attempts) {
      if (should_try_usgs_fallback(row, function_name, finalized, variable, start_date, end_date)) {
        station_id <- as.character(row[["station_id"]])
        country_code <- as.character(row[["country"]])
        message(sprintf(
          "Falling back to direct USGS daily API for %s:%s after RivRetrieve %s: %s",
          country_code,
          station_id,
          finalized$fetch_status[[1]],
          finalized$fetch_error[[1]]
        ))
        fallback_result <- fetch_usgs_daily_direct(row, variable, start_date, end_date, function_name)
        fallback_result <- annotate_fetch_result(fallback_result, attempt, FALSE)
        if (station_pause_seconds > 0) {
          Sys.sleep(station_pause_seconds)
        }
        return(fallback_result)
      }

      if (station_pause_seconds > 0) {
        Sys.sleep(station_pause_seconds)
      }
      return(finalized)
    }

    wait_seconds <- retry_backoff_seconds * (2 ^ (attempt - 1L))
    station_id <- as.character(row[["station_id"]])
    country_code <- as.character(row[["country"]])
    message(sprintf(
      "Retrying %s:%s attempt %s/%s after transient error: %s",
      country_code,
      station_id,
      attempt + 1L,
      total_attempts,
      finalized$fetch_error[[1]]
    ))
    if (wait_seconds > 0) {
      Sys.sleep(wait_seconds)
    }
  }
}

build_status_row <- function(
  row,
  variable,
  function_name,
  fetch_status,
  fetch_error,
  fetch_source = "rivretrieve",
  fallback_used = FALSE
) {
  data.frame(
    station_id = as.character(row[["station_id"]]),
    country = as.character(row[["country"]]),
    variable = variable,
    source_function = as.character(function_name),
    fetch_status = fetch_status,
    fetch_error = fetch_error,
    fetch_source = fetch_source,
    fallback_used = as.logical(fallback_used),
    fetch_attempts = NA_integer_,
    fetch_retryable = FALSE,
    stringsAsFactors = FALSE
  )
}

finalize_station_result <- function(
  result,
  row,
  variable,
  function_name,
  fetch_source = "rivretrieve",
  fallback_used = FALSE
) {
  if (is.data.frame(result) && "fetch_status" %in% names(result)) {
    return(result)
  }

  df <- as.data.frame(result, stringsAsFactors = FALSE)
  if (nrow(df) == 0) {
    return(build_status_row(
      row = row,
      variable = variable,
      function_name = function_name,
      fetch_status = "empty",
      fetch_error = "No data returned",
      fetch_source = fetch_source,
      fallback_used = fallback_used
    ))
  }

  df$station_id <- as.character(row[["station_id"]])
  df$country <- as.character(row[["country"]])
  df$variable <- variable
  df$source_function <- function_name
  df$fetch_status <- "ok"
  df$fetch_source <- fetch_source
  df$fallback_used <- as.logical(fallback_used)
  if (!("fetch_error" %in% names(df))) {
    df$fetch_error <- NA_character_
  }
  if (!("fetch_attempts" %in% names(df))) {
    df$fetch_attempts <- NA_integer_
  }
  if (!("fetch_retryable" %in% names(df))) {
    df$fetch_retryable <- FALSE
  }
  df
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  options(readr.show_col_types = FALSE)
  if (!requireNamespace("RivRetrieve", quietly = TRUE)) {
    stop("RivRetrieve is not installed. Install it in R before running this script.", call. = FALSE)
  }
  if (is.null(args$input) || is.null(args$output) || is.null(args$variable)) {
    stop("Usage: fetch_rivretrieve_timeseries.R --input gauges_cleaned.parquet --output outputs/gauge_timeseries.parquet --variable discharge", call. = FALSE)
  }

  stations <- read_table(args$input)
  required <- c("station_id", "country")
  missing <- setdiff(required, names(stations))
  if (length(missing) > 0) {
    stop(sprintf("Input station table is missing required columns: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }

  function_map <- parse_function_map(args[["function-map"]])
  max_retries <- parse_integer_arg(args[["max-retries"]], 3L)
  retry_backoff_seconds <- parse_numeric_arg(args[["retry-backoff-seconds"]], 2.0)
  station_pause_seconds <- parse_numeric_arg(args[["station-pause-seconds"]], 0.1)
  country_pause_seconds <- parse_numeric_arg(args[["country-pause-seconds"]], 2.0)
  n_stations <- nrow(stations)
  message(sprintf("Fetching timeseries for %s stations", n_stations))
  progress_breaks <- unique(pmax(1L, ceiling(seq(0.05, 1.00, by = 0.05) * n_stations)))
  next_progress_idx <- 1L
  processed_count <- 0L
  rows <- vector("list", n_stations)

  stations$country <- toupper(as.character(stations$country))
  country_order <- unique(stations$country)
  for (country_idx in seq_along(country_order)) {
    country_code <- country_order[[country_idx]]
    country_rows <- which(stations$country == country_code)
    message(sprintf("Starting country %s (%s stations)", country_code, length(country_rows)))
    flush.console()

    for (idx in country_rows) {
      rows[[idx]] <- fetch_station_timeseries(
        stations[idx, , drop = FALSE],
        args$variable,
        args[["start-date"]],
        args[["end-date"]],
        function_map,
        max_retries = max_retries,
        retry_backoff_seconds = retry_backoff_seconds,
        station_pause_seconds = station_pause_seconds
      )
      processed_count <- processed_count + 1L
      if (next_progress_idx <= length(progress_breaks) && processed_count >= progress_breaks[[next_progress_idx]]) {
        pct <- round(100 * processed_count / n_stations)
        message(sprintf(
          "Progress %s%% (%s/%s stations)",
          pct,
          processed_count,
          n_stations
        ))
        flush.console()
        next_progress_idx <- next_progress_idx + 1L
      }
    }

    if (country_pause_seconds > 0 && country_idx < length(country_order)) {
      message(sprintf("Pausing %.1f seconds before next country", country_pause_seconds))
      flush.console()
      Sys.sleep(country_pause_seconds)
    }
  }
  result <- bind_rows_fill(rows)
  write_table(result, args$output)
  message(sprintf("Wrote %s timeseries rows to %s", nrow(result), args$output))
}

if (sys.nframe() == 0) {
  main()
}
