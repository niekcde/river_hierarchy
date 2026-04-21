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

pick_column <- function(df, candidates) {
  lower_names <- tolower(names(df))
  for (candidate in candidates) {
    idx <- match(tolower(candidate), lower_names)
    if (!is.na(idx)) {
      return(df[[idx]])
    }
  }
  NULL
}

coalesce_column <- function(df, candidates, default = NA) {
  values <- pick_column(df, candidates)
  if (is.null(values)) {
    return(rep(default, nrow(df)))
  }
  values
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

normalize_site_table <- function(tbl, country_code, function_name) {
  df <- as.data.frame(tbl, stringsAsFactors = FALSE)
  if (nrow(df) == 0) {
    return(data.frame(
      station_id = character(),
      station_name = character(),
      lat = numeric(),
      lon = numeric(),
      country = character(),
      agency = character(),
      drainage_area = numeric(),
      river_name = character(),
      source_function = character()
    ))
  }

  normalized <- data.frame(
    station_id = as.character(coalesce_column(df, c("station_id", "site", "site_no", "site_number", "station_number", "code_station"))),
    station_name = as.character(coalesce_column(df, c("station_name", "site_name", "name", "station_nm"))),
    lat = suppressWarnings(as.numeric(coalesce_column(df, c("lat", "latitude", "dec_lat_va", "y")))),
    lon = suppressWarnings(as.numeric(coalesce_column(df, c("lon", "longitude", "dec_long_va", "x")))),
    country = rep(country_code, nrow(df)),
    agency = as.character(coalesce_column(df, c("agency", "provider", "network", "source_agency"), function_name)),
    drainage_area = suppressWarnings(as.numeric(coalesce_column(
      df,
      c("drainage_area", "drainagearea", "catchment_area", "drainage_area_km2", "basin_area", "area")
    ))),
    river_name = as.character(coalesce_column(df, c("river_name", "stream_name", "river", "watercourse_name"))),
    source_function = rep(function_name, nrow(df)),
    stringsAsFactors = FALSE
  )

  extra_names <- setdiff(names(df), names(normalized))
  cbind(normalized, df[, extra_names, drop = FALSE], stringsAsFactors = FALSE)
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

fetch_country_sites <- function(country_code, function_map) {
  function_name <- function_map[[country_code]]
  if (is.null(function_name) || !nzchar(function_name)) {
    stop(sprintf("No RivRetrieve function mapping configured for country %s", country_code), call. = FALSE)
  }
  if (!exists(function_name, where = asNamespace("RivRetrieve"), mode = "function")) {
    stop(sprintf("RivRetrieve does not export a function named '%s' for country %s", function_name, country_code), call. = FALSE)
  }
  fun <- get(function_name, envir = asNamespace("RivRetrieve"))
  raw <- tryCatch(
    fun(sites = TRUE),
    error = function(err) {
      stop(sprintf("Failed to fetch sites for %s via %s: %s", country_code, function_name, conditionMessage(err)), call. = FALSE)
    }
  )
  normalize_site_table(raw, country_code, function_name)
}

main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (!requireNamespace("RivRetrieve", quietly = TRUE)) {
    stop("RivRetrieve is not installed. Install it in R before running this script.", call. = FALSE)
  }
  if (is.null(args$countries) || is.null(args$output)) {
    stop("Usage: fetch_rivretrieve_metadata.R --countries US,FR --output outputs/gauges_raw.parquet", call. = FALSE)
  }

  countries <- toupper(trimws(strsplit(args$countries, ",", fixed = TRUE)[[1]]))
  countries <- countries[nzchar(countries)]
  function_map <- parse_function_map(args[["function-map"]])

  tables <- lapply(countries, function(country_code) fetch_country_sites(country_code, function_map))
  result <- bind_rows_fill(tables)
  write_table(result, args$output)
  message(sprintf("Wrote %s station rows to %s", nrow(result), args$output))
}

main()

