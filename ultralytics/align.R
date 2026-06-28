#!/usr/bin/env Rscript
# import_peaks_to_xcms.R
#
# 将外部峰表（CSV）导入 XCMS 做 alignment / correspondence / fillPeaks
#
# 要求每个 CSV 包含列（至少）： mz, mzmin, mzmax, rt, rtmin, rtmax, into, maxo
# rt 单位必须为 秒（如果你的 CSV 用 分钟，请使用 --rt_in_minutes TRUE 参数）
#
suppressPackageStartupMessages({
  library(optparse)
  library(xcms)
  library(MSnbase)
  library(BiocParallel)
  library(tools)
})

option_list <- list(
  make_option(c("--mzml_dir"), type="character", default=Sys.getenv("PEAK_MZML_DIR", "./data/mzml"),
              help="Directory containing .mzML files (each file = one sample)", metavar="DIR"),
  make_option(c("--csv_dir"), type="character", default=Sys.getenv("PEAK_CSV_DIR", "./data/csv"),
              help="Directory containing peak CSV files (one CSV per sample) OR one CSV with sample column", metavar="DIR"),
  make_option(c("--out_dir"), type="character", default=Sys.getenv("PEAK_OUT_DIR", "./output"),
              help="Output directory", metavar="DIR"),
  make_option(c("--rt_in_minutes"), action="store_true", default=TRUE,
              help="If set, the rt values in CSV are in minutes and will be converted to seconds"),
  make_option(c("--retcor_method"), type="character", default="obiwarp",
              help="RT correction method: obiwarp or peakgroups", metavar="METHOD"),
  make_option(c("--group_mzwid"), type="double", default=0.01,
              help="m/z width (Da) for grouping (mzwid)", metavar="DOUBLE"),
  make_option(c("--group_bw"), type="double", default=10,
              help="RT bandwidth (seconds) for grouping (bw)", metavar="DOUBLE"),
  make_option(c("--group_minfrac"), type="double", default=0.3,
              help="minfrac for groupChromPeaks (fraction of samples) (minfrac)", metavar="DOUBLE"),
  make_option(c("--threads"), type="integer", default=1,
              help="Number of threads for BiocParallel", metavar="INT"),
  make_option(c("--verbose"), action="store_true", default=FALSE,
              help="Verbose output")
)

opt <- parse_args(OptionParser(option_list=option_list))

# ---- basic checks ----
if (is.null(opt$mzml_dir) || is.null(opt$csv_dir)) {
  stop("Please provide both --mzml_dir and --csv_dir")
}
dir_mzml <- normalizePath(opt$mzml_dir)
dir_csv  <- normalizePath(opt$csv_dir)
dir_out  <- normalizePath(opt$out_dir, mustWork = FALSE)
dir.create(dir_out, recursive = TRUE, showWarnings = FALSE)

# list files
mzml_files <- list.files(dir_mzml, pattern="\\.mzML$", full.names=TRUE, ignore.case=TRUE)
csv_files  <- list.files(dir_csv,  pattern="\\.csv$", full.names=TRUE, ignore.case=TRUE)

if (length(mzml_files) == 0) stop("No mzML files found in --mzml_dir")
if (length(csv_files) == 0) stop("No CSV files found in --csv_dir")

message("Found ", length(mzml_files), " mzML files and ", length(csv_files), " CSV files.")

# sort by natural order to keep deterministic mapping
mzml_files <- mzml_files[order(basename(mzml_files))]
csv_files  <- csv_files[order(basename(csv_files))]

# Attempt 3 strategies to obtain per-sample peak tables:
# 1) If there is exactly one CSV and it contains 'sample' column -> use it directly
# 2) If #csv == #mzml -> match by basename (without ext) OR positional mapping
# 3) Otherwise try to match each csv basename to mzml basename (substring). If ambiguous, stop.

make_sample_peak_df_list <- function(mzml_files, csv_files, rt_in_minutes=FALSE) {
  sample_tables <- list()
  
  # 1) single CSV with sample column?
  if (length(csv_files) == 1) {
    df <- read.csv(csv_files[1], stringsAsFactors = FALSE)
    if ("sample" %in% tolower(names(df))) {
      # normalize sample column name (find actual)
      sample_col <- names(df)[tolower(names(df)) == "sample"][1]
      # ensure required columns exist
      required <- c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo")
      missing <- setdiff(required, tolower(names(df)))
      if (length(missing) > 0) stop("Single CSV provided but missing required columns: ", paste(missing, collapse=", "))
      # to keep original names, map lowercase to actual names
      name_map <- setNames(names(df), tolower(names(df)))
      # convert rt/rtmin/rtmax if in minutes
      if (rt_in_minutes) {
        df[[name_map["rt"]]]   <- df[[name_map["rt"]]] * 60
        df[[name_map["rtmin"]]]<- df[[name_map["rtmin"]]] * 60
        df[[name_map["rtmax"]]]<- df[[name_map["rtmax"]]] * 60
      }
      # split by sample value
      samples_vals <- unique(df[[sample_col]])
      for (s in samples_vals) {
        sub <- df[df[[sample_col]] == s, , drop=FALSE]
        # attempt to map sample label s to an mzml file index
        # try basename match
        b <- basename(as.character(s))
        matched_idx <- which(tolower(tools::file_path_sans_ext(basename(mzml_files))) == tolower(b))
        if (length(matched_idx) == 0) {
          # if s is numeric and within range, treat as index
          if (suppressWarnings(!is.na(as.numeric(as.character(s))))) {
            idx <- as.integer(as.character(s))
            if (idx >=1 && idx <= length(mzml_files)) matched_idx <- idx
          }
        }
        if (length(matched_idx) != 1) {
          stop("Cannot unambiguously map sample label '", s, "' to a mzML file. Please ensure sample labels match mzML basenames or provide one CSV per sample.")
        }
        sample_tables[[as.character(matched_idx)]] <- sub
      }
      return(sample_tables)
    }
  }
  
  # 2) equal counts -> try basename match, else positional
  if (length(csv_files) == length(mzml_files)) {
    # build map by basename without ext
    mzml_base <- tolower(tools::file_path_sans_ext(basename(mzml_files)))
    csv_base  <- tolower(tools::file_path_sans_ext(basename(csv_files)))
    map_idx <- rep(NA_integer_, length(csv_files))
    for (i in seq_along(csv_files)) {
      # try exact basename match
      match_pos <- which(mzml_base == csv_base[i])
      if (length(match_pos) == 1) {
        map_idx[i] <- match_pos
      } else if (length(match_pos) > 1) {
        stop("Ambiguous basename match for CSV: ", csv_files[i])
      } else {
        # fallback to positional mapping
        map_idx[i] <- i
      }
    }
    # read csvs and assign to corresponding sample index
    for (i in seq_along(csv_files)) {
      df <- read.csv(csv_files[i], stringsAsFactors = FALSE)
      sample_tables[[ as.character(map_idx[i]) ]] <- df
    }
    return(sample_tables)
  }
  
  # 3) try substring matching: for each csv basename find single mzml basename that contains csv basename or vice versa
  mzml_base <- tolower(tools::file_path_sans_ext(basename(mzml_files)))
  csv_base  <- tolower(tools::file_path_sans_ext(basename(csv_files)))
  mapped <- rep(NA_integer_, length(csv_files))
  for (i in seq_along(csv_files)) {
    candidates <- which(grepl(csv_base[i], mzml_base, fixed=TRUE) | grepl(mzml_base, csv_base[i], fixed=TRUE))
    if (length(candidates) == 1) {
      mapped[i] <- candidates
    } else {
      mapped[i] <- NA
    }
  }
  if (any(is.na(mapped))) {
    stop("Cannot reliably match CSV files to mzML files. Please ensure either:\n  (a) one CSV with sample column, or\n  (b) same number of CSVs and mzMLs with matching basenames, or\n  (c) CSV basenames matching mzML basenames as substrings.")
  }
  for (i in seq_along(csv_files)) {
    df <- read.csv(csv_files[i], stringsAsFactors = FALSE)
    sample_tables[[ as.character(mapped[i]) ]] <- df
  }
  return(sample_tables)
}

# Build sample -> peak df list
sample_peak_list <- make_sample_peak_df_list(mzml_files, csv_files, rt_in_minutes = opt$rt_in_minutes)

# Validate and normalize each sample table
normalize_peak_df <- function(df, sample_index, rt_in_minutes=FALSE) {
  # expected lower-case names
  required <- c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo")
  # map existing columns by lowercase
  names_lower <- tolower(names(df))
  name_map <- setNames(names(df), names_lower)
  missing <- setdiff(required, names_lower)
  if (length(missing) > 0) stop("Sample ", sample_index, " missing required columns: ", paste(missing, collapse=", "))
  # build canonical df
  canonical <- data.frame(
    mz   = as.numeric(df[[ name_map["mz"] ]]),
    mzmin= as.numeric(df[[ name_map["mzmin"] ]]),
    mzmax= as.numeric(df[[ name_map["mzmax"] ]]),
    rt   = as.numeric(df[[ name_map["rt"] ]]),
    rtmin= as.numeric(df[[ name_map["rtmin"] ]]),
    rtmax= as.numeric(df[[ name_map["rtmax"] ]]),
    into = as.numeric(df[[ name_map["into"] ]]),
    maxo = as.numeric(df[[ name_map["maxo"] ]]),
    stringsAsFactors = FALSE
  )
  if (rt_in_minutes) {
    canonical$rt    <- canonical$rt * 60
    canonical$rtmin <- canonical$rtmin * 60
    canonical$rtmax <- canonical$rtmax * 60
  }
  canonical$sample <- as.integer(sample_index)
  # optionally coerce NA -> 0 for into
  canonical$into[is.na(canonical$into)] <- 0
  canonical$maxo[is.na(canonical$maxo)] <- 0
  return(canonical[, c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo","sample")])
}

# assemble all peaks into a single data.frame
all_peaks <- do.call(rbind, lapply(names(sample_peak_list), function(k){
  idx <- as.integer(k)
  df <- sample_peak_list[[k]]
  normalize_peak_df(df, sample_index = idx, rt_in_minutes = opt$rt_in_minutes)
}))

# ensure numeric and no NA in required columns
stopifnot(all(c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo","sample") %in% names(all_peaks)))
all_peaks <- as.data.frame(all_peaks, stringsAsFactors = FALSE)
for (nm in c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo","sample")) {
  if (!is.numeric(all_peaks[[nm]])) all_peaks[[nm]] <- as.numeric(all_peaks[[nm]])
}
if (any(is.na(all_peaks$rt))) stop("Some rt values are NA after normalization. Please check input CSVs.")

# sort rows by sample then rt
all_peaks <- all_peaks[order(all_peaks$sample, all_peaks$rt), ]

# Now build XCMSnExp object
message("Reading mzML files into MSnbase (onDisk) ...")
raw <- readMSData(files = mzml_files, mode = "onDisk")

# create minimal XCMSnExp wrapper if needed
xdata <- as(raw, "XCMSnExp")

# Convert all_peaks to matrix and assign to chromPeaks
# chromPeaks expects numeric matrix where columns correspond to:
# mz, mzmin, mzmax, rt, rtmin, rtmax, into, maxo, sample, (possibly extra)
chrom_mat <- as.matrix(all_peaks[, c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo","sample")])
colnames(chrom_mat) <- c("mz","mzmin","mzmax","rt","rtmin","rtmax","into","maxo","sample")

message("Assigning chromPeaks (num peaks = ", nrow(chrom_mat), ") ...")
chromPeaks(xdata) <- chrom_mat

# optionally free memory
rm(all_peaks, chrom_mat)
gc()

# Set parallel backend
bp <- MulticoreParam(workers = opt$threads)
register(bp)

# RT correction
message("Performing RT correction (method = ", opt$retcor_method, ") ...")
if (tolower(opt$retcor_method) == "obiwarp") {
  xdata <- adjustRtime(xdata, param = ObiwarpParam())
} else if (tolower(opt$retcor_method) == "peakgroups") {
  xdata <- adjustRtime(xdata, param = PeakGroupsParam())
} else {
  stop("Unknown retcor_method: ", opt$retcor_method)
}

# Group / correspondence
message("Grouping peaks across samples (mzwid=", opt$group_mzwid, ", bw=", opt$group_bw, ", minfrac=", opt$group_minfrac, ") ...")
grpParam <- PeakDensityParam(sampleGroups = rep(1, length(mzml_files)), # placeholder; grouping uses numeric sample indices from chromPeaks
                             minFraction = opt$group_minfrac,
                             bw = opt$group_bw,
                             binSize = opt$group_mzwid)
xdata <- groupChromPeaks(xdata, param = grpParam)

# Fill missing peaks
message("Filling missing peaks (fillChromPeaks) ...")
xdata <- fillChromPeaks(xdata)

# Extract feature matrix (using 'into' intensities)
message("Extracting feature x sample matrix ...")
feature_mat <- featureValues(xdata, value = "into")
feature_def <- featureDefinitions(xdata)
feature_def2 <- feature_def[sapply(feature_def, function(x) !is.list(x))]

# save outputs
feat_csv <- file.path(dir_out, "feature_table_into.csv")
def_csv  <- file.path(dir_out, "feature_definitions.csv")
chrom_csv <- file.path(dir_out, "aligned_chromPeaks.csv")

# -------------------------------------------------
# 保存结果
# -------------------------------------------------
message("Saving feature matrix to: ", feat_csv)
feature_mat <- featureValues(xdata, method = "medret", value = "into")  # 生成矩阵
write.csv(feature_mat, feat_csv, row.names = TRUE)

message("Saving feature definitions to: ", def_csv)
feature_def <- featureDefinitions(xdata)
write.csv(as.data.frame(feature_def2), def_csv, row.names = FALSE)

# 保存对齐后的所有chromPeaks表
message("Saving aligned chromPeaks (all peaks) to: ", chrom_csv)
chrom_df <- as.data.frame(chromPeaks(xdata))
if (is.null(colnames(chrom_df))) colnames(chrom_df) <- colnames(chromPeaks(xdata))
write.csv(chrom_df, chrom_csv, row.names = FALSE)

message("✅ All results saved successfully.")
