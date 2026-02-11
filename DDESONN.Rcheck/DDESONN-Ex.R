pkgname <- "DDESONN"
source(file.path(R.home("share"), "R", "examples-header.R"))
options(warn = 1)
options(pager = "console")
base::assign(".ExTimings", "DDESONN-Ex.timings", pos = 'CheckExEnv')
base::cat("name\tuser\tsystem\telapsed\n", file=base::get(".ExTimings", pos = 'CheckExEnv'))
base::assign(".format_ptime",
function(x) {
  if(!is.na(x[4L])) x[1L] <- x[1L] + x[4L]
  if(!is.na(x[5L])) x[2L] <- x[2L] + x[5L]
  options(OutDec = '.')
  format(x[1L:3L], digits = 7L)
},
pos = 'CheckExEnv')

### * </HEADER>
library('DDESONN')

base::assign(".oldSearch", base::search(), pos = 'CheckExEnv')
base::assign(".old_wd", base::getwd(), pos = 'CheckExEnv')
cleanEx()
nameEx("ddesonn_activation_defaults")
### * ddesonn_activation_defaults

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_activation_defaults
### Title: Default activation sequences for DDESONN helpers
### Aliases: ddesonn_activation_defaults

### ** Examples

ddesonn_activation_defaults("binary", hidden_sizes = c(32, 16))
ddesonn_activation_defaults("regression", hidden_sizes = 64, stage = "predict")




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_activation_defaults", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_dropout_defaults")
### * ddesonn_dropout_defaults

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_dropout_defaults
### Title: Default dropout configuration
### Aliases: ddesonn_dropout_defaults

### ** Examples

ddesonn_dropout_defaults(c(64, 32))




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_dropout_defaults", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_fit")
### * ddesonn_fit

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_fit
### Title: Fit a 'ddesonn_model' with tidy inputs
### Aliases: ddesonn_fit

### ** Examples

data <- mtcars
x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
y <- data$am
model <- ddesonn_model(input_size = ncol(x), output_size = 1, hidden_sizes = 8)
ddesonn_fit(model, x, y, num_epochs = 1, lr = 0.05, validation_metrics = FALSE)

# Regression example (mtcars) with explicit scheduler controls.
# If you do NOT want LR decay, set lr_decay_rate = 1.0.
reg_x <- mtcars[, c("disp", "hp", "wt", "qsec", "drat")]
reg_y <- mtcars$mpg
reg_model <- ddesonn_model(
  input_size = ncol(reg_x), # number of input features
  output_size = 1, # one numeric target
  hidden_sizes = c(16, 8), # hidden-layer widths
  classification_mode = "regression" # problem type
)
ddesonn_fit(
  model = reg_model, # model object from ddesonn_model()
  x = reg_x, # training predictors
  y = reg_y, # training target
  num_epochs = 10, # training epochs
  lr = 0.05, # initial learning rate
  lr_decay_rate = 0.5, # decay multiplier (use 1.0 to disable)
  lr_decay_epoch = 20L, # decay step interval in epochs
  lr_min = 1e-5, # lower bound for learning rate
  validation_metrics = FALSE # disable validation metric pass in this example
)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_fit", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_model")
### * ddesonn_model

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_model
### Title: Create a high-level DDESONN model wrapper
### Aliases: ddesonn_model

### ** Examples

model <- ddesonn_model(
  input_size = 5,
  output_size = 1,
  hidden_sizes = c(32, 16),
  classification_mode = "binary"
)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_model", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_optimizer_options")
### * ddesonn_optimizer_options

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_optimizer_options
### Title: Supported optimizer identifiers
### Aliases: ddesonn_optimizer_options

### ** Examples

ddesonn_optimizer_options()




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_optimizer_options", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_predict")
### * ddesonn_predict

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_predict
### Title: Generate predictions from a fitted 'ddesonn_model'
### Aliases: ddesonn_predict

### ** Examples

data <- mtcars
x <- data[, c("disp", "hp", "wt", "qsec", "drat")]
y <- data$am
model <- ddesonn_model(input_size = ncol(x), output_size = 1, hidden_sizes = 8)
ddesonn_fit(model, x, y, num_epochs = 1, lr = 0.05, validation_metrics = FALSE)
preds <- ddesonn_predict(model, x)
head(preds$prediction)




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_predict", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_run")
### * ddesonn_run

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_run
### Title: Run DDESONN across common ensemble scenarios.
### Aliases: ddesonn_run

### ** Examples

## No test: 
# ============================================================
# DDESONN — FULL example using package data in inst/extdata
# (binary classification; train/valid/test split; scale train-only)
# ============================================================

library(DDESONN)

set.seed(111)

# ------------------------------------------------------------
# 1) Locate package extdata folder (robust across check/install)  #$$$$$$$$$$$$$
# ------------------------------------------------------------
ext_dir <- system.file("extdata", package = "DDESONN")
if (!nzchar(ext_dir)) {
  stop("Could not find DDESONN extdata folder. Is the package installed?",
       call. = FALSE)
}

# ------------------------------------------------------------
# 1b) Find CSVs (recursive + check-dir edge cases)               #$$$$$$$$$$$$$
# ------------------------------------------------------------
csvs <- list.files(
  ext_dir,
  pattern = "\\\\.csv$",
  full.names = TRUE,
  recursive = TRUE
)

# Defensive fallback for rare nested layouts
if (!length(csvs)) {                                             #$$$$$$$$$$$$$
  ext_dir2 <- file.path(ext_dir, "inst", "extdata")               #$$$$$$$$$$$$$
  if (dir.exists(ext_dir2)) {                                    #$$$$$$$$$$$$$
    csvs <- list.files(
      ext_dir2,
      pattern = "\\\\.csv$",
      full.names = TRUE,
      recursive = TRUE
    )
  }
}

if (!length(csvs)) {
  message(sprintf(
    "No .csv files found under: %s — skipping example.",
    ext_dir
  ))
} else {

  hf_path <- file.path(ext_dir, "heart_failure_clinical_records.csv")
  data_path <- if (file.exists(hf_path)) hf_path else csvs[[1]]

  cat("[extdata] using:", data_path, "\\n")

# ------------------------------------------------------------
# 2) Load data
# ------------------------------------------------------------
df <- read.csv(data_path)

# Prefer DEATH_EVENT if present; otherwise infer a binary target
target_col <- if ("DEATH_EVENT" %in% names(df)) {
  "DEATH_EVENT"
} else {
  cand <- names(df)[vapply(df, function(col) {
    v <- suppressWarnings(as.numeric(col))
    if (all(is.na(v))) return(FALSE)
    u <- unique(v[is.finite(v)])
    length(u) <= 2 && all(sort(u) %in% c(0, 1))
  }, logical(1))]
  if (!length(cand)) {
    stop(
      "Could not infer a binary target column. ",
      "Provide a binary CSV in extdata or rename target to DEATH_EVENT.",
      call. = FALSE
    )
  }
  cand[[1]]
}

cat("[data] target_col =", target_col, "\\n")

# ------------------------------------------------------------
# 3) Build X and y
# ------------------------------------------------------------
y_all <- matrix(as.integer(df[[target_col]]), ncol = 1)

x_df <- df[, setdiff(names(df), target_col), drop = FALSE]
x_all <- as.matrix(x_df)
storage.mode(x_all) <- "double"

# ------------------------------------------------------------
# 4) Split 70 / 15 / 15
# ------------------------------------------------------------
n <- nrow(x_all)
idx <- sample.int(n)

n_train <- floor(0.70 * n)
n_valid <- floor(0.15 * n)

i_tr <- idx[1:n_train]
i_va <- idx[(n_train + 1):(n_train + n_valid)]
i_te <- idx[(n_train + n_valid + 1):n]

x_train <- x_all[i_tr, , drop = FALSE]
y_train <- y_all[i_tr, , drop = FALSE]

x_valid <- x_all[i_va, , drop = FALSE]
y_valid <- y_all[i_va, , drop = FALSE]

x_test  <- x_all[i_te, , drop = FALSE]
y_test  <- y_all[i_te, , drop = FALSE]

cat(sprintf("[split] train=%d valid=%d test=%d\\n",
            nrow(x_train), nrow(x_valid), nrow(x_test)))

# ------------------------------------------------------------
# 5) Scale using TRAIN stats only (no leakage)
# ------------------------------------------------------------
x_train_s <- scale(x_train)
ctr <- attr(x_train_s, "scaled:center")
scl <- attr(x_train_s, "scaled:scale")
scl[!is.finite(scl) | scl == 0] <- 1

x_valid_s <- sweep(sweep(x_valid, 2, ctr, "-"), 2, scl, "/")
x_test_s  <- sweep(sweep(x_test,  2, ctr, "-"), 2, scl, "/")

mx <- suppressWarnings(max(abs(x_train_s)))
if (!is.finite(mx) || mx == 0) mx <- 1

x_train <- x_train_s / mx
x_valid <- x_valid_s / mx
x_test  <- x_test_s  / mx

# ------------------------------------------------------------
# 6) Run DDESONN
# ------------------------------------------------------------
res <- ddesonn_run(
  x = x_train,
  y = y_train,
  classification_mode = "binary",

  hidden_sizes = c(64, 32),
  seeds = 1L,
  do_ensemble = FALSE,

  validation = list(
    x = x_valid,
    y = y_valid
  ),

  test = list(
    x = x_test,
    y = y_test
  ),

  training_overrides = list(
    init_method = "he",
    optimizer = "adagrad",
    lr = 0.125,
    lambda = 0.00028,

    activation_functions = list(relu, relu, sigmoid),
    dropout_rates = list(0.10),
    loss_type = "CrossEntropy",

    validation_metrics = TRUE,
    num_epochs = 360,
    final_summary_decimals = 6L
  ),

  plot_controls = list(
    evaluate_report = list(
      roc_curve = TRUE,
      pr_curve  = FALSE
    )
  )
)
}
## End(No test)



base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_run", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
cleanEx()
nameEx("ddesonn_training_defaults")
### * ddesonn_training_defaults

flush(stderr()); flush(stdout())

base::assign(".ptime", proc.time(), pos = "CheckExEnv")
### Name: ddesonn_training_defaults
### Title: Construct default training controls
### Aliases: ddesonn_training_defaults

### ** Examples

ddesonn_training_defaults("binary", hidden_sizes = c(32, 16))

# Inspect regression defaults (includes LR decay by default).
cfg_reg <- ddesonn_training_defaults("regression", hidden_sizes = c(16, 8))
cfg_reg$lr
cfg_reg$lr_decay_rate
cfg_reg$lr_decay_epoch
cfg_reg$lr_min

# If you prefer a fixed LR in regression, disable decay explicitly.
cfg_reg$lr_decay_rate <- 1.0




base::assign(".dptime", (proc.time() - get(".ptime", pos = "CheckExEnv")), pos = "CheckExEnv")
base::cat("ddesonn_training_defaults", base::get(".format_ptime", pos = 'CheckExEnv')(get(".dptime", pos = "CheckExEnv")), "\n", file=base::get(".ExTimings", pos = 'CheckExEnv'), append=TRUE, sep="\t")
### * <FOOTER>
###
cleanEx()
options(digits = 7L)
base::cat("Time elapsed: ", proc.time() - base::get("ptime", pos = 'CheckExEnv'),"\n")
grDevices::dev.off()
###
### Local variables: ***
### mode: outline-minor ***
### outline-regexp: "\\(> \\)?### [*]+" ***
### End: ***
quit('no')
