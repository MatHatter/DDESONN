# ===============================================================
# DeepDynamic - DDESONN
# Deep Dynamic Experimental Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# Licensed for academic and personal research use only.
# Commercial use, redistribution, or incorporation into any
# profit-seeking product or service is strictly prohibited.
#
# This license applies to all versions of DeepDynamic/DDESONN,
# past, present, and future, including legacy releases.
#
# Intended future distribution: CRAN package.
# ===============================================================

# ================================================================
# evaluate_predictions_report.R  (FULL - fixed)
# - PRESERVED: your original binary + regression + multiclass logic, plots, Excel sheet layout,
#              legacy plot scanning, library metric calls, combine_for_report behavior
# -  FIX: ALL verbose output is now gated under if (isTRUE(verbose))
# -  FIX: ALL plot writes (png/ggsave/dev.off) are gated under per-plot toggles (NO enable_plots)  #$$$$$$$$$$$$$
# -  FIX: Paths now use ddesonn_artifacts_root() + ddesonn_plots_dir()
# -  FIX: Excel writing is optional (export_excel). No openxlsx hard dependency unless enabled
# -  ADD: Optional RDS save (save_rds) to reports dir as Rdata_predictions.rds
# -  ADD: Returns legacy scalar list PLUS an attached structured report object (return$report)
# -  #$$$$$$$$$$$$$ FIX: accuracy_plot remains logical gate; accuracy_plot_mode controls which accuracy plots to emit
# -  #$$$$$$$$$$$$$ ADD: viewAllPlots + plot_roc + plot_pr toggles (remove enable_plots)
# -  #$$$$$$$$$$$$$ FIX: remove magrittr %>% usage (no pipe dependency)
#
# -  #$$$$$$$$$$$$$ STYLE ONLY: global theme + canonical eval colors + title normalization (CRAN-safe)
# ================================================================

# ================================================================
# SECTION: Safe %||% fallback
# ================================================================
`%||%` <- get0("%||%", ifnotfound = function(a, b) {
  if (!is.null(a) && length(a) && !all(is.na(a))) a else b
})

# ================================================================
# SECTION: Global eval plot style (CRAN-safe)                      #$$$$$$$$$$$$$
# ================================================================
.ddesonn_plot_theme <- function() {  # #$$$$$$$$$$$$$
  ggplot2::theme_minimal(base_size = 13) +
    ggplot2::theme(
      plot.title = ggplot2::element_text(
        hjust = 0.5,
        face = "bold",
        size = 14
      ),
      axis.title = ggplot2::element_text(size = 12),
      axis.text  = ggplot2::element_text(size = 11),
      legend.title = ggplot2::element_text(size = 11),
      legend.text  = ggplot2::element_text(size = 10)
    )
}

COL_NAVY <- "#2C3E50"  # #$$$$$$$$$$$$$

# ================================================================
# SECTION: EvaluatePredictionsReport
# ================================================================

EvaluatePredictionsReport <- function(
    X_validation, y_validation, CLASSIFICATION_MODE,
    probs,                       # last-epoch fallback (matrix or vector)
    predicted_outputAndTime,     # metadata list from training (optional)
    threshold_function,          # kept for signature compatibility (not used)
    all_best_val_probs,          # best snapshot probs (optional)
    all_best_val_labels,         # best snapshot labels (optional)
    verbose = FALSE,
    # Plot selection ONLY (results always include both fixed and tuned):
    accuracy_plot = TRUE,
    accuracy_plot_mode = c("accuracy", "accuracy_tuned", "both"),
    tuned_threshold_override = NULL,
    show_auprc = TRUE,
    SONN,
    # Optional extras for library metric calls; they are safely ignored if missing:
    Rdata = NULL,
    labels = NULL,
    lr = NULL,
    num_epochs = NULL,
    model_iter_num = NULL,
    ensemble_number = NULL,
    weights = NULL,
    biases = NULL,
    activation_functions = NULL,
    dropout_rates = NULL,
    threshold = 0.5,
    cluster_assignments = NULL,
    run_id = NULL,
    grid = NULL,
    learn_time = NULL,
    output_root = NULL,
    # ================================================================
    # SECTION: Output controls (new, opt-in only)
    # ================================================================
    viewAllPlots = FALSE,   # #$$$$$$$$$$$$$
    plot_roc = TRUE,        # #$$$$$$$$$$$$$
    plot_pr  = TRUE,        # #$$$$$$$$$$$$$
    saveEnabled = TRUE,     # #$$$$$$$$$$$$$
    export_excel = FALSE,
    save_rds = FALSE,
    rds_name = "Rdata_predictions.rds"
) {
  
  # ================================================================
  # SECTION: Package guard (conditional / CRAN-safe)
  # ================================================================
  # Base evaluation has no hard deps. Dependencies are required only when you opt-in:
  # - any plots -> ggplot2/pROC/PRROC/reshape2 (+ dplyr/tidyr used in existing plot code)
  # - export_excel -> openxlsx
  required_pkgs <- character(0)
  
  do_any_plots <- isTRUE(viewAllPlots) || isTRUE(accuracy_plot) || isTRUE(plot_roc) || isTRUE(plot_pr)  # #$$$$$$$$$$$$$
  
  if (isTRUE(do_any_plots)) {  # #$$$$$$$$$$$$$
    required_pkgs <- unique(c(required_pkgs, "ggplot2", "dplyr", "tidyr", "pROC", "PRROC", "reshape2"))
  }
  if (isTRUE(export_excel)) {
    required_pkgs <- unique(c(required_pkgs, "openxlsx"))
  }
  missing_pkgs <- required_pkgs[!vapply(required_pkgs, requireNamespace, logical(1), quietly = TRUE)]
  if (length(missing_pkgs)) {
    stop(
      sprintf("The following packages are required but not installed: %s", paste(missing_pkgs, collapse = ", ")),
      call. = FALSE
    )
  }
  
  # ================================================================
  # SECTION: Arg normalization (PRESERVED + FIX)
  # ================================================================
  # #$$$$$$$$$$$$$ FIX: accuracy_plot is a logical gate; do NOT match.arg() it
  accuracy_plot_mode <- match.arg(accuracy_plot_mode)  # #$$$$$$$$$$$$$
  if (isTRUE(verbose)) cat("[Eval] Begin EvaluatePredictionsReport()\n")
  
  # ================================================================
  # SECTION: Setup paths (artifacts-root + plots helper)
  # ================================================================
  # PRESERVE: you still get a reports dir and a plot dir, but now both live under artifacts/.
  artifacts_root <- ddesonn_artifacts_root(output_root)
  reports_dir    <- file.path(artifacts_root, "reports")
  plot_dir       <- file.path(ddesonn_plots_dir(output_root),
                              "EvaluatePredictionsReportPlots")
  
  dir.create(reports_dir, recursive = TRUE, showWarnings = FALSE)
  dir.create(plot_dir,    recursive = TRUE, showWarnings = FALSE)
  
  report_wb_path <- file.path(reports_dir, "Rdata_predictions.xlsx")
  report_rds_path <- file.path(reports_dir, rds_name)
  
  # ===== Legacy plot candidates (READ-ONLY fallback) =====
  legacy_plot_dir <- file.path(getwd(), "EvaluatePredictionsReportPlots")
  plot_candidates <- c(plot_dir, legacy_plot_dir, getwd())
  plot_candidates <- plot_candidates[!is.na(plot_candidates) & nzchar(plot_candidates)]
  plot_candidates <- unique(plot_candidates)
  
  if (isTRUE(verbose)) {
    cat("[Eval] artifacts_root:", artifacts_root, "\n")
    cat("[Eval] reports_dir:", reports_dir, "\n")
    cat("[Eval] plot_dir:", plot_dir, "\n")
    if (dir.exists(legacy_plot_dir)) cat("[Eval] legacy_plot_dir detected (read-only):", legacy_plot_dir, "\n")
    cat("[Eval] viewAllPlots:", viewAllPlots, " plot_roc:", plot_roc, " plot_pr:", plot_pr, "\n")  # #$$$$$$$$$$$$$
    cat("[Eval] export_excel:", export_excel, " save_rds:", save_rds, "\n")
    cat("[Eval] accuracy_plot:", accuracy_plot, " accuracy_plot_mode:", accuracy_plot_mode, "\n")  # #$$$$$$$$$$$$$
  }
  
  find_plot_file <- function(filename) {
    for (d in plot_candidates) {
      p <- file.path(d, filename)
      if (file.exists(p)) return(p)
    }
    NULL
  }
  
  # ================================================================
  # SECTION: Safety defaults (official) (PRESERVED)
  # ================================================================
  if (!exists("viewTables", inherits = TRUE)) viewTables <- FALSE
  if (!exists("ML_NN", inherits = TRUE))      ML_NN      <- FALSE
  if (!exists("train", inherits = TRUE))      train      <- FALSE
  if (isTRUE(verbose)) {
    cat("[Eval] flags -> viewTables:", viewTables, "  ML_NN:", ML_NN, "  train:", train, "\n")
  }
  
  # ================================================================
  # SECTION: Inspect predictions/errors (optional) (PRESERVED)
  # ================================================================
  pred_vec   <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$learn_output),
                         error = function(e) rep(NA_real_, length.out = 0))
  err_vec    <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$error),
                         error = function(e) rep(NA_real_, length.out = 0))
  labels_vec <- tryCatch(as.vector(y_validation), error = function(e) rep(NA_real_, length.out = 0))
  max_points <- min(length(pred_vec), length(err_vec), length(labels_vec))
  
  if (isTRUE(verbose)) {
    cat("[Eval] pred/err/labels lengths:", length(pred_vec), length(err_vec), length(labels_vec),
        "  max_points:", max_points, "\n")
  }
  
  # PLOT GATE (no files unless you toggled any plots)
  if (isTRUE(do_any_plots) && max_points > 0) {  # #$$$$$$$$$$$$$
    if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
      tryCatch({
        grDevices::png(file.path(plot_dir, "pred_vs_error_scatter.png"), width = 800, height = 600)
        plot(pred_vec[seq_len(max_points)], err_vec[seq_len(max_points)],
             main = "Prediction vs. Error", xlab = "Prediction", ylab = "Error",
             col = "steelblue", pch = 16, adj = 0.5)  #$$$$$$$$$$$$$
        abline(h = 0, col = "gray", lty = 2)
        grDevices::dev.off()
        if (isTRUE(verbose)) cat("[Eval] pred_vs_error_scatter saved.\n")
      }, error = function(e) {
        if (isTRUE(verbose)) message("[Eval] Pred-vs-Error plot failed: ", conditionMessage(e))
      })
    }  # #$$$$$$$$$$$$$
  }
  
  # ================================================================
  # SECTION: weights summary (robust) (PRESERVED)
  # ================================================================
  weights_summary <- NULL
  if (isTRUE(ML_NN)) {
    w_mat <- tryCatch(as.matrix(predicted_outputAndTime$weights_record[[1]]),
                      error = function(e) matrix(NA_real_, nrow = 0, ncol = 0))
    if (length(w_mat)) {
      weights_summary <- round(rowMeans(w_mat), 5)
      if (isTRUE(verbose)) cat(">> Multi-layer weights summary (first layer) - rows:", nrow(w_mat), "cols:", ncol(w_mat), "\n")
    }
  } else {
    w_raw <- tryCatch(predicted_outputAndTime$weights_record[[1]], error = function(e) numeric(0))
    if (length(w_raw)) {
      weights_summary <- round(as.numeric(w_raw), 5)
      if (isTRUE(verbose)) cat(">> Single-layer weights summary len:", length(weights_summary), "\n")
    }
  }
  
  # ================================================================
  # SECTION: Select evaluation data (PRESERVED)
  # ================================================================
  use_best <- (!is.null(all_best_val_probs) && !is.null(all_best_val_labels))
  if (use_best) {
    probs_use  <- all_best_val_probs
    labels_use <- all_best_val_labels
    if (isTRUE(verbose)) cat("[Eval] Using BEST snapshot from training (probs/labels).\n")
  } else {
    probs_use  <- probs
    labels_use <- y_validation
    if (isTRUE(verbose)) cat("[Eval] Using LAST-epoch predictions.\n")
  }
  
  to_mat <- function(x) {
    if (is.list(x) && !is.null(x$predicted_output)) x <- x$predicted_output
    if (is.data.frame(x)) x <- as.matrix(x)
    if (!is.matrix(x))    x <- matrix(x, ncol = 1L)
    storage.mode(x) <- "double"
    x
  }
  L <- to_mat(labels_use)
  P <- to_mat(probs_use)
  n_eff <- min(nrow(L), nrow(P))
  
  if (isTRUE(verbose)) {
    cat("[Eval] Shapes L:", nrow(L), "x", ncol(L),
        "  P:", nrow(P), "x", ncol(P),
        "  n_eff:", n_eff, "\n")
  }
  if (n_eff <= 0) stop("[EvaluatePredictionsReport] No overlapping rows between probs and labels.")
  L <- L[seq_len(n_eff), , drop = FALSE]
  P <- P[seq_len(n_eff), , drop = FALSE]
  
  # ================================================================
  # SECTION: Mode inference (PRESERVED)
  # ================================================================
  infer_mode <- function(L, P, fallback = "binary") {
    if (tolower(CLASSIFICATION_MODE) %in% c("binary","multiclass","regression")) return(tolower(CLASSIFICATION_MODE))
    if (max(ncol(L), ncol(P)) > 1L) "multiclass" else fallback
  }
  mode <- infer_mode(L, P, "binary")
  if (isTRUE(verbose)) cat(sprintf("[Eval] mode=%s | n_eff=%d | ncol(L)=%d | ncol(P)=%d\n", mode, n_eff, ncol(L), ncol(P)))
  
  # ================================================================
  # SECTION: Helpers for plot writing (GATED)
  # ================================================================
  .close_devices <- function() {
    if (length(dev.list())) invisible(lapply(dev.list(), function(x) try(dev.off(), silent = TRUE)))
  }
  
  # ================================================================
  # SECTION: Regression branch (PRESERVED; Excel gated)
  # ================================================================
  if (identical(mode, "regression")) {
    if (isTRUE(verbose)) cat("[Eval-Regression] Enter\n")
    
    y    <- suppressWarnings(as.numeric(L[,1]))
    yhat <- suppressWarnings(as.numeric(P[,1]))
    keep <- is.finite(y) & is.finite(yhat)
    y    <- y[keep]; yhat <- yhat[keep]
    if (!length(y)) stop("Regression mode: no finite overlapping y / yhat.")
    
    residuals <- yhat - y
    SSE  <- sum(residuals^2)
    SST  <- sum((y - mean(y))^2)
    RMSE <- sqrt(mean(residuals^2))
    MAE  <- mean(abs(residuals))
    MAPE <- if (any(y != 0)) mean(abs(residuals / y)) else NA_real_
    R2   <- if (SST > 0) 1 - SSE/SST else NA_real_
    Corr <- suppressWarnings(stats::cor(y, yhat))
    
    if (isTRUE(verbose)) cat("[Eval-Regression] RMSE:", RMSE, "  MAE:", MAE, "  R2:", R2, "  Corr:", Corr, "\n")
    
    legacy_df <- data.frame(y_true = y, y_pred = yhat, residual = yhat - y)
    
    report <- list(
      mode = "regression",
      paths = list(reports_dir = reports_dir, plot_dir = plot_dir, xlsx = report_wb_path, rds = report_rds_path),
      tables = list(Rdata_Predictions = legacy_df),
      metrics = list(RMSE = RMSE, MAE = MAE, MAPE = MAPE, R2 = R2, Correlation = Corr),
      plots = list(),
      artifacts = list()
    )
    
    # Optional Excel export
    if (isTRUE(export_excel)) {
      wb <- openxlsx::createWorkbook()
      openxlsx::addWorksheet(wb, "Metrics_Summary")
      suppressWarnings(openxlsx::writeData(wb, "Metrics_Summary",
                                           data.frame(Metric=c("RMSE","MAE","MAPE","R2","Correlation"), Value=c(RMSE,MAE,MAPE,R2,Corr))
      ))
      openxlsx::addWorksheet(wb, "Rdata_Predictions")
      suppressWarnings(openxlsx::writeData(wb, "Rdata_Predictions", legacy_df))
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch(openxlsx::saveWorkbook(wb, report_wb_path, overwrite = TRUE),
                 error = function(e) if (isTRUE(verbose)) message("[Eval-Regression] Workbook save failed: ", conditionMessage(e)))
        if (isTRUE(verbose)) cat("[Eval-Regression] Workbook saved.\n")
      }  # #$$$$$$$$$$$$$
    }
    
    # Optional RDS save
    if (isTRUE(save_rds)) {
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch(saveRDS(report, report_rds_path),
                 error = function(e) if (isTRUE(verbose)) message("[Eval-Regression] saveRDS failed: ", conditionMessage(e)))
        if (isTRUE(verbose)) cat("[Eval-Regression] RDS saved:", report_rds_path, "\n")
      }  # #$$$$$$$$$$$$$
    }
    
    return(list(
      best_threshold  = NA_real_,
      accuracy        = NA_real_,
      precision       = NA_real_,
      recall          = NA_real_,
      f1_score        = NA_real_,
      accuracy_tuned  = NA_real_,
      precision_tuned = NA_real_,
      recall_tuned    = NA_real_,
      f1_tuned        = NA_real_,
      confusion_matrix = NULL,
      y_pred_class     = NULL,
      y_pred_class_tuned = NULL,
      auc = NA_real_,
      roc_curve = NULL,
      report = report
    ))
  }
  
  # ================================================================
  # SECTION: Binary branch (PRESERVED)
  # ================================================================
  if (identical(mode, "binary")) {
    if (isTRUE(verbose)) cat("[Eval-Binary] Enter\n")
    
    y_true <- if (ncol(L) == 1L) {
      v <- as.numeric(L[,1])
      if (all(v %in% c(0,1), na.rm = TRUE)) as.integer(v) else as.integer(v >= 0.5)
    } else {
      as.integer(max.col(L, ties.method = "first") - 1L)
    }
    if (length(y_true) != n_eff) stop("[Eval-Binary] y_true length mismatch.")
    if (ncol(P) != 1L) stop("[Eval-Binary] Expected 1-column probabilities/logits; got ", ncol(P))
    p_pos <- as.numeric(P[,1])
    
    if (isTRUE(verbose)) {
      cat("[Eval-Binary] y_true len:", length(y_true), "  p_pos len:", length(p_pos), "\n")
      cat("[Eval-Binary] y_true table:\n"); print(table(y_true, useNA="ifany"))
      cat("[Eval-Binary] p_pos summary: min=", suppressWarnings(min(p_pos, na.rm=TRUE)),
          " max=", suppressWarnings(max(p_pos, na.rm=TRUE)),
          " mean=", suppressWarnings(mean(p_pos, na.rm=TRUE)),
          " NA_count=", sum(!is.finite(p_pos)), "\n", sep="")
    }
    
    if (any(p_pos < 0 | p_pos > 1, na.rm = TRUE)) {
      if (isTRUE(verbose)) cat("[Eval-Binary][Fixed] Detected logits; applying sigmoid to get probabilities.\n")
      p_pos <- 1 / (1 + exp(-p_pos))
      if (isTRUE(verbose)) {
        cat("[Eval-Binary][After Sigmoid] p_pos summary: min=", suppressWarnings(min(p_pos, na.rm=TRUE)),
            " max=", suppressWarnings(max(p_pos, na.rm=TRUE)),
            " mean=", suppressWarnings(mean(p_pos, na.rm=TRUE)), "\n", sep = "")
      }
    }
    
    if (isTRUE(verbose)) cat("[Eval-Binary][Fixed] Computing metrics @ 0.5\n")
    thr_fixed <- 0.5
    y_pred_fixed <- as.integer(p_pos >= thr_fixed)
    TP <- sum(y_pred_fixed == 1L & y_true == 1L, na.rm = TRUE)
    TN <- sum(y_pred_fixed == 0L & y_true == 0L, na.rm = TRUE)
    FP <- sum(y_pred_fixed == 1L & y_true == 0L, na.rm = TRUE)
    FN <- sum(y_pred_fixed == 0L & y_true == 1L, na.rm = TRUE)
    n_valid <- sum(is.finite(y_true) & is.finite(p_pos))
    acc_fixed <- if (n_valid > 0) (TP + TN) / n_valid else NA_real_
    pre_fixed <- if ((TP + FP) > 0) TP / (TP + FP) else 0
    rec_fixed <- if ((TP + FN) > 0) TP / (TP + FN) else 0
    f1_fixed  <- if ((pre_fixed + rec_fixed) > 0) 2 * pre_fixed * rec_fixed / (pre_fixed + rec_fixed) else 0
    if (isTRUE(verbose)) cat("[Eval-Binary][Fixed] TP:",TP," FP:",FP," TN:",TN," FN:",FN,"  acc:",acc_fixed,"  f1:",f1_fixed,"\n")
    
    # ROC/AUC (object always computed; PNG only if plot_roc/viewAllPlots)  # #$$$$$$$$$$$$$
    if (isTRUE(verbose)) cat("[Eval-Binary][ROC] Computing ROC/AUC\n")
    roc_obj <- tryCatch(
      pROC::roc(response = y_true, predictor = p_pos, levels = c(0,1), direction = "<", quiet = TRUE),
      error = function(e) NULL
    )
    auc_val <- tryCatch(as.numeric(pROC::auc(roc_obj)), error = function(e) NA_real_)
    roc_df  <- if (!is.null(roc_obj)) {
      data.frame(fpr = 1 - roc_obj$specificities,
                 tpr = roc_obj$sensitivities,
                 threshold = roc_obj$thresholds)
    } else NULL
    if (isTRUE(verbose)) cat("[Eval-Binary][ROC] AUC:", ifelse(is.na(auc_val),"NA",sprintf("%.6f",auc_val)),"\n")
    
    roc_png <- file.path(plot_dir, "roc_curve.png")
    if ((isTRUE(viewAllPlots) || isTRUE(plot_roc)) && !is.null(roc_df) && nrow(roc_df) > 1) {  # #$$$$$$$$$$$$$
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch({
          p_roc <- ggplot2::ggplot(roc_df, ggplot2::aes(x = fpr, y = tpr)) +
            ggplot2::geom_line(linewidth = 1.1, color = COL_NAVY) +  # #$$$$$$$$$$$$$
            ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
            ggplot2::labs(  # #$$$$$$$$$$$$$
              title = "ROC Curve",
              x = "False Positive Rate",
              y = "True Positive Rate"
            ) +
            ggplot2::annotate(  # #$$$$$$$$$$$$$
              "text",
              x = 0.75,
              y = 0.1,
              label = sprintf("AUC = %.4f", auc_val),
              size = 4
            ) +
            .ddesonn_plot_theme()  # #$$$$$$$$$$$$$
          ggplot2::ggsave(filename = roc_png, p_roc, width = 6, height = 4, dpi = 300)
          .close_devices()
          if (isTRUE(verbose)) cat("[Eval-Binary][ROC] ROC PNG saved:", roc_png, "\n")
        }, error = function(e) {
          if (isTRUE(verbose)) message("[Eval-Binary][ROC] ggsave failed: ", conditionMessage(e))
        })
      }  # #$$$$$$$$$$$$$
    }
    
    # Threshold tuning (PRESERVED)
    tuned <- NULL
    if (is.numeric(tuned_threshold_override) && is.finite(tuned_threshold_override)) {
      best_thr <- as.numeric(tuned_threshold_override)
      if (isTRUE(verbose)) message(sprintf("[Eval-Binary] Forced tuned_threshold_override=%.4f", best_thr))
      
      if (isTRUE(verbose)) {
        cat("[Eval-Binary][Override] START\n")
        cat("[Eval-Binary][Override] lengths -> y_true:", length(y_true), " p_pos:", length(p_pos), "\n")
        cat("[Eval-Binary][Override] y_true table:\n"); print(table(y_true, useNA="ifany"))
        cat("[Eval-Binary][Override] p_pos head:\n"); print(head(p_pos, 10))
        cat("[Eval-Binary][Override] p_pos tail:\n"); print(tail(p_pos, 10))
        cat("[Eval-Binary][Override] NA checks -> y_true:", sum(!is.finite(y_true)),
            " p_pos:", sum(!is.finite(p_pos)), "\n")
        cat("[Eval-Binary][Override] threshold:", best_thr, "\n")
      }
      
      y_pred_tuned <- as.integer(p_pos >= best_thr)
      TPt <- sum(y_pred_tuned == 1 & y_true == 1, na.rm = TRUE)
      TNt <- sum(y_pred_tuned == 0 & y_true == 0, na.rm = TRUE)
      FPt <- sum(y_pred_tuned == 1 & y_true == 0, na.rm = TRUE)
      FNt <- sum(y_pred_tuned == 0 & y_true == 1, na.rm = TRUE)
      n_valid_t <- sum(is.finite(y_true) & is.finite(p_pos))
      acc_tuned <- if (n_valid_t > 0) (TPt + TNt) / n_valid_t else NA_real_
      pre_tuned <- if ((TPt + FPt) > 0) TPt / (TPt + FPt) else 0
      rec_tuned <- if ((TPt + FNt) > 0) TPt / (TPt + FNt) else 0
      f1_tuned  <- if ((pre_tuned + rec_tuned) > 0) 2 * pre_tuned * rec_tuned / (pre_tuned + rec_tuned) else 0
      
      tuned <- list(
        accuracy = acc_tuned, precision = pre_tuned, recall = rec_tuned, f1 = f1_tuned,
        details  = list(best_threshold = best_thr, y_pred_class = y_pred_tuned)
      )
      
    } else {
      if (isTRUE(verbose)) cat("[Eval-Binary][Tune] Grid sweep begin\n")
      thr_grid <- seq(0.05, 0.95, by = 0.01)
      keep     <- is.finite(y_true) & is.finite(p_pos)
      yy       <- as.integer(y_true[keep])
      pp       <- as.numeric(p_pos[keep])
      n_y      <- length(yy)
      if (n_y == 0L) stop("[Eval-Binary] No finite data for tuning.")
      pos_idx <- (yy == 1L); neg_idx <- !pos_idx
      best_i  <- 1L; best_acc <- -Inf
      
      for (i in seq_along(thr_grid)) {
        thr    <- thr_grid[i]
        ypi    <- as.integer(pp >= thr)
        TPi    <- sum(ypi[pos_idx] == 1L)
        TNi    <- sum(ypi[neg_idx] == 0L)
        acci   <- (TPi + TNi) / n_y
        if (acci > best_acc) { best_acc <- acci; best_i <- i }
      }
      best_thr <- thr_grid[best_i]
      y_best   <- as.integer(pp >= best_thr)
      TPb      <- sum(y_best[pos_idx] == 1L)
      TNb      <- sum(y_best[neg_idx] == 0L)
      FPb      <- sum(y_best[neg_idx] == 1L)
      FNb      <- sum(y_best[pos_idx] == 0L)
      pre_b    <- if ((TPb + FPb) > 0) TPb / (TPb + FPb) else 0
      rec_b    <- if ((TPb + FNb) > 0) TPb / (TPb + FNb) else 0
      f1_b     <- if ((pre_b + rec_b) > 0) 2 * pre_b * rec_b / (pre_b + rec_b) else 0
      y_pred_tuned_full <- integer(length(p_pos)); y_pred_tuned_full[] <- as.integer(NA)
      y_pred_tuned_full[keep] <- as.integer(pp >= best_thr)
      
      tuned <- list(
        accuracy = best_acc, precision = pre_b, recall = rec_b, f1 = f1_b,
        details = list(best_threshold = best_thr, y_pred_class = y_pred_tuned_full)
      )
      if (isTRUE(verbose)) cat("[Eval-Binary][Tune] Grid sweep done. Best thr:", best_thr, "  acc:", best_acc, "\n")
    }
    
    acc_tuned    <- tuned$accuracy
    pre_tuned    <- tuned$precision
    rec_tuned    <- tuned$recall
    f1_tuned     <- tuned$f1
    best_thr     <- as.numeric(tuned$details$best_threshold)
    y_pred_tuned <- as.integer(tuned$details$y_pred_class)
    if (isTRUE(verbose)) cat("[Eval-Binary][Tuned] best_thr:", best_thr, "  acc:", acc_tuned, "  f1:", f1_tuned, "\n")
    
    # Plot bundle helper (PRESERVED; FILE WRITES GATED)  # #$$$$$$$$$$$$$
    maybe_plot_binary <- function(mode_label, bin_preds, threshold_used, suffix) {
      if (!(isTRUE(viewAllPlots) || isTRUE(accuracy_plot))) return(invisible(NULL))  # #$$$$$$$$$$$$$
      if (isTRUE(verbose)) cat("[Eval-Binary][Plot] start:", mode_label, "  thr:", threshold_used, "  suffix:", suffix, "\n")
      
      TPp <- sum(bin_preds == 1 & y_true == 1, na.rm = TRUE)
      TNp <- sum(bin_preds == 0 & y_true == 0, na.rm = TRUE)
      FPp <- sum(bin_preds == 1 & y_true == 0, na.rm = TRUE)
      FNp <- sum(bin_preds == 0 & y_true == 1, na.rm = TRUE)
      conf_matrix_df <- data.frame(
        Actual    = c("0","0","1","1"),
        Predicted = c("0","1","0","1"),
        Count     = c(TNp, FPp, FNp, TPp)
      )
      
      heatmap_path <- file.path(plot_dir, paste0("confusion_matrix_heatmap", suffix, ".png"))
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch({
          p_conf <- ggplot2::ggplot(conf_matrix_df, ggplot2::aes(x = Predicted, y = Actual, fill = Count)) +
            ggplot2::geom_tile(color = "white") +
            ggplot2::geom_text(ggplot2::aes(label = Count), size = 5, fontface = "bold") +  # #$$$$$$$$$$$$$
            ggplot2::scale_fill_gradient(low = "white", high = "#D73027") +  # #$$$$$$$$$$$$$
            ggplot2::labs(
              title = sprintf(
                "%s - Threshold = %.4f",
                ifelse(grepl("tuned", tolower(mode_label)), "Tuned Accuracy", "Accuracy"),  # #$$$$$$$$$$$$$                                                             # #$$$$$$$$$$$$$
                as.numeric(threshold_used)                                                # #$$$$$$$$$$$$$
              )
            ) +  # #$$$$$$$$$$$$$
            ggplot2::scale_y_discrete(limits = rev) +  # #$$$$$$$$$$$$$ FIX: conventional confusion-matrix row order
            .ddesonn_plot_theme()  # #$$$$$$$$$$$$$
          ggplot2::ggsave(heatmap_path, p_conf, width = 5, height = 4, dpi = 300)
          .close_devices()
          if (isTRUE(verbose)) cat("[Eval-Binary][Plot] heatmap saved:", heatmap_path, "\n")
        }, error = function(e) {
          if (isTRUE(verbose)) message("[Eval-Binary][Plot] Failed to save heatmap: ", conditionMessage(e))
        })
      }  # #$$$$$$$$$$$$$
      
      # ============================================================
      # Calibration bins (NO %>% PIPE)                              #$$$$$$$$$$$$$
      # ============================================================
      df_cal <- data.frame(prob = p_pos, label = y_true)  # #$$$$$$$$$$$$$
      df_cal <- dplyr::filter(df_cal, is.finite(prob), is.finite(label))  # #$$$$$$$$$$$$$
      df_cal <- dplyr::mutate(df_cal, prob_bin = dplyr::ntile(prob, 10))  # #$$$$$$$$$$$$$
      df_cal <- dplyr::group_by(df_cal, prob_bin)  # #$$$$$$$$$$$$$
      df_cal <- dplyr::summarise(  # #$$$$$$$$$$$$$
        df_cal,
        bin_mid = mean(prob, na.rm = TRUE),
        actual_rate = mean(label, na.rm = TRUE),
        .groups = "drop"
      )
      df_cal <- dplyr::mutate(df_cal, prob_bin = factor(prob_bin))  # #$$$$$$$$$$$$$
      
      plot1_path   <- file.path(plot_dir, paste0("plot1_bar_actual_rate", suffix, ".png"))
      plot2_path   <- file.path(plot_dir, paste0("plot2_calibration_curve", suffix, ".png"))
      overlay_path <- file.path(plot_dir, paste0("plot_overlay_with_legend_below", suffix, ".png"))
      
      tryCatch({
        p1 <- ggplot2::ggplot(df_cal, ggplot2::aes(x = prob_bin, y = actual_rate)) +
          ggplot2::geom_col(fill = COL_NAVY) +  # #$$$$$$$$$$$$$
          ggplot2::labs(title = paste("Observed Rate by Risk Bin (", mode_label, ")", sep = ""),
                        x = "Predicted Risk Decile (1=low,10=high)", y = "Observed Positive Rate") +
          .ddesonn_plot_theme()  # #$$$$$$$$$$$$$
        if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
          ggplot2::ggsave(plot1_path, p1, width = 6, height = 4, dpi = 300)
          .close_devices()
          if (isTRUE(verbose)) cat("[Eval-Binary][Plot] plot1 saved:", plot1_path, "\n")
        }  # #$$$$$$$$$$$$$
      }, error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][Plot] plot1 failed: ", conditionMessage(e)))
      
      tryCatch({
        p2 <- ggplot2::ggplot(df_cal, ggplot2::aes(x = bin_mid, y = actual_rate)) +
          ggplot2::geom_line(linewidth = 1.2, color = "black") +  # #$$$$$$$$$$$$$
          ggplot2::geom_point(size = 3, color = "black") +  # #$$$$$$$$$$$$$
          ggplot2::geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
          ggplot2::labs(title = paste("Calibration Curve (", mode_label, ")", sep = ""),
                        x = "Avg Predicted Probability", y = "Observed Rate") +
          .ddesonn_plot_theme()  # #$$$$$$$$$$$$$
        if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
          ggplot2::ggsave(plot2_path, p2, width = 6, height = 4, dpi = 300)
          .close_devices()
          if (isTRUE(verbose)) cat("[Eval-Binary][Plot] plot2 saved:", plot2_path, "\n")
        }  # #$$$$$$$$$$$$$
      }, error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][Plot] plot2 failed: ", conditionMessage(e)))
      
      tryCatch({
        p3 <- ggplot2::ggplot(df_cal, ggplot2::aes(x = prob_bin)) +
          ggplot2::geom_col(ggplot2::aes(y = actual_rate), fill = COL_NAVY) +  # #$$$$$$$$$$$$$
          ggplot2::geom_point(ggplot2::aes(y = bin_mid), size = 3, shape = 21, stroke = 1.2, color = COL_NAVY) +  # #$$$$$$$$$$$$$
          ggplot2::labs(title = paste("Overlay: Observed vs Predicted (", mode_label, ")", sep = ""),
                        x = "Predicted Risk Decile", y = "Rate", fill = NULL, color = NULL) +
          .ddesonn_plot_theme() +  # #$$$$$$$$$$$$$
          ggplot2::theme(legend.position = "bottom")
        if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
          ggplot2::ggsave(overlay_path, p3, width = 6, height = 4, dpi = 300)
          .close_devices()
          if (isTRUE(verbose)) cat("[Eval-Binary][Plot] overlay saved:", overlay_path, "\n")
        }  # #$$$$$$$$$$$$$
      }, error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][Plot] overlay plot failed: ", conditionMessage(e)))
      
      invisible(list(
        heatmap_path = heatmap_path,
        plot1_path = plot1_path,
        plot2_path = plot2_path,
        overlay_path = overlay_path
      ))
    }
    
    # ================================================================
    # SECTION: Accuracy plot gating (FIXED)
    # ================================================================
    artifacts <- list()
    # #$$$$$$$$$$$$$ FIX: accuracy_plot (logical) gates all accuracy plots
    # #$$$$$$$$$$$$$ FIX: accuracy_plot_mode selects which (fixed/tuned/both)
    if (isTRUE(accuracy_plot) && accuracy_plot_mode %in% c("accuracy","both")) {  # #$$$$$$$$$$$$$
      artifacts$fixed <- maybe_plot_binary("accuracy", y_pred_fixed, 0.5, "_fixed")
    }
    if (isTRUE(accuracy_plot) && accuracy_plot_mode %in% c("accuracy_tuned","both")) {  # #$$$$$$$$$$$$$
      artifacts$tuned <- maybe_plot_binary(sprintf("accuracy_tuned (thr=%.2f)", best_thr),
                                           y_pred_tuned, best_thr, "_tuned")
    }
    
    # PR curve (object always computed; PNG only if plot_pr/viewAllPlots)  # #$$$$$$$$$$$$$
    # ================================================================
    # SECTION: Precision???Recall Curve (Binary)                         #$$$$$$$$$$$$$
    # - FIX: remove "AUC =" from PR title (PR uses AUPRC, not ROC AUC)  #$$$$$$$$$$$$$
    # - FIX: remove any "???" title artifacts (use clean title)         #$$$$$$$$$$$$$
    # - PRESERVE: PRROC::pr.curve usage + AUPRC legend + gating         #$$$$$$$$$$$$$
    # ================================================================
    
    labels_numeric <- as.numeric(y_true)
    probs_numeric  <- as.numeric(p_pos)
    
    pr_obj <- tryCatch(
      PRROC::pr.curve(
        scores.class0 = probs_numeric[labels_numeric == 1],
        scores.class1 = probs_numeric[labels_numeric == 0],
        curve = TRUE
      ),
      error = function(e) NULL
    )
    
    auprc_val <- tryCatch(
      round(pr_obj$auc.integral, 6),
      error = function(e) NA_real_
    )
    
    pr_png <- file.path(plot_dir, "pr_curve.png")
    
    if ((isTRUE(viewAllPlots) || isTRUE(plot_pr)) && !is.null(pr_obj)) {  # #$$$$$$$$$$$$$
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch({
          grDevices::png(pr_png, width = 800, height = 600)
          
          # #$$$$$$$$$$$$$ FIX: clean PR title (no AUC, no ???)
          plot(pr_obj, main = "Precision-Recall Curve", lwd = 2, adj = 0.5, auc.main = FALSE)  #$$$$$$$$$$$$$
          
          if (isTRUE(show_auprc) && is.finite(auprc_val)) {  # #$$$$$$$$$$$$$
            legend(
              "bottomright",
              legend = sprintf("AUPRC = %.4f", auprc_val),
              bty = "n"
            )
          }
          
          graphics::grid()
          grDevices::dev.off()
          
          if (isTRUE(verbose)) cat("[Eval-Binary][PR] PR PNG saved:", pr_png, "\n")
        }, error = function(e) {
          if (isTRUE(verbose)) message("[Eval-Binary][PR] PR plot failed: ", conditionMessage(e))
          tryCatch(grDevices::dev.off(), error = function(e2) NULL)
        })
      }  # #$$$$$$$$$$$$$
    }
    
    
    # Misclassified (PRESERVED)
    binary_preds_fixed <- y_pred_fixed
    labels_flat <- as.vector(y_true)
    
    wrong_idx <- which(binary_preds_fixed != labels_flat)
    misclassified <- if (length(wrong_idx)) {
      cbind(
        predicted_prob = probs_numeric[wrong_idx],
        predicted_label = binary_preds_fixed[wrong_idx],
        actual_label = labels_flat[wrong_idx],
        as.data.frame(X_validation)[wrong_idx, , drop = FALSE]
      )
    } else {
      data.frame(predicted_prob = numeric(0),
                 predicted_label = integer(0),
                 actual_label = integer(0))
    }
    
    conf_matrix <- matrix(c(TP, FN, FP, TN), nrow = 2, byrow = TRUE,
                          dimnames = list("Actual" = c("Positive (1)", "Negative (0)"),
                                          "Predicted" = c("Positive (1)", "Negative (0)")))
    conf_long <- reshape2::melt(conf_matrix)
    colnames(conf_long) <- c("Actual", "Predicted", "Count")
    
    # Rdata_predictions
    # ============================================================
    # Validation Predictions Frame (NO %>% PIPE)
    # ============================================================
    
    Rdata_predictions <- as.data.frame(X_validation)
    
    Rdata_predictions[["label"]]       <- labels_flat
    Rdata_predictions[["Predictions"]] <- binary_preds_fixed
    Rdata_predictions[["prob"]]        <- probs_numeric
    
    mean_0 <- suppressWarnings(mean(Rdata_predictions$prob[Rdata_predictions$label == 0], na.rm = TRUE))
    mean_1 <- suppressWarnings(mean(Rdata_predictions$prob[Rdata_predictions$label == 1], na.rm = TRUE))
    commentary_text <- if (is.finite(mean_0) && is.finite(mean_1)) {
      if (mean_0 < 0.2 && mean_1 > 0.8) {
        sprintf("Since your model produces mean %.4f for true label 0, and %.4f for true label 1, it???s making sharp, confident, and accurate predictions.", mean_0, mean_1)
      } else if (mean_0 > 0.35 && mean_1 < 0.65) {
        sprintf("Warning: predicted probabilities are close together (%.4f vs %.4f) ??? model may not be separating classes clearly.", mean_0, mean_1)
      } else {
        sprintf("Model separation is moderate (%.4f vs %.4f) ??? might benefit from output sharpening or additional tuning.", mean_0, mean_1)
      }
    } else {
      "One or both class mean probabilities are NA ??? likely due to class imbalance or empty subset."
    }
    commentary_df_means <- data.frame(Interpretation = commentary_text)
    
    metrics_legacy <- data.frame(
      Accuracy  = acc_fixed,
      Precision = pre_fixed,
      Recall    = rec_fixed,
      F1_Score  = f1_fixed,
      TP = TP, TN = TN, FP = FP, FN = FN,
      AUC = auc_val, AUPRC = auprc_val,
      Threshold = 0.5,               
      Threshold_Tuned = best_thr
    )
    
    misclassified <- as.data.frame(misclassified)
    if (nrow(misclassified)) {
      misclassified$Type <- ifelse(
        misclassified$predicted_label == 1 & misclassified$actual_label == 0, "False Positive",
        "False Negative"
      )
      misclassified_sorted <- misclassified[order(-misclassified$predicted_prob), , drop = FALSE]
    } else {
      misclassified_sorted <- misclassified
    }
    
    # Library metrics (PRESERVED; safe wrappers; verbose gated)
    safe_call <- function(fn, ...) {
      f <- get0(fn, ifnotfound = NULL)
      if (is.function(f)) {
        tryCatch(as.numeric(f(SONN, Rdata, labels, CLASSIFICATION_MODE, probs_use, ...)),
                 error = function(e) NA_real_)
      } else NA_real_
    }
    
    lib_metrics <- data.frame(
      quantization_error = tryCatch(get0("quantization_error", ifnotfound=NULL)(SONN, Rdata, run_id, verbose), error=function(e) NA_real_),
      topographic_error  = tryCatch(get0("topographic_error", ifnotfound=NULL)(SONN, Rdata, threshold, verbose), error=function(e) NA_real_),
      clustering_quality_db = tryCatch(get0("clustering_quality_db", ifnotfound=NULL)(SONN, Rdata, cluster_assignments, verbose), error=function(e) NA_real_),
      MSE   = safe_call("MSE"),
      MAE   = safe_call("MAE"),
      RMSE  = safe_call("RMSE"),
      R2    = safe_call("R2"),
      MAPE  = safe_call("MAPE"),
      SMAPE = safe_call("SMAPE"),
      WMAPE = safe_call("WMAPE"),
      MASE  = safe_call("MASE"),
      accuracy  = safe_call("accuracy"),
      precision = safe_call("precision"),
      recall    = safe_call("recall"),
      f1_score  = safe_call("f1_score"),
      hit_rate  = tryCatch(get0("hit_rate", ifnotfound=NULL)(SONN, Rdata, CLASSIFICATION_MODE, probs_use, labels, verbose), error=function(e) NA_real_),
      ndcg      = tryCatch(get0("ndcg", ifnotfound=NULL)(SONN, Rdata, CLASSIFICATION_MODE, probs_use, labels, verbose), error=function(e) NA_real_),
      diversity = tryCatch(get0("diversity", ifnotfound=NULL)(SONN, Rdata, CLASSIFICATION_MODE, probs_use, verbose), error=function(e) NA_real_),
      serendipity = tryCatch(get0("serendipity", ifnotfound=NULL)(SONN, Rdata, CLASSIFICATION_MODE, probs_use, verbose), error=function(e) NA_real_),
      generalization_ability = tryCatch(get0("generalization_ability", ifnotfound=NULL)(SONN, Rdata, labels, CLASSIFICATION_MODE, probs_use, verbose = FALSE), error=function(e) NA_real_),
      speed        = tryCatch(get0("speed", ifnotfound=NULL)(SONN, predicted_outputAndTime$prediction_time %||% NA_real_, verbose), error=function(e) NA_real_),
      speed_learn  = tryCatch(get0("speed_learn", ifnotfound=NULL)(SONN, predicted_outputAndTime$learn_time %||% learn_time %||% NA_real_, verbose), error=function(e) NA_real_),
      memory_usage = tryCatch(get0("memory_usage", ifnotfound=NULL)(SONN, Rdata, verbose), error=function(e) NA_real_),
      robustness   = tryCatch(get0("robustness", ifnotfound=NULL)(
        SONN, Rdata, labels, lr, CLASSIFICATION_MODE, num_epochs, model_iter_num,
        probs_use, ensemble_number, weights, biases, activation_functions, dropout_rates, verbose),
        error=function(e) NA_real_)
    )
    
    # ================================================================
    # SECTION: Structured report object (new)
    # ================================================================
    report <- list(
      mode = "binary",
      configuration = list(
        classification_mode = CLASSIFICATION_MODE,
        accuracy_plot = accuracy_plot,
        accuracy_plot_mode = accuracy_plot_mode,  # #$$$$$$$$$$$$$
        tuned_threshold_override = tuned_threshold_override,
        show_auprc = show_auprc,
        viewAllPlots = viewAllPlots,  # #$$$$$$$$$$$$$
        plot_roc = plot_roc,          # #$$$$$$$$$$$$$
        plot_pr  = plot_pr            # #$$$$$$$$$$$$$
      ),
      paths = list(
        artifacts_root = artifacts_root,
        reports_dir = reports_dir,
        plot_dir = plot_dir,
        xlsx = report_wb_path,
        rds = report_rds_path
      ),
      tables = list(
        Rdata_Predictions = Rdata_predictions,
        Metrics_Summary   = metrics_legacy,
        Misclassified     = misclassified_sorted,
        Prediction_Means  = data.frame(Mean_Prob_Label_0 = mean_0, Mean_Prob_Label_1 = mean_1),
        Metrics_Library   = t(lib_metrics)
      ),
      metrics = list(
        fixed = list(threshold = 0.5, accuracy = acc_fixed, precision = pre_fixed, recall = rec_fixed, f1 = f1_fixed,
                     confusion = list(TP = TP, FP = FP, TN = TN, FN = FN)),
        tuned = list(threshold = best_thr, accuracy = acc_tuned, precision = pre_tuned, recall = rec_tuned, f1 = f1_tuned),
        auc = auc_val,
        auprc = auprc_val
      ),
      curves = list(
        roc_curve = roc_df
        # PR curve is retained as a PNG or PRROC object only when you plot; keeping it light here
      ),
      artifacts = list(
        plots = list(
          roc_png = if (file.exists(roc_png)) roc_png else NA_character_,
          pr_png  = if (file.exists(pr_png)) pr_png else NA_character_,
          fixed_plots = artifacts$fixed,
          tuned_plots = artifacts$tuned
        )
      )
    )
    
    # Optional Excel export (PRESERVED layout; now gated)
    if (isTRUE(export_excel)) {
      if (isTRUE(verbose)) cat("[Eval-Binary][WB] createWorkbook()\n")
      wb <- openxlsx::createWorkbook()
      
      openxlsx::addWorksheet(wb, "Fixed")
      cm_tbl <- data.frame(
        Metric = c("TP","FP","TN","FN","Accuracy","Precision","Recall","F1","Threshold"),
        Value  = c(TP, FP, TN, FN, acc_fixed, pre_fixed, rec_fixed, f1_fixed, 0.5)
      )
      suppressWarnings(openxlsx::writeData(wb, "Fixed", cm_tbl))
      
      openxlsx::addWorksheet(wb, "Tuned")
      tuned_tbl <- data.frame(
        Metric = c("Accuracy","Precision","Recall","F1","Best_Threshold"),
        Value  = c(acc_tuned, pre_tuned, rec_tuned, f1_tuned, best_thr)
      )
      suppressWarnings(openxlsx::writeData(wb, "Tuned", tuned_tbl))
      
      openxlsx::addWorksheet(wb, "ROC")
      suppressWarnings(openxlsx::writeData(wb, "ROC", data.frame(AUC = auc_val, AUPRC = auprc_val)))
      
      if (file.exists(roc_png)) {
        if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
          tryCatch(openxlsx::insertImage(wb, "ROC", roc_png, startRow = 5, startCol = 1, width = 6, height = 4),
                   error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] insertImage ROC failed: ", conditionMessage(e)))
        }  # #$$$$$$$$$$$$$
      }
      if (file.exists(pr_png)) {
        if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
          tryCatch(openxlsx::insertImage(wb, "ROC", pr_png, startRow = 25, startCol = 1, width = 6, height = 4),
                   error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] insertImage PR failed: ", conditionMessage(e)))
        }  # #$$$$$$$$$$$$$
      }
      
      if (!is.null(artifacts$fixed)) {
        for (p in unlist(artifacts$fixed, use.names = FALSE)) {
          if (file.exists(p)) {
            if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
              tryCatch(openxlsx::insertImage(wb, "Fixed", p, startRow = 20, startCol = 1, width = 6, height = 4),
                       error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] insertImage (Fixed) failed: ", conditionMessage(e)))
            }  # #$$$$$$$$$$$$$
          }
        }
      }
      if (!is.null(artifacts$tuned)) {
        for (p in unlist(artifacts$tuned, use.names = FALSE)) {
          if (file.exists(p)) {
            if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
              tryCatch(openxlsx::insertImage(wb, "Tuned", p, startRow = 20, startCol = 1, width = 6, height = 4),
                       error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] insertImage (Tuned) failed: ", conditionMessage(e)))
            }  # #$$$$$$$$$$$$$
          }
        }
      }
      
      openxlsx::addWorksheet(wb, "Rdata_Predictions")
      suppressWarnings(openxlsx::writeData(wb, "Rdata_Predictions", Rdata_predictions))
      
      openxlsx::addWorksheet(wb, "Metrics_Summary")
      suppressWarnings(openxlsx::writeData(wb, "Metrics_Summary", metrics_legacy))
      
      openxlsx::addWorksheet(wb, "Prediction_Means")
      suppressWarnings(openxlsx::writeData(wb, "Prediction_Means",
                                           data.frame(Mean_Prob_Label_0 = mean_0, Mean_Prob_Label_1 = mean_1)))
      suppressWarnings(openxlsx::writeData(wb, "Prediction_Means", commentary_df_means, startRow = 5))
      
      openxlsx::addWorksheet(wb, "Misclassified")
      suppressWarnings(openxlsx::writeData(wb, "Misclassified", misclassified_sorted))
      
      openxlsx::addWorksheet(wb, "Misclass_Summary")
      if (nrow(misclassified_sorted)) {
        known_cols <- intersect(c("age","serum_creatinine","ejection_fraction","time"), names(misclassified_sorted))
        if (length(known_cols)) {
          # (PRESERVED) dplyr usage is fine; no %>% pipe needed here
          summary_by_type <- dplyr::group_by(misclassified_sorted, Type)
          summary_by_type <- dplyr::summarise(
            summary_by_type,
            dplyr::across(dplyr::all_of(known_cols), function(x) mean(x, na.rm = TRUE))
          )
        } else {
          summary_by_type <- data.frame()
        }
        
        suppressWarnings(openxlsx::writeData(wb, "Misclass_Summary", summary_by_type))
        
        legacy_mis_heat <- find_plot_file("misclassification_heatmap.png")
        legacy_box_sc   <- find_plot_file("boxplot_serum_creatinine.png")
        
        if (!is.null(legacy_mis_heat)) {
          if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
            tryCatch(openxlsx::insertImage(wb, "Misclass_Summary", legacy_mis_heat, startRow = 10, startCol = 1, width = 6, height = 4),
                     error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] legacy misclass heatmap insert failed: ", conditionMessage(e)))
          }  # #$$$$$$$$$$$$$
        }
        if (!is.null(legacy_box_sc)) {
          if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
            tryCatch(openxlsx::insertImage(wb, "Misclass_Summary", legacy_box_sc, startRow = 25, startCol = 1, width = 6, height = 4),
                     error = function(e) if (isTRUE(verbose)) message("[Eval-Binary][WB] legacy boxplot insert failed: ", conditionMessage(e)))
          }  # #$$$$$$$$$$$$$
        }
      }
      
      openxlsx::addWorksheet(wb, "Metrics_Library")
      suppressWarnings(openxlsx::writeData(wb, "Metrics_Library", t(lib_metrics)))
      
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch({
          if (isTRUE(verbose)) cat("[Eval-Binary][WB] saveWorkbook() begin\n")
          openxlsx::saveWorkbook(wb, report_wb_path, overwrite = TRUE)
          if (isTRUE(verbose)) cat("[Eval-Binary][WB] saveWorkbook() done\n")
        }, error = function(e) {
          if (isTRUE(verbose)) message("[Eval-Binary][WB] Workbook save failed: ", conditionMessage(e))
        })
      }  # #$$$$$$$$$$$$$
    }
    
    # Optional RDS save
    if (isTRUE(save_rds)) {
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch(saveRDS(report, report_rds_path),
                 error = function(e) if (isTRUE(verbose)) message("[Eval-Binary] saveRDS failed: ", conditionMessage(e)))
        if (isTRUE(verbose)) cat("[Eval-Binary] RDS saved:", report_rds_path, "\n")
      }  # #$$$$$$$$$$$$$
    }
    
    if (isTRUE(verbose)) cat("[Eval-Binary] RETURN\n")
    return(list(
      best_threshold  = best_thr,
      accuracy        = acc_fixed,
      precision       = pre_fixed,
      recall          = rec_fixed,
      f1_score        = f1_fixed,
      accuracy_tuned  = acc_tuned,
      precision_tuned = pre_tuned,
      recall_tuned    = rec_tuned,
      f1_tuned        = f1_tuned,
      confusion_matrix = list(TP = TP, FP = FP, TN = TN, FN = FN),
      y_pred_class       = y_pred_fixed,
      y_pred_class_tuned = y_pred_tuned,
      auc = auc_val,
      roc_curve = roc_df,
      report = report
    ))
  } # end binary
  
  # ================================================================
  # SECTION: Multiclass branch (PRESERVED; verbose + plots gated)
  # ================================================================
  if (isTRUE(verbose)) cat("[Eval-Multiclass] Enter\n")
  
  if (ncol(L) > 1L) {
    y_true_ids <- max.col(L, ties.method = "first")
  } else {
    cls <- suppressWarnings(as.integer(L[,1]))
    if (min(cls, na.rm = TRUE) == 0L) cls <- cls + 1L
    cls[!is.finite(cls)] <- 1L
    K <- max(2L, ncol(P))
    cls[cls < 1L] <- 1L; cls[cls > K] <- K
    y_true_ids <- cls
  }
  if (ncol(P) > 1L) {
    pred_ids <- max.col(P, ties.method = "first")
    K <- ncol(P)
  } else {
    pred_ids <- rep(1L, length(y_true_ids)); K <- max(y_true_ids, na.rm = TRUE)
  }
  
  acc_mc <- mean(pred_ids == y_true_ids, na.rm = TRUE)
  if (isTRUE(verbose)) cat("[Eval-Multiclass] K:", K, "  accuracy:", acc_mc, "\n")
  
  TPk <- FPk <- FNk <- rep(0L, K)
  for (k in seq_len(K)) {
    TPk[k] <- sum(pred_ids == k & y_true_ids == k)
    FPk[k] <- sum(pred_ids == k & y_true_ids != k)
    FNk[k] <- sum(pred_ids != k & y_true_ids == k)
  }
  Prec_k <- ifelse((TPk + FPk) > 0, TPk / (TPk + FPk), 0)
  Rec_k  <- ifelse((TPk + FNk) > 0, TPk / (TPk + FNk), 0)
  F1_k   <- ifelse((Prec_k + Rec_k) > 0, 2 * Prec_k * Rec_k / (Prec_k + Rec_k), 0)
  macro_precision <- mean(Prec_k)
  macro_recall    <- mean(Rec_k)
  macro_f1        <- mean(F1_k)
  
  conf_tab <- table(Actual=factor(y_true_ids, levels=1:K), Predicted=factor(pred_ids, levels=1:K))
  conf_matrix_df <- as.data.frame(conf_tab); names(conf_matrix_df)[3] <- "Count"
  
  heatmap_path_mc <- file.path(plot_dir, "confusion_matrix_multiclass_heatmap.png")
  if (isTRUE(do_any_plots)) {  # #$$$$$$$$$$$$$
    if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
      tryCatch({
        p_mc <- ggplot2::ggplot(conf_matrix_df, ggplot2::aes(x=factor(Predicted), y=factor(Actual), fill=Count)) +
          ggplot2::geom_tile(color="white") +
          ggplot2::geom_text(ggplot2::aes(label=Count), size=3, fontface="bold") +
          ggplot2::scale_fill_gradient(low="white", high="#D73027") +  # #$$$$$$$$$$$$$
          ggplot2::labs(title="Confusion Matrix", x="Predicted", y="Actual") +  # #$$$$$$$$$$$$$
          .ddesonn_plot_theme()  # #$$$$$$$$$$$$$
        ggplot2::ggsave(heatmap_path_mc, p_mc, width=6, height=5, dpi=300)
        .close_devices()
        if (isTRUE(verbose)) cat("[Eval-Multiclass] heatmap saved:", heatmap_path_mc, "\n")
      }, error = function(e) if (isTRUE(verbose)) message("[Eval-Multiclass] heatmap failed: ", conditionMessage(e)))
    }  # #$$$$$$$$$$$$$
  }
  
  if (isTRUE(verbose)) {
    cat("=== diagnostics for evaluate_predictions_report ===\n")
    cat("nrow(X_validation):", NROW(X_validation), "\n")
    cat("length(y_true_ids):", length(y_true_ids), "\n")
    cat("length(pred_ids):", length(pred_ids), "\n\n")
    str(list(
      X_validation = utils::head(X_validation, 3),
      y_true_ids   = utils::head(y_true_ids, 10),
      pred_ids     = utils::head(pred_ids, 10)
    ))
  }
  
  combine_for_report <- function(X, y, p, verbose = TRUE) {
    nX <- NROW(X); ny <- length(y); np <- length(p)
    if (is.matrix(y) && ncol(y) == 1L) y <- as.vector(y)
    if (is.matrix(p) && ncol(p) == 1L) p <- as.vector(p)
    
    rnx <- rownames(X); ny_names <- names(y); np_names <- names(p)
    
    if (!is.null(rnx) && (!is.null(ny_names) || !is.null(np_names))) {
      common <- rnx
      if (!is.null(ny_names)) common <- intersect(common, ny_names)
      if (!is.null(np_names)) common <- intersect(common, np_names)
      if (length(common)) {
        Xdf <- as.data.frame(X, check.names = FALSE)
        return(data.frame(
          Xdf[common, , drop = FALSE],
          label = y[common],
          pred  = p[common],
          check.names = FALSE
        ))
      }
    }
    
    m <- min(nX, ny, np)
    if (isTRUE(verbose) && (nX != m || ny != m || np != m)) {  # #$$$$$$$$$$$$$
      message(sprintf(
        "[EvaluatePredictionsReport] Row mismatch: X=%d, y=%d, p=%d ??? truncating to %d rows for 'Combined'.",
        nX, ny, np, m
      ))
    }
    Xdf <- as.data.frame(X, check.names = FALSE)
    data.frame(
      Xdf[seq_len(m), , drop = FALSE],
      label = y[seq_len(m)],
      pred  = p[seq_len(m)],
      check.names = FALSE
    )
  }
  
  combined_df <- combine_for_report(X_validation, y_true_ids, pred_ids, verbose = isTRUE(verbose))
  ms <- data.frame(
    Class     = c(as.character(seq_len(K)), "macro avg"),
    Precision = c(Prec_k, macro_precision),
    Recall    = c(Rec_k,  macro_recall),
    F1_Score  = c(F1_k,   macro_f1),
    Accuracy  = c(rep(acc_mc, K), acc_mc)
  )
  
  predictions_df <- combine_for_report(X_validation, y_true_ids, pred_ids, verbose = FALSE)
  rid <- rownames(predictions_df)
  if (is.null(rid)) rid <- seq_len(nrow(predictions_df))
  predictions_df <- cbind(RowID = rid, predictions_df)
  
  report <- list(
    mode = "multiclass",
    configuration = list(
      classification_mode = CLASSIFICATION_MODE,
      viewAllPlots = viewAllPlots,  # #$$$$$$$$$$$$$
      plot_roc = plot_roc,          # #$$$$$$$$$$$$$
      plot_pr  = plot_pr            # #$$$$$$$$$$$$$
    ),
    paths = list(
      artifacts_root = artifacts_root,
      reports_dir = reports_dir,
      plot_dir = plot_dir,
      xlsx = report_wb_path,
      rds = report_rds_path
    ),
    tables = list(
      Combined = combined_df,
      Metrics_Summary = ms,
      Rdata_Predictions = predictions_df
    ),
    metrics = list(
      accuracy = acc_mc,
      macro_precision = macro_precision,
      macro_recall = macro_recall,
      macro_f1 = macro_f1
    ),
    artifacts = list(
      plots = list(
        multiclass_heatmap = if (file.exists(heatmap_path_mc)) heatmap_path_mc else NA_character_
      )
    )
  )
  
  # Optional Excel export (PRESERVED; gated)
  if (isTRUE(export_excel)) {
    wb <- openxlsx::createWorkbook()
    openxlsx::addWorksheet(wb, "Combined")
    suppressWarnings(openxlsx::writeData(wb, "Combined", combined_df))
    
    openxlsx::addWorksheet(wb, "Metrics_Summary")
    suppressWarnings(openxlsx::writeData(wb, "Metrics_Summary", ms))
    if (file.exists(heatmap_path_mc)) {
      if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
        tryCatch(openxlsx::insertImage(wb, "Metrics_Summary", heatmap_path_mc, startRow = nrow(ms) + 6,
                                       startCol = 1, width = 6, height = 4),
                 error = function(e) if (isTRUE(verbose)) message("[Eval-Multiclass] insertImage failed: ", conditionMessage(e)))
      }  # #$$$$$$$$$$$$$
    }
    
    openxlsx::addWorksheet(wb, "Rdata_Predictions")
    suppressWarnings(openxlsx::writeData(wb, "Rdata_Predictions", predictions_df))
    
    if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
      tryCatch(openxlsx::saveWorkbook(wb, report_wb_path, overwrite = TRUE),
               error = function(e) if (isTRUE(verbose)) message("[Eval-Multiclass] Workbook save failed: ", conditionMessage(e)))
      if (isTRUE(verbose)) cat("[Eval-Multiclass] Workbook saved.\n")
    }  # #$$$$$$$$$$$$$
  }
  
  # Optional RDS save
  if (isTRUE(save_rds)) {
    if (isTRUE(saveEnabled)) {  # #$$$$$$$$$$$$$
      tryCatch(saveRDS(report, report_rds_path),
               error = function(e) if (isTRUE(verbose)) message("[Eval-Multiclass] saveRDS failed: ", conditionMessage(e)))
      if (isTRUE(verbose)) cat("[Eval-Multiclass] RDS saved:", report_rds_path, "\n")
    }  # #$$$$$$$$$$$$$
  }
  
  if (isTRUE(verbose)) cat("[Eval-Multiclass] RETURN\n")
  return(list(
    best_threshold   = NA_real_,
    accuracy         = acc_mc,
    precision        = macro_precision,
    recall           = macro_recall,
    f1_score         = macro_f1,
    accuracy_tuned   = NA_real_,
    precision_tuned  = NA_real_,
    recall_tuned     = NA_real_,
    f1_tuned         = NA_real_,
    confusion_matrix = NULL,
    y_pred_class     = pred_ids,
    y_pred_class_tuned = NULL,
    auc = NA_real_,
    roc_curve = NULL,
    report = report
  ))
}
