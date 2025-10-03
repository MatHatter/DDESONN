# ===============================================================
# DeepDynamic — DDESONN
# Deep Dynamic Ensemble Self-Organizing Neural Network
# ---------------------------------------------------------------
# Copyright (c) 2024-2025 Mathew William Fok
# 
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
# evaluate_predictions_report.R  (FULL — accuracy + accuracy_tuned + ROC/AUC)
# ================================================================
source("utils/utils.R")

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(tidyr)
  library(gridExtra)
  library(grid)
  library(pROC)
  library(PRROC)
  library(ggplotify)
  library(openxlsx)
})

# ----------------------------------------------------------------
# EvaluatePredictionsReport
# ----------------------------------------------------------------
EvaluatePredictionsReport <- function(
    X_validation, y_validation, CLASSIFICATION_MODE,
    probs,                       # last-epoch fallback (matrix or vector)
    predicted_outputAndTime,     # metadata list from training (optional)
    threshold_function,          # kept for signature compatibility (not used)
    all_best_val_probs,          # best snapshot probs (optional)
    all_best_val_labels,         # best snapshot labels (optional)
    verbose = FALSE,
    # Plot selection ONLY (results always include both fixed and tuned):
    accuracy_plot = c("accuracy", "accuracy_tuned", "both"),
    tuned_threshold_override = NULL,
    SONN
) {
  accuracy_plot <- match.arg(accuracy_plot)
  
  # ------------------------- Setup: plots dir -------------------------
  plot_dir <- file.path(getwd(), "EvaluatePredictionsReportPlots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  
  # ------------------------- safety defaults -------------------------
  if (!exists("viewTables", inherits = TRUE)) viewTables <- FALSE
  if (!exists("ML_NN", inherits = TRUE))      ML_NN      <- FALSE
  if (!exists("train", inherits = TRUE))      train      <- FALSE
  
  # ------------------------- Inspect predictions/errors (optional) ---
  pred_vec   <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$learn_output),
                         error = function(e) rep(NA_real_, length.out = 0))
  err_vec    <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$error),
                         error = function(e) rep(NA_real_, length.out = 0))
  labels_vec <- tryCatch(as.vector(y_validation), error = function(e) rep(NA_real_, length.out = 0))
  max_points <- min(length(pred_vec), length(err_vec), length(labels_vec))
  if (max_points > 0) {
    tryCatch({
      png(file.path(plot_dir, "pred_vs_error_scatter.png"), width = 800, height = 600)
      plot(pred_vec[seq_len(max_points)], err_vec[seq_len(max_points)],
           main = "Prediction vs. Error", xlab = "Prediction", ylab = "Error",
           col = "steelblue", pch = 16)
      abline(h = 0, col = "gray", lty = 2)
      dev.off()
    }, error = function(e) message("⚠️ Pred-vs-Error plot failed: ", e$message))
  }
  
  # ------------------------- weights summary (robust) -----------------
  if (isTRUE(ML_NN)) {
    w_mat <- tryCatch(as.matrix(predicted_outputAndTime$weights_record[[1]]),
                      error = function(e) matrix(NA_real_, nrow = 0, ncol = 0))
    if (length(w_mat)) {
      weights_summary <- round(rowMeans(w_mat), 5)
      if (verbose) { cat(">> Multi-layer weights summary (first layer):\n"); print(weights_summary) }
    }
  } else {
    w_raw <- tryCatch(predicted_outputAndTime$weights_record[[1]], error = function(e) numeric(0))
    if (length(w_raw)) {
      weights_summary <- round(as.numeric(w_raw), 5)
      if (verbose) { cat(">> Single-layer weights summary:\n"); print(weights_summary) }
    }
  }
  
  # ------------------------- Select evaluation data -------------------
  # Prefer BEST snapshot from training if available (parity with best_val_* in loop)
  use_best <- (!is.null(all_best_val_probs) && !is.null(all_best_val_labels))
  if (use_best) {
    probs_use  <- all_best_val_probs
    labels_use <- all_best_val_labels
    if (verbose) cat("[Eval] Using BEST validation snapshot (probs/labels) from training.\n")
  } else {
    probs_use  <- probs
    labels_use <- y_validation
    if (verbose) cat("[Eval] Using LAST-epoch predictions (no best snapshot provided).\n")
  }
  
  # Coerce to matrices and align rows
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
  if (n_eff <= 0) stop("[EvaluatePredictionsReport] No overlapping rows between probs and labels.")
  L <- L[seq_len(n_eff), , drop = FALSE]
  P <- P[seq_len(n_eff), , drop = FALSE]
  
  # ------------------------- Mode inference ---------------------------
  infer_mode <- function(L, P, fallback = "binary") {
    if (tolower(CLASSIFICATION_MODE) %in% c("binary","multiclass","regression")) return(tolower(CLASSIFICATION_MODE))
    if (max(ncol(L), ncol(P)) > 1L) "multiclass" else fallback
  }
  mode <- infer_mode(L, P, "binary")
  if (verbose) cat(sprintf("[Eval] mode=%s | n_eff=%d | ncol(L)=%d | ncol(P)=%d\n", mode, n_eff, ncol(L), ncol(P)))
  
  # ------------------------- Regression branch ------------------------
  if (identical(mode, "regression")) {
    y    <- suppressWarnings(as.numeric(L[,1]))
    yhat <- suppressWarnings(as.numeric(P[,1]))
    keep <- is.finite(y) & is.finite(yhat)
    y    <- y[keep]
    yhat <- yhat[keep]
    if (!length(y)) stop("Regression mode: no finite overlapping y / yhat.")
    
    residuals <- yhat - y
    SSE  <- sum(residuals^2)
    SST  <- sum((y - mean(y))^2)
    RMSE <- sqrt(mean(residuals^2))
    MAE  <- mean(abs(residuals))
    MAPE <- if (any(y != 0)) mean(abs(residuals / y)) else NA_real_
    R2   <- if (SST > 0) 1 - SSE/SST else NA_real_
    Corr <- suppressWarnings(stats::cor(y, yhat))
    
    # Minimal workbook
    wb <- createWorkbook()
    addWorksheet(wb, "Metrics_Summary")
    suppressWarnings(writeData(wb, "Metrics_Summary",
                               data.frame(Metric=c("RMSE","MAE","MAPE","R2","Correlation"),
                                          Value=c(RMSE,MAE,MAPE,R2,Corr))))
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    
    return(list(
      best_threshold  = NA_real_,
      # headline metrics (not applicable)
      accuracy        = NA_real_,
      precision       = NA_real_,
      recall          = NA_real_,
      f1_score        = NA_real_,
      # tuned set (n/a)
      accuracy_tuned  = NA_real_,
      precision_tuned = NA_real_,
      recall_tuned    = NA_real_,
      f1_tuned        = NA_real_,
      # confusion + preds + ROC (n/a)
      confusion_matrix = NULL,
      y_pred_class     = NULL,
      y_pred_class_tuned = NULL,
      auc = NA_real_,
      roc_curve = NULL
    ))
  }
  
  # ------------------------- Binary branch ----------------------------
  if (identical(mode, "binary")) {
    # Labels → 0/1
    y_true <- if (ncol(L) == 1L) {
      v <- as.numeric(L[,1]); if (all(v %in% c(0,1))) as.integer(v) else as.integer(v >= 0.5)
    } else {
      as.integer(max.col(L, ties.method = "first") - 1L)
    }
    # Probs → p(y=1)
    if (ncol(P) != 1L) stop("[Eval-Binary] Expected 1-column probabilities; got ", ncol(P))
    p_pos <- as.numeric(P[,1])
    
    # ----- FIXED METRICS (accuracy = fixed 0.5, via your helpers) -----
    acc_fixed <- accuracy(
      SONN = SONN,
      Rdata = tryCatch(X_validation[seq_len(n_eff), , drop = FALSE], error = function(e) NULL),
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      verbose = FALSE
    )
    pre_fixed <- precision(
      SONN = SONN,
      Rdata = tryCatch(X_validation[seq_len(n_eff), , drop = FALSE], error = function(e) NULL),
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      verbose = FALSE
    )
    rec_fixed <- recall(
      SONN = SONN,
      Rdata = tryCatch(X_validation[seq_len(n_eff), , drop = FALSE], error = function(e) NULL),
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      verbose = FALSE
    )
    f1_fixed <- f1_score(
      SONN = SONN,
      Rdata = tryCatch(X_validation[seq_len(n_eff), , drop = FALSE], error = function(e) NULL),
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      verbose = FALSE
    )
    cm_fixed <- confusion_matrix(
      SONN = SONN,
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      threshold = 0.5,
      verbose = FALSE
    )
    TP <- cm_fixed$TP; FP <- cm_fixed$FP; TN <- cm_fixed$TN; FN <- cm_fixed$FN
    y_pred_fixed <- as.integer(p_pos >= 0.5)
    
    # ----- ROC / AUC (binary) -----
    roc_obj <- tryCatch(
      pROC::roc(response = y_true, predictor = p_pos, levels = c(0,1), direction = "<", quiet = TRUE),
      error = function(e) NULL
    )
    auc_val <- tryCatch(as.numeric(pROC::auc(roc_obj)), error = function(e) NA_real_)
    roc_df  <- if (!is.null(roc_obj)) {
      data.frame(
        fpr = 1 - roc_obj$specificities,
        tpr = roc_obj$sensitivities,
        threshold = roc_obj$thresholds
      )
    } else NULL
    
    # Optional: plot ROC
    if (!is.null(roc_df) && nrow(roc_df) > 1) {
      try({
        p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
          geom_line(size = 1.1) +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
          labs(title = sprintf("ROC Curve (AUC = %.4f)", auc_val), x = "FPR", y = "TPR") +
          theme_minimal()
        ggsave(filename = file.path(plot_dir, "roc_curve.png"), p_roc, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
      }, silent = TRUE)
    }
    
    # ----- TUNED METRICS -----
    if (is.numeric(tuned_threshold_override) && is.finite(tuned_threshold_override)) {
      DDESONN_set_threshold(tuned_threshold_override)
      if (verbose) message(sprintf("[Eval-Binary] Forced tuned_threshold_override=%.4f", tuned_threshold_override))
    } else {
      invisible(TRUE)
    }
    
    tuned <- accuracy_precision_recall_f1_tuned(
      SONN = SONN,
      Rdata = tryCatch(X_validation[seq_len(n_eff), , drop = FALSE], error = function(e) NULL),
      labels = matrix(y_true, ncol = 1L),
      CLASSIFICATION_MODE = "binary",
      predicted_output = matrix(p_pos, ncol = 1L),
      metric_for_tuning = "accuracy",
      threshold_grid = seq(0.05, 0.95, by = 0.01),
      verbose = verbose
    )
    acc_tuned <- tuned$accuracy
    pre_tuned <- tuned$precision
    rec_tuned <- tuned$recall
    f1_tuned  <- tuned$f1
    best_thr  <- as.numeric(tuned$details$best_threshold)
    y_pred_tuned <- as.integer(tuned$details$y_pred_class)
    
    # ------------------- PLOTTING (selection only) --------------------
    maybe_plot_binary <- function(mode_label, bin_preds, threshold_used, suffix) {
      TPp <- sum(bin_preds == 1 & y_true == 1)
      TNp <- sum(bin_preds == 0 & y_true == 0)
      FPp <- sum(bin_preds == 1 & y_true == 0)
      FNp <- sum(bin_preds == 0 & y_true == 1)
      conf_matrix_df <- data.frame(
        Actual    = c("0","0","1","1"),
        Predicted = c("0","1","0","1"),
        Count     = c(TNp, FPp, FNp, TPp)
      )
      heatmap_path <- file.path(plot_dir, paste0("confusion_matrix_heatmap", suffix, ".png"))
      tryCatch({
        plot_conf_matrix <- ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
          geom_tile(color = "white") +
          geom_text(aes(label = Count), size = 6, fontface = "bold") +
          scale_fill_gradient(low = "white", high = "red") +
          labs(title = paste("Confusion Matrix Heatmap", toupper(mode_label))) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
        ggsave(heatmap_path, plot_conf_matrix, width = 5, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
      }, error = function(e) message("❌ Failed to generate confusion matrix heatmap: ", e$message))
      
      # Calibration & overlay
      df_cal <- data.frame(prob = p_pos, label = y_true) %>%
        filter(is.finite(prob), is.finite(label)) %>%
        mutate(prob_bin = ntile(prob, 10)) %>%
        group_by(prob_bin) %>%
        summarise(
          bin_mid = mean(prob, na.rm = TRUE),
          actual_rate = mean(label, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        mutate(prob_bin = factor(prob_bin))
      
      plot1_path <- file.path(plot_dir, paste0("plot1_bar_actual_rate", suffix, ".png"))
      plot2_path <- file.path(plot_dir, paste0("plot2_calibration_curve", suffix, ".png"))
      overlay_path <- file.path(plot_dir, paste0("plot_overlay_with_legend_below", suffix, ".png"))
      
      tryCatch({
        p1 <- ggplot(df_cal, aes(x = prob_bin, y = actual_rate)) +
          geom_col() +
          labs(title = paste("Observed Rate by Risk Bin (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile (1=low,10=high)", y = "Observed Positive Rate") +
          theme_minimal() + theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot1_path, p1, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
      }, error = function(e) message("❌ plot1 failed: ", e$message))
      
      tryCatch({
        p2 <- ggplot(df_cal, aes(x = bin_mid, y = actual_rate)) +
          geom_line(size = 1.2) + geom_point(size = 3) +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
          labs(title = paste("Calibration Curve (", mode_label, ")", sep = ""),
               x = "Avg Predicted Probability", y = "Observed Rate") +
          theme_minimal() + theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot2_path, p2, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
      }, error = function(e) message("❌ plot2 failed: ", e$message))
      
      tryCatch({
        p3 <- ggplot(df_cal, aes(x = prob_bin)) +
          geom_col(aes(y = actual_rate)) +
          geom_point(aes(y = bin_mid), size = 3, shape = 21, stroke = 1.2) +
          labs(title = paste("Overlay: Observed vs Predicted (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile", y = "Rate", fill = NULL, color = NULL) +
          theme_minimal() + theme(legend.position = "bottom",
                                  plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(overlay_path, p3, width = 6, height = 4, dpi = 300)
        while (!is.null(dev.list())) dev.off()
      }, error = function(e) message("❌ overlay plot failed: ", e$message))
      
      invisible(list(
        heatmap_path = heatmap_path,
        plot1_path = plot1_path,
        plot2_path = plot2_path,
        overlay_path = overlay_path
      ))
    }
    
    artifacts <- list()
    if (accuracy_plot %in% c("accuracy","both")) {
      artifacts$fixed <- maybe_plot_binary("accuracy", y_pred_fixed, 0.5, "_fixed")
    }
    if (accuracy_plot %in% c("accuracy_tuned","both")) {
      artifacts$tuned <- maybe_plot_binary(sprintf("accuracy_tuned (thr=%.2f)", best_thr),
                                           y_pred_tuned, best_thr, "_tuned")
    }
    
    # ------------------- Workbook (two sheets) ------------------------
    wb <- createWorkbook()
    
    addWorksheet(wb, "Fixed")
    cm_tbl <- data.frame(
      Metric = c("TP","FP","TN","FN","Accuracy","Precision","Recall","F1","Threshold"),
      Value  = c(TP, FP, TN, FN, acc_fixed, pre_fixed, rec_fixed, f1_fixed, 0.5)
    )
    suppressWarnings(writeData(wb, "Fixed", cm_tbl))
    
    addWorksheet(wb, "Tuned")
    tuned_tbl <- data.frame(
      Metric = c("Accuracy","Precision","Recall","F1","Best_Threshold"),
      Value  = c(acc_tuned, pre_tuned, rec_tuned, f1_tuned, best_thr)
    )
    suppressWarnings(writeData(wb, "Tuned", tuned_tbl))
    
    # ROC sheet
    addWorksheet(wb, "ROC")
    suppressWarnings(writeData(wb, "ROC", data.frame(AUC = auc_val)))
    roc_png <- file.path(plot_dir, "roc_curve.png")
    if (file.exists(roc_png)) {
      tryCatch(insertImage(wb, "ROC", roc_png, startRow = 5, startCol = 1, width = 6, height = 4),
               error = function(e) {})
    }
    
    # Insert confusion/calibration plots if they exist
    if (!is.null(artifacts$fixed)) {
      for (p in unlist(artifacts$fixed, use.names = FALSE)) {
        if (file.exists(p)) tryCatch(insertImage(wb, "Fixed", p, startRow = 20, startCol = 1, width = 6, height = 4),
                                     error = function(e) {})
      }
    }
    if (!is.null(artifacts$tuned)) {
      for (p in unlist(artifacts$tuned, use.names = FALSE)) {
        if (file.exists(p)) tryCatch(insertImage(wb, "Tuned", p, startRow = 20, startCol = 1, width = 6, height = 4),
                                     error = function(e) {})
      }
    }
    
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    
    # ------------------- Return (clean names) --------------------
    return(list(
      # tuned threshold
      best_threshold  = best_thr,
      # headline metrics (FIXED 0.5) — these drive best_val_acc
      accuracy        = acc_fixed,
      precision       = pre_fixed,
      recall          = rec_fixed,
      f1_score        = f1_fixed,
      # tuned set
      accuracy_tuned  = acc_tuned,
      precision_tuned = pre_tuned,
      recall_tuned    = rec_tuned,
      f1_tuned        = f1_tuned,
      # confusion + preds + ROC
      confusion_matrix = list(TP = TP, FP = FP, TN = TN, FN = FN),
      y_pred_class       = y_pred_fixed,
      y_pred_class_tuned = y_pred_tuned,
      auc = auc_val,
      roc_curve = roc_df
    ))
  }
  
  # ------------------------- Multiclass branch ------------------------
  # True ids
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
  # Pred ids
  if (ncol(P) > 1L) {
    pred_ids <- max.col(P, ties.method = "first")
    K <- ncol(P)
  } else {
    pred_ids <- rep(1L, length(y_true_ids)); K <- max(y_true_ids, na.rm = TRUE)
  }
  
  acc_mc <- mean(pred_ids == y_true_ids, na.rm = TRUE)
  
  # Macro metrics
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
  
  # Heatmap
  conf_tab <- table(Actual=factor(y_true_ids, levels=1:K), Predicted=factor(pred_ids, levels=1:K))
  conf_matrix_df <- as.data.frame(conf_tab); names(conf_matrix_df)[3] <- "Count"
  heatmap_path_mc <- file.path(plot_dir, "confusion_matrix_multiclass_heatmap.png")
  tryCatch({
    plot_conf_matrix_mc <- ggplot(conf_matrix_df, aes(x=factor(Predicted), y=factor(Actual), fill=Count)) +
      geom_tile(color="white") + geom_text(aes(label=Count), size=3, fontface="bold") +
      scale_fill_gradient(low="white", high="red") +
      labs(title="Confusion Matrix Heatmap (Multiclass)", x="Predicted", y="Actual") +
      theme_minimal() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))
    ggsave(heatmap_path_mc, plot_conf_matrix_mc, width=6, height=5, dpi=300)
    while (!is.null(dev.list())) dev.off()
  }, error = function(e) message("❌ Multiclass heatmap failed: ", e$message))
  
  # Workbook
  wb <- createWorkbook()
  addWorksheet(wb, "Combined")
  suppressWarnings(writeData(wb, "Combined",
                             cbind(as.data.frame(X_validation),
                                   label=y_true_ids, pred=pred_ids)))
  addWorksheet(wb, "Metrics_Summary")
  ms <- data.frame(
    Class     = c(as.character(seq_len(K)), "macro avg"),
    Precision = c(Prec_k, macro_precision),
    Recall    = c(Rec_k,  macro_recall),
    F1_Score  = c(F1_k,   macro_f1),
    Accuracy  = c(rep(acc_mc, K), acc_mc)
  )
  suppressWarnings(writeData(wb, "Metrics_Summary", ms))
  if (file.exists(heatmap_path_mc)) {
    tryCatch(insertImage(wb, "Metrics_Summary", heatmap_path_mc, startRow = nrow(ms) + 6,
                         startCol = 1, width = 6, height = 4), error = function(e) {})
  }
  saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
  
  return(list(
    best_threshold   = NA_real_,
    # headline metrics
    accuracy         = acc_mc,
    precision        = macro_precision,
    recall           = macro_recall,
    f1_score         = macro_f1,
    # tuned set (n/a for multiclass)
    accuracy_tuned   = NA_real_,
    precision_tuned  = NA_real_,
    recall_tuned     = NA_real_,
    f1_tuned         = NA_real_,
    # confusion + preds + ROC (n/a)
    confusion_matrix = NULL,
    y_pred_class     = pred_ids,
    y_pred_class_tuned = NULL,
    auc = NA_real_,
    roc_curve = NULL
  ))
}
