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
# evaluate_predictions_report.R  (FULL - accuracy + accuracy_tuned + ROC/AUC)
# ================================================================
source("R/utils.R")

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
    learn_time = NULL
) {
  accuracy_plot <- match.arg(accuracy_plot)
  if (isTRUE(verbose)) cat("[Eval] Begin EvaluatePredictionsReport()\n")
  
  # ------------------------- Setup: plots dir -------------------------
  plot_dir <- file.path(getwd(), "EvaluatePredictionsReportPlots")
  if (!dir.exists(plot_dir)) dir.create(plot_dir, recursive = TRUE)
  if (isTRUE(verbose)) cat("[Eval] plot_dir:", plot_dir, "\n")
  
  # ------------------------- safety defaults -------------------------
  if (!exists("viewTables", inherits = TRUE)) viewTables <- FALSE
  if (!exists("ML_NN", inherits = TRUE))      ML_NN      <- FALSE
  if (!exists("train", inherits = TRUE))      train      <- FALSE
  if (isTRUE(verbose)) cat("[Eval] flags -> viewTables:", viewTables, "  ML_NN:", ML_NN, "  train:", train, "\n")
  
  # ------------------------- Inspect predictions/errors (optional) ---
  pred_vec   <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$learn_output),
                         error = function(e) rep(NA_real_, length.out = 0))
  err_vec    <- tryCatch(as.vector(predicted_outputAndTime$predicted_output_l2$error),
                         error = function(e) rep(NA_real_, length.out = 0))
  labels_vec <- tryCatch(as.vector(y_validation), error = function(e) rep(NA_real_, length.out = 0))
  max_points <- min(length(pred_vec), length(err_vec), length(labels_vec))
  if (isTRUE(verbose)) cat("[Eval] pred/err/labels lengths:", length(pred_vec), length(err_vec), length(labels_vec), "  max_points:", max_points, "\n")
  if (max_points > 0) {
    tryCatch({
      png(file.path(plot_dir, "pred_vs_error_scatter.png"), width = 800, height = 600)
      plot(pred_vec[seq_len(max_points)], err_vec[seq_len(max_points)],
           main = "Prediction vs. Error", xlab = "Prediction", ylab = "Error",
           col = "steelblue", pch = 16)
      abline(h = 0, col = "gray", lty = 2)
      dev.off()
      if (isTRUE(verbose)) cat("[Eval] pred_vs_error_scatter saved.\n")
    }, error = function(e) message("[Eval] Pred-vs-Error plot failed: ", conditionMessage(e)))
  }
  
  # ------------------------- weights summary (robust) -----------------
  if (isTRUE(ML_NN)) {
    w_mat <- tryCatch(as.matrix(predicted_outputAndTime$weights_record[[1]]),
                      error = function(e) matrix(NA_real_, nrow = 0, ncol = 0))
    if (length(w_mat)) {
      weights_summary <- round(rowMeans(w_mat), 5)
      if (verbose) { cat(">> Multi-layer weights summary (first layer) - rows:", nrow(w_mat), "cols:", ncol(w_mat), "\n") }
    }
  } else {
    w_raw <- tryCatch(predicted_outputAndTime$weights_record[[1]], error = function(e) numeric(0))
    if (length(w_raw)) {
      weights_summary <- round(as.numeric(w_raw), 5)
      if (verbose) { cat(">> Single-layer weights summary len:", length(weights_summary), "\n") }
    }
  }
  
  # ------------------------- Select evaluation data -------------------
  use_best <- (!is.null(all_best_val_probs) && !is.null(all_best_val_labels))
  if (use_best) {
    probs_use  <- all_best_val_probs
    labels_use <- all_best_val_labels
    if (verbose) cat("[Eval] Using BEST snapshot from training (probs/labels).\n")
  } else {
    probs_use  <- probs
    labels_use <- y_validation
    if (verbose) cat("[Eval] Using LAST-epoch predictions.\n")
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
  if (isTRUE(verbose)) cat("[Eval] Shapes L:", nrow(L), "x", ncol(L), "  P:", nrow(P), "x", ncol(P), "  n_eff:", n_eff, "\n")
  if (n_eff <= 0) stop("[EvaluatePredictionsReport] No overlapping rows between probs and labels.")
  L <- L[seq_len(n_eff), , drop = FALSE]
  P <- P[seq_len(n_eff), , drop = FALSE]
  
  # ------------------------- Mode inference ---------------------------
  infer_mode <- function(L, P, fallback = "binary") {
    if (tolower(CLASSIFICATION_MODE) %in% c("binary","multiclass","regression")) return(tolower(CLASSIFICATION_MODE))
    if (max(ncol(L), ncol(P)) > 1L) "multiclass" else fallback
  }
  mode <- infer_mode(L, P, "binary")
  if (isTRUE(verbose)) cat(sprintf("[Eval] mode=%s | n_eff=%d | ncol(L)=%d | ncol(P)=%d\n", mode, n_eff, ncol(L), ncol(P)))
  
  # ------------------------- Regression branch ------------------------
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
    
    # === Workbook (regression) ===
    wb <- createWorkbook()
    addWorksheet(wb, "Metrics_Summary")
    suppressWarnings(writeData(wb, "Metrics_Summary",
                               data.frame(Metric=c("RMSE","MAE","MAPE","R2","Correlation"),
                                          Value=c(RMSE,MAE,MAPE,R2,Corr))))
    
    # Legacy-style Rdata_Predictions sheet for regression
    addWorksheet(wb, "Rdata_Predictions")
    legacy_df <- data.frame(
      y_true = y, y_pred = yhat,
      residual = yhat - y
    )
    suppressWarnings(writeData(wb, "Rdata_Predictions", legacy_df))
    
    saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
    if (isTRUE(verbose)) cat("[Eval-Regression] Workbook saved.\n")
    
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
      roc_curve = NULL
    ))
  }
  
  # ------------------------- Binary branch (NO HELPERS) ----------------
  if (identical(mode, "binary")) {
    if (isTRUE(verbose)) cat("[Eval-Binary] Enter\n")
    
    # Labels (0/1 vector)
    y_true <- if (ncol(L) == 1L) {
      v <- as.numeric(L[,1])
      if (all(v %in% c(0,1), na.rm = TRUE)) as.integer(v) else as.integer(v >= 0.5)
    } else {
      as.integer(max.col(L, ties.method = "first") - 1L)
    }
    if (length(y_true) != n_eff) stop("[Eval-Binary] y_true length mismatch.")
    
    # Probs/logits (numeric vector)
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
    
    # If outside [0,1], assume logits; apply sigmoid
    if (any(p_pos < 0 | p_pos > 1, na.rm = TRUE)) {
      if (isTRUE(verbose)) cat("[Eval-Binary][Fixed] Detected logits; applying sigmoid to get probabilities.\n")
      p_pos <- 1 / (1 + exp(-p_pos))
      if (isTRUE(verbose)) {
        cat("[Eval-Binary][After Sigmoid] p_pos summary: min=", suppressWarnings(min(p_pos, na.rm=TRUE)),
            " max=", suppressWarnings(max(p_pos, na.rm=TRUE)),
            " mean=", suppressWarnings(mean(p_pos, na.rm=TRUE)), "\n", sep = "")
      }
    }
    
    # --------- FIXED METRICS @ 0.5 (NO HELPERS) ----------
    if (isTRUE(verbose)) cat("[Eval-Binary][Fixed] Computing metrics @ 0.5 (no helpers)\n")
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
    
    # --------- ROC / AUC ----------
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
    
    if (!is.null(roc_df) && nrow(roc_df) > 1) {
      try({
        if (isTRUE(verbose)) cat("[Eval-Binary][ROC] ggsave start\n")
        p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
          geom_line(size = 1.1) +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
          labs(title = sprintf("ROC Curve (AUC = %.4f)", auc_val), x = "FPR", y = "TPR") +
          theme_minimal()
        ggsave(filename = file.path(plot_dir, "roc_curve.png"), p_roc, width = 6, height = 4, dpi = 300)
        if (length(dev.list())) try(dev.off(), silent = TRUE)
        if (isTRUE(verbose)) cat("[Eval-Binary][ROC] ggsave done\n")
      }, silent = TRUE)
    }
    
    # --------- TUNED METRICS (override respected) ----------
    tuned <- NULL
    if (is.numeric(tuned_threshold_override) && is.finite(tuned_threshold_override)) {
      best_thr <- as.numeric(tuned_threshold_override)
      message(sprintf("[Eval-Binary] Forced tuned_threshold_override=%.4f", best_thr))
      
      # DETAILED DEBUG PRINTS
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
    
    # Extract tuned outputs
    acc_tuned    <- tuned$accuracy
    pre_tuned    <- tuned$precision
    rec_tuned    <- tuned$recall
    f1_tuned     <- tuned$f1
    best_thr     <- as.numeric(tuned$details$best_threshold)
    y_pred_tuned <- as.integer(tuned$details$y_pred_class)
    if (isTRUE(verbose)) cat("[Eval-Binary][Tuned] best_thr:", best_thr, "  acc:", acc_tuned, "  f1:", f1_tuned, "\n")
    
    # ------------------- PLOTTING (selection only) --------------------
    maybe_plot_binary <- function(mode_label, bin_preds, threshold_used, suffix) {
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
      tryCatch({
        p_conf <- ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
          geom_tile(color = "white") +
          geom_text(aes(label = Count), size = 6, fontface = "bold") +
          scale_fill_gradient(low = "white", high = "red") +
          labs(title = paste("Confusion Matrix Heatmap", toupper(mode_label))) +
          theme_minimal() +
          theme(plot.title = element_text(hjust = 0.5, face = "bold"))
        ggsave(heatmap_path, p_conf, width = 5, height = 4, dpi = 300)
        if (length(dev.list())) try(dev.off(), silent = TRUE)
        if (isTRUE(verbose)) cat("[Eval-Binary][Plot] heatmap saved:", heatmap_path, "\n")
      }, error = function(e) message("[Eval-Binary][Plot] Failed to save heatmap: ", conditionMessage(e)))
      
      df_cal <- data.frame(prob = p_pos, label = y_true) %>%
        dplyr::filter(is.finite(prob), is.finite(label)) %>%
        dplyr::mutate(prob_bin = ntile(prob, 10)) %>%
        dplyr::group_by(prob_bin) %>%
        dplyr::summarise(
          bin_mid = mean(prob, na.rm = TRUE),
          actual_rate = mean(label, na.rm = TRUE),
          .groups = "drop"
        ) %>%
        dplyr::mutate(prob_bin = factor(prob_bin))
      
      plot1_path   <- file.path(plot_dir, paste0("plot1_bar_actual_rate", suffix, ".png"))
      plot2_path   <- file.path(plot_dir, paste0("plot2_calibration_curve", suffix, ".png"))
      overlay_path <- file.path(plot_dir, paste0("plot_overlay_with_legend_below", suffix, ".png"))
      
      tryCatch({
        p1 <- ggplot(df_cal, aes(x = prob_bin, y = actual_rate)) +
          geom_col() +
          labs(title = paste("Observed Rate by Risk Bin (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile (1=low,10=high)", y = "Observed Positive Rate") +
          theme_minimal() + theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot1_path, p1, width = 6, height = 4, dpi = 300)
        if (length(dev.list())) try(dev.off(), silent = TRUE)
        if (isTRUE(verbose)) cat("[Eval-Binary][Plot] plot1 saved:", plot1_path, "\n")
      }, error = function(e) message("[Eval-Binary][Plot] plot1 failed: ", conditionMessage(e)))
      
      tryCatch({
        p2 <- ggplot(df_cal, aes(x = bin_mid, y = actual_rate)) +
          geom_line(size = 1.2) + geom_point(size = 3) +
          geom_abline(slope = 1, intercept = 0, linetype = "dashed") +
          labs(title = paste("Calibration Curve (", mode_label, ")", sep = ""),
               x = "Avg Predicted Probability", y = "Observed Rate") +
          theme_minimal() + theme(plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(plot2_path, p2, width = 6, height = 4, dpi = 300)
        if (length(dev.list())) try(dev.off(), silent = TRUE)
        if (isTRUE(verbose)) cat("[Eval-Binary][Plot] plot2 saved:", plot2_path, "\n")
      }, error = function(e) message("[Eval-Binary][Plot] plot2 failed: ", conditionMessage(e)))
      
      tryCatch({
        p3 <- ggplot(df_cal, aes(x = prob_bin)) +
          geom_col(aes(y = actual_rate)) +
          geom_point(aes(y = bin_mid), size = 3, shape = 21, stroke = 1.2) +
          labs(title = paste("Overlay: Observed vs Predicted (", mode_label, ")", sep = ""),
               x = "Predicted Risk Decile", y = "Rate", fill = NULL, color = NULL) +
          theme_minimal() + theme(legend.position = "bottom",
                                  plot.title = element_text(face = "bold", hjust = 0.5))
        ggsave(overlay_path, p3, width = 6, height = 4, dpi = 300)
        if (length(dev.list())) try(dev.off(), silent = TRUE)
        if (isTRUE(verbose)) cat("[Eval-Binary][Plot] overlay saved:", overlay_path, "\n")
      }, error = function(e) message("[Eval-Binary][Plot] overlay plot failed: ", conditionMessage(e)))
      
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
    
    # ------------------- PR Curve (legacy wanted) -------------------
    labels_numeric <- as.numeric(y_true)
    probs_numeric  <- as.numeric(p_pos)
    pr_obj <- tryCatch(
      PRROC::pr.curve(scores.class0 = probs_numeric[labels_numeric == 1],
                      scores.class1 = probs_numeric[labels_numeric == 0],
                      curve = TRUE),
      error = function(e) NULL
    )
    auprc_val <- tryCatch(round(pr_obj$auc.integral, 6), error = function(e) NA_real_)
    
    # Save PR curve plot (legacy style)
    pr_png <- file.path(plot_dir, "pr_curve.png")
    if (!is.null(pr_obj)) {
      try({
        png(pr_png, width = 800, height = 600)
        plot(pr_obj, main = "Precision-Recall Curve - Neural Network", lwd = 2)
        grid()
        dev.off()
      }, silent = TRUE)
    }
    
    # ------------------- Build legacy table assets -------------------
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
    
    # Confusion heatmap (legacy)
    conf_matrix <- matrix(c(TP, FP, FN, TN), nrow = 2, byrow = TRUE,
                          dimnames = list("Actual" = c("Positive (1)", "Negative (0)"),
                                          "Predicted" = c("Positive (1)", "Negative (0)")))
    conf_long <- as.data.frame(conf_matrix)
    conf_long <- reshape2::melt(conf_matrix)
    colnames(conf_long) <- c("Actual", "Predicted", "Count")
    
    heatmap_path_legacy <- file.path(plot_dir, "confusion_heatmap_legacy.png")
    try({
      heatmap_plot <- ggplot(conf_long, aes(x = Predicted, y = Actual, fill = Count)) +
        geom_tile() +
        geom_text(aes(label = Count), color = "white", size = 5, fontface = "bold") +
        scale_fill_gradient(low = "#4575b4", high = "#d73027") +
        theme_minimal() + ggtitle("Confusion Matrix Heatmap")
      ggsave(heatmap_path_legacy, heatmap_plot, width = 6, height = 4, dpi = 300)
      if (length(dev.list())) try(dev.off(), silent = TRUE)
    }, silent = TRUE)
    
    # Mean predictions / probabilities (legacy commentary)
    Rdata_predictions <- as.data.frame(X_validation) %>%
      mutate(
        label = labels_flat,
        Predictions = binary_preds_fixed,
        prob = probs_numeric
      )
    
    mean_0 <- suppressWarnings(mean(Rdata_predictions$prob[Rdata_predictions$label == 0], na.rm = TRUE))
    mean_1 <- suppressWarnings(mean(Rdata_predictions$prob[Rdata_predictions$label == 1], na.rm = TRUE))
    commentary_text <- if (is.finite(mean_0) && is.finite(mean_1)) {
      if (mean_0 < 0.2 && mean_1 > 0.8) {
        sprintf("Since your model produces mean %.4f for true label 0, and %.4f for true label 1, it’s making sharp, confident, and accurate predictions.", mean_0, mean_1)
      } else if (mean_0 > 0.35 && mean_1 < 0.65) {
        sprintf("Warning: predicted probabilities are close together (%.4f vs %.4f) — model may not be separating classes clearly.", mean_0, mean_1)
      } else {
        sprintf("Model separation is moderate (%.4f vs %.4f) — might benefit from output sharpening or additional tuning.", mean_0, mean_1)
      }
    } else {
      "One or both class mean probabilities are NA — likely due to class imbalance or empty subset."
    }
    commentary_df_means <- data.frame(Interpretation = commentary_text)
    
    # Legacy metrics summary (fixed @ 0.5)
    metrics_legacy <- data.frame(
      Accuracy  = acc_fixed,
      Precision = pre_fixed,
      Recall    = rec_fixed,
      F1_Score  = f1_fixed,
      TP = TP, TN = TN, FP = FP, FN = FN,
      AUC = auc_val, AUPRC = auprc_val,
      Threshold = 0.5
    )
    
    # Misclassification typing (legacy)
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
    
    # === Workbook (merge new + legacy) ===
    if (isTRUE(verbose)) cat("[Eval-Binary][WB] createWorkbook()\n")
    wb <- createWorkbook()
    
    # New-style sheets (Fixed / Tuned / ROC)
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
    
    addWorksheet(wb, "ROC")
    suppressWarnings(writeData(wb, "ROC", data.frame(AUC = auc_val, AUPRC = auprc_val)))
    roc_png <- file.path(plot_dir, "roc_curve.png")
    if (file.exists(roc_png)) {
      tryCatch(
        insertImage(wb, "ROC", roc_png, startRow = 5, startCol = 1, width = 6, height = 4),
        error = function(e) message("[Eval-Binary][WB] insertImage ROC failed: ", conditionMessage(e))
      )
    }
    if (file.exists(pr_png)) {
      tryCatch(
        insertImage(wb, "ROC", pr_png, startRow = 25, startCol = 1, width = 6, height = 4),
        error = function(e) message("[Eval-Binary][WB] insertImage PR failed: ", conditionMessage(e))
      )
    }
    
    # Insert confusion/calibration/overlay images (new)
    if (!is.null(artifacts$fixed)) {
      for (p in unlist(artifacts$fixed, use.names = FALSE)) {
        if (file.exists(p)) {
          tryCatch(
            insertImage(wb, "Fixed", p, startRow = 20, startCol = 1, width = 6, height = 4),
            error = function(e) message("[Eval-Binary][WB] insertImage (Fixed) failed: ", conditionMessage(e))
          )
        }
      }
    }
    if (!is.null(artifacts$tuned)) {
      for (p in unlist(artifacts$tuned, use.names = FALSE)) {
        if (file.exists(p)) {
          tryCatch(
            insertImage(wb, "Tuned", p, startRow = 20, startCol = 1, width = 6, height = 4),
            error = function(e) message("[Eval-Binary][WB] insertImage (Tuned) failed: ", conditionMessage(e))
          )
        }
      }
    }
    
    # ---------------- LEGACY SHEETS (restored) ----------------
    # 1) Rdata_Predictions
    addWorksheet(wb, "Rdata_Predictions")
    suppressWarnings(writeData(wb, "Rdata_Predictions", Rdata_predictions))
    
    # 2) Metrics_Summary (legacy)
    addWorksheet(wb, "Metrics_Summary")
    suppressWarnings(writeData(wb, "Metrics_Summary", metrics_legacy))
    if (file.exists(heatmap_path_legacy)) {
      tryCatch(
        insertImage(wb, "Metrics_Summary", heatmap_path_legacy, startRow = 15, startCol = 1, width = 6, height = 4),
        error = function(e) message("[Eval-Binary][WB] insertImage (legacy heatmap) failed: ", conditionMessage(e))
      )
    }
    
    # 3) Prediction_Means
    addWorksheet(wb, "Prediction_Means")
    suppressWarnings(writeData(wb, "Prediction_Means",
                               data.frame(Mean_Prob_Label_0 = mean_0, Mean_Prob_Label_1 = mean_1)))
    suppressWarnings(writeData(wb, "Prediction_Means", commentary_df_means, startRow = 5))
    
    # 4) Misclassified + 5) Misclass_Summary (+ legacy plots if available)
    addWorksheet(wb, "Misclassified")
    suppressWarnings(writeData(wb, "Misclassified", misclassified_sorted))
    
    addWorksheet(wb, "Misclass_Summary")
    if (nrow(misclassified_sorted)) {
      # Try to guess a few common columns, but stay robust if absent
      known_cols <- intersect(c("age","serum_creatinine","ejection_fraction","time"), names(misclassified_sorted))
      if (length(known_cols)) {
        summary_by_type <- misclassified_sorted %>%
          group_by(Type) %>%
          summarise(across(all_of(known_cols), \(x) mean(x, na.rm = TRUE)))
      } else {
        summary_by_type <- data.frame()
      }
      
      suppressWarnings(writeData(wb, "Misclass_Summary", summary_by_type))
      
      # Optional legacy plots if they exist from previous runs; otherwise skip silently
      legacy_mis_heat <- file.path(getwd(), "misclassification_heatmap.png")
      legacy_box_sc   <- file.path(getwd(), "boxplot_serum_creatinine.png")
      if (file.exists(legacy_mis_heat)) {
        tryCatch(insertImage(wb, "Misclass_Summary", legacy_mis_heat, startRow = 10, startCol = 1, width = 6, height = 4),
                 error = function(e) message("[Eval-Binary][WB] legacy misclass heatmap insert failed: ", conditionMessage(e)))
      }
      if (file.exists(legacy_box_sc)) {
        tryCatch(insertImage(wb, "Misclass_Summary", legacy_box_sc, startRow = 25, startCol = 1, width = 6, height = 4),
                 error = function(e) message("[Eval-Binary][WB] legacy boxplot insert failed: ", conditionMessage(e)))
      }
    }
    
    # 6) (Optional) Metrics_Library sheet using your prebuilt functions if available
    safe_call <- function(fn, ...) {
      f <- get0(fn, ifnotfound = NULL)
      if (is.function(f)) {
        tryCatch(as.numeric(f(SONN, Rdata, labels, CLASSIFICATION_MODE, probs_use, ...)),
                 error = function(e) NA_real_)
      } else NA_real_
    }
    addWorksheet(wb, "Metrics_Library")
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
    suppressWarnings(writeData(wb, "Metrics_Library", t(lib_metrics))) # vertical list
    
    # === SAFE WRITE + DEVICE CLEANUP ===
    tryCatch({
      if (length(dev.list())) {
        if (isTRUE(verbose)) cat("[Eval-Binary][WB] Closing open graphics devices...\n")
        invisible(lapply(dev.list(), function(x) try(dev.off(), silent = TRUE)))
      }
      if (file.exists("Rdata_predictions.xlsx")) {
        if (isTRUE(verbose)) cat("[Eval-Binary][WB] Removing locked workbook...\n")
        file.remove("Rdata_predictions.xlsx")
      }
      if (isTRUE(verbose)) cat("[Eval-Binary][WB] saveWorkbook() begin\n")
      saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
      if (isTRUE(verbose)) cat("[Eval-Binary][WB] saveWorkbook() done\n")
    }, error = function(e) {
      message("[Eval-Binary][WB] Workbook save failed: ", conditionMessage(e))
    })
    
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
      roc_curve = roc_df
    ))
  } # end binary
  
  # ------------------------- Multiclass branch ------------------------
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
  tryCatch({
    p_mc <- ggplot(conf_matrix_df, aes(x=factor(Predicted), y=factor(Actual), fill=Count)) +
      geom_tile(color="white") + geom_text(aes(label=Count), size=3, fontface="bold") +
      scale_fill_gradient(low="white", high="red") +
      labs(title="Confusion Matrix Heatmap (Multiclass)", x="Predicted", y="Actual") +
      theme_minimal() + theme(plot.title = element_text(hjust = 0.5, face = "bold"))
    ggsave(heatmap_path_mc, p_mc, width=6, height=5, dpi=300)
    if (length(dev.list())) try(dev.off(), silent = TRUE)
    if (isTRUE(verbose)) cat("[Eval-Multiclass] heatmap saved:", heatmap_path_mc, "\n")
  }, error = function(e) message("[Eval-Multiclass] heatmap failed: ", conditionMessage(e)))
  
  cat("=== diagnostics for evaluate_predictions_report ===\n")
  cat("nrow(X_validation):", NROW(X_validation), "\n")
  cat("length(y_true_ids):", length(y_true_ids), "\n")
  cat("length(pred_ids):", length(pred_ids), "\n\n")
  
  str(list(
    X_validation = head(X_validation, 3),
    y_true_ids   = head(y_true_ids, 10),
    pred_ids     = head(pred_ids, 10)
  ))
  
  
  wb <- createWorkbook()
  addWorksheet(wb, "Combined")
  
  # helper (define once; if you place it earlier in the file, remove this local copy)
  combine_for_report <- function(X, y, p, verbose = TRUE) {
    nX <- NROW(X); ny <- length(y); np <- length(p)
    if (is.matrix(y) && ncol(y) == 1L) y <- as.vector(y)
    if (is.matrix(p) && ncol(p) == 1L) p <- as.vector(p)
    
    rnx <- rownames(X); ny_names <- names(y); np_names <- names(p)
    
    # Try name alignment first (future-proof)
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
    
    # Fallback: truncate to common length so the report still writes
    m <- min(nX, ny, np)
    if (verbose && (nX != m || ny != m || np != m)) {
      message(sprintf(
        "[EvaluatePredictionsReport] Row mismatch: X=%d, y=%d, p=%d → truncating to %d rows for 'Combined'.",
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
  
  combined_df <- combine_for_report(X_validation, y_true_ids, pred_ids, verbose = TRUE)
  suppressWarnings(writeData(wb, "Combined", combined_df))
  
  
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
    tryCatch(
      insertImage(wb, "Metrics_Summary", heatmap_path_mc, startRow = nrow(ms) + 6,
                  startCol = 1, width = 6, height = 4),
      error = function(e) message("[Eval-Multiclass] insertImage failed: ", conditionMessage(e))
    )
  }
  
  # Legacy-friendly drop-in: Predictions sheet for multiclass as well
  addWorksheet(wb, "Rdata_Predictions")
  
  predictions_df <- combine_for_report(X_validation, y_true_ids, pred_ids, verbose = FALSE)
  
  # Optional: include a stable row identifier as the first column
  rid <- rownames(predictions_df)
  if (is.null(rid)) rid <- seq_len(nrow(predictions_df))
  predictions_df <- cbind(RowID = rid, predictions_df)
  
  suppressWarnings(writeData(wb, "Rdata_Predictions", predictions_df))
  
  
  saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
  if (isTRUE(verbose)) cat("[Eval-Multiclass] Workbook saved. RETURN\n")
  
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
    roc_curve = NULL
  ))
}
