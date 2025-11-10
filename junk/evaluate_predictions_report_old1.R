source("utils/utils.R")
# install.packages("pROC")
# install.packages("ggplotify")
library(pROC)
library(ggplotify)
EvaluatePredictionsReport <- function(X_validation, y_validation, probs, predicted_outputAndTime, threshold_function, best_val_probs, best_val_labels, verbose) {
# ------------------------------------------------------------------------------
# evaluate_predictions_report.R
# ------------------------------------------------------------------------------
# Purpose: Generate a full report of prediction-based metrics, plots, and
# commentary based on model output.
#
# This file consumes model predictions and label data, and summarizes:
# - Threshold-tuned classification metrics (accuracy, F1, etc.)
# - Confusion matrix heatmap
# - Prediction probability separation
# - Calibration curves and commentary
#
# This is NOT a core diagnostic metric function. See calculatePerformance.R
# and calculateRelevance.R for those atomic helpers. (right now they are helper functions; will designate into 2 separate .R files later.)
# ------------------------------------------------------------------------------


# === Extract vectors ===
pred_vec   <- as.vector(predicted_outputAndTime$predicted_output_l2$learn_output)
err_vec    <- as.vector(predicted_outputAndTime$predicted_output_l2$error)
labels_vec  <- as.vector(y_validation)

# === Ensure equal lengths ===
max_points <- min(length(pred_vec), length(err_vec), length(labels_vec))


# === Plot: Prediction vs. Error ===
plot(pred_vec, err_vec,
     main = "Prediction vs. Error",
     xlab = "Prediction",
     ylab = "Error",
     col = "steelblue", pch = 16)
abline(h = 0, col = "gray", lty = 2)

# === Combined Commentary Table ===
comment_error     <- character(max_points)
comment_alignment <- character(max_points)

for (i in 1:max_points) {
  # Error commentary
  if (labels_vec[i] == 1) {
    comment_error[i] <- paste0("Overpredicted by ", round(pred_vec[i] - 1, 3))
  } else {
    comment_error[i] <- paste0("Underpredicted by ", round(pred_vec[i], 3))
  }
  
  # Alignment commentary
  if (abs(pred_vec[i] - labels_vec[i]) > 0.5) {
    comment_alignment[i] <- paste0("Misaligned (", round(pred_vec[i], 2), " vs ", labels_vec[i], ")")
  } else {
    comment_alignment[i] <- "Aligned prediction"
  }
}

df_sample_commentary <- data.frame(
  Row = 1:max_points,
  Prediction = round(pred_vec[1:max_points], 5),
  Label = labels_vec[1:max_points],
  Error = round(err_vec[1:max_points], 5),
  Comment_Error = comment_error,
  Alignment_Comment = comment_alignment,
  stringsAsFactors = FALSE
)

# === View Combined Table ===
if(viewTables){
View(df_sample_commentary)
}


# === Extract first-layer weights from weights_record ===
# (replace index as needed for deeper analysis)
if (ML_NN) {
  # Multi-layer network: standard weight matrix, safe for rowMeans
  weights_mat <- predicted_outputAndTime$weights_record[[1]]
  weights_summary <- round(rowMeans(as.matrix(weights_mat)), 5)
  cat(">> Multi-layer weights summary (first layer):\n")
  print(weights_summary)
} else {
  # Single-layer network: handle vector or scalar gracefully
  w_raw <- predicted_outputAndTime$weights_record[[1]]
  w_mat <- matrix(as.numeric(w_raw), ncol = 1L)
  weights_summary <- round(as.numeric(w_mat), 5)
  cat(">> Single-layer weights summary:\n")
  print(weights_summary)
}


# Make sure length aligns with number of features or pad/trim
max_len <- min(length(pred_vec), length(err_vec), length(labels_vec), length(weights_summary), 15)

# Create the inspection table
df_inspect <- data.frame(
  Row = 1:max_len,
  Prediction = round(pred_vec[1:max_len], 5),
  Label = labels_vec[1:max_len],
  Error = round(err_vec[1:max_len], 5),
  Avg_Input_Weight = weights_summary[1:max_len],
  comment_observation = character(max_len),
  comment_suggestion = character(max_len),
  stringsAsFactors = FALSE
)

# Add comments based on logic
for (j in 1:max_len) {
  p <- df_inspect$Prediction[j]
  l <- df_inspect$Label[j]
  e <- df_inspect$Error[j]
  w <- df_inspect$Avg_Input_Weight[j]
  
  if (l == 0) {
    if (abs(e - p) < 1e-4) {
      df_inspect$comment_observation[j] <- "Label = 0, error = prediction"
    } else {
      df_inspect$comment_observation[j] <- "Label = 0, error ≠ prediction"
      df_inspect$comment_suggestion[j] <- "Review logic for label = 0"
    }
  } else if (l == 1) {
    expected_error <- (p - 1)  # assuming sample_weights were 1
    if (abs(e - expected_error) < 1e-4) {
      df_inspect$comment_observation[j] <- "Label = 1, error as expected"
    } else {
      df_inspect$comment_observation[j] <- "Label = 1, error mismatch"
      df_inspect$comment_suggestion[j] <- "Inspect how error is scaled"
    }
  } else {
    df_inspect$comment_observation[j] <- "Unknown label"
    df_inspect$comment_suggestion[j] <- "Check label values"
  }
}

# Show the result
if(viewTables){
View(df_inspect)
}



calculate_accuracy <- function(predictions, actual_labels) {
  correct_predictions <- sum(predictions == actual_labels)
  accuracy <- (correct_predictions / length(actual_labels)) * 100
  return(accuracy)
}

cat("Str in predicted_outputAndTime:\n")
print(str(predicted_outputAndTime))

cat("str of predicted_output:\n")
print(str(predicted_outputAndTime$predicted_output))

if (!is.null(predicted_outputAndTime$predicted_output_l2)) {
  cat("str predicted_output_l2$predicted_output:\n")
  print(str(predicted_outputAndTime$predicted_output_l2$predicted_output))
}

# probs <- predicted_outputAndTime$predicted_output_l2$predicted_output
# binary_preds <- probs
# binary_preds <- ifelse(probs >= 0.5, 1, 0)
# binary_preds <- ifelse(probs >= 0.1, 1, 0)
if(train){
  probs <- best_val_probs
  labels <- best_val_labels
}

# === Ensure numeric vectors ===
probs  <- as.numeric(probs)
labels <- as.numeric(labels)

# === Auto-tune threshold based on F1 ===
# threshold_result <- tune_threshold(probs, labels)
threshold_result <- do.call(threshold_function, list(probs, labels))

best_threshold <- threshold_result$best_threshold
# print(paste("Best F1 threshold:", best_threshold))
if (verbose) {
print(paste("Best threshold (accuracy-optimized):", best_threshold))
}
# === Binary prediction using best threshold ===
predict_with_threshold <- function(probs, threshold) {
  ifelse(probs >= threshold, 1, 0)
}

binary_preds <- predict_with_threshold(probs, best_threshold)

# === Accuracy Evaluation ===
labels_flat <- as.vector(labels)
accuracy <- calculate_accuracy(binary_preds, labels_flat)
if (verbose) {
print(paste("Accuracy:", accuracy))
}


# === Precision / Recall / F1 ===
metrics <- evaluate_classification_metrics(binary_preds, labels_flat)
print(metrics)

# === Accuracy & Confusion Matrix ===
TP <- sum(binary_preds == 1 & labels_flat == 1)
TN <- sum(binary_preds == 0 & labels_flat == 0)
FP <- sum(binary_preds == 1 & labels_flat == 0)
FN <- sum(binary_preds == 0 & labels_flat == 1)
accuracy <- (TP + TN) / length(labels_flat)

# === Confusion Matrix DataFrame ===
conf_matrix_df <- data.frame(
  Actual = c("0", "0", "1", "1"),
  Predicted = c("0", "1", "0", "1"),
  Count = c(TN, FP, FN, TP)
)

# === Heatmap Path Plot ===
tryCatch({
  heatmap_path <- "confusion_matrix_heatmap.png"
  plot_conf_matrix <- ggplot(conf_matrix_df, aes(x = Predicted, y = Actual, fill = Count)) +
    geom_tile(color = "white") +
    geom_text(aes(label = Count), size = 6, fontface = "bold") +
    scale_fill_gradient(low = "white", high = "red") +
    labs(title = "Confusion Matrix Heatmap") +
    theme_minimal() +
    theme(plot.title = element_text(hjust = 0.5, face = "bold"))
  ggsave(heatmap_path, plot_conf_matrix, width = 5, height = 4)
  while (!is.null(dev.list())) dev.off()
  print(plot_conf_matrix)
}, error = function(e) {
  message("❌ Failed to generate confusion matrix heatmap: ", e$message)
})

# === Combine Full Rdata, Labels, Predictions ===
Rdata_df <- as.data.frame(X_validation)
labels_flat <- as.numeric(labels)
binary_preds <- ifelse(probs >= best_threshold, 1, 0)

Rdata_with_labels <- cbind(Rdata_df, Label = labels_flat)
Rdata_predictions <- Rdata_with_labels %>%
  mutate(
    Predictions = binary_preds,
    Predicted_Prob = as.vector(probs)
  )

# === Mean Predicted Probability by Class ===
prediction_means <- Rdata_predictions %>%
  group_by(Label) %>%
  summarise(
    MeanProbability = mean(Predicted_Prob, na.rm = TRUE),
    StdDev = sd(Predicted_Prob, na.rm = TRUE),
    Count = n()
  ) %>%
  rename(Class = Label)

# Extract means for use in commentary
mean_0 <- prediction_means$MeanProbability[prediction_means$Class == 0]
mean_1 <- prediction_means$MeanProbability[prediction_means$Class == 1]

# === Interpret Prediction Separation Commentary ===
if (!is.na(mean_0) && !is.na(mean_1)) {
  commentary_text <- if (mean_0 < 0.2 && mean_1 > 0.8) {
    paste0("✅ Since your model produces mean ", round(mean_0, 4), " for true label 0, and ",
           round(mean_1, 4), " for true label 1, it’s making sharp, confident, and accurate predictions.")
  } else if (mean_0 > 0.35 && mean_1 < 0.65) {
    paste0("⚠️ Warning: predicted probabilities are close together (", round(mean_0, 4),
           " vs ", round(mean_1, 4), ") — model may not be separating classes clearly.")
  } else {
    paste0("ℹ️ Model separation is moderate (", round(mean_0, 4), " vs ", round(mean_1, 4),
           ") — might benefit from output sharpening or additional tuning.")
  }
} else {
  commentary_text <- "❌ One or both class mean probabilities are NA — likely due to lack of class balance or empty subset."
}
commentary_df_means <- data.frame(Interpretation = commentary_text)
print(commentary_df_means)

# === Identify Misclassified Samples ===
wrong <- which(binary_preds != labels_flat)
misclassified <- cbind(
  predicted_prob = probs[wrong],
  predicted_label = binary_preds[wrong],
  actual_label = labels_flat[wrong],
  as.data.frame(Rdata)[wrong, , drop = FALSE]
)

# === Sorted Misclassified Samples (by confidence error) ===
misclassified_sorted <- misclassified %>%
  mutate(
    error = abs(predicted_prob - actual_label),
    Type = ifelse(predicted_label == 1 & actual_label == 0, "False Positive", "False Negative")
  ) %>%
  arrange(desc(error))

# === Summary by Misclassification Type (if available) ===
if (nrow(misclassified_sorted) > 0) {
  # cat("\n📌 Top Misclassified Samples (sorted by confidence error):\n")
  # print(head(misclassified_sorted, 10))
  
  summary_by_type <- misclassified_sorted %>%
    group_by(Type) %>%
    summarise(across(c(age, serum_creatinine, ejection_fraction, time), ~mean(.x, na.rm = TRUE)))
  
  # Save to RDS for viewing later
  saveRDS(summary_by_type, file = "summary_by_type.rds")
  
  cat("\n💾 summary_by_type saved as summary_by_type.rds\n")
} else {
  message("⚠️ No misclassified samples to summarize.")
}


# === Per-Class Support ===
support0 <- sum(labels_flat == 0)
support1 <- sum(labels_flat == 1)
total    <- support0 + support1

# === Per-Class Precision / Recall / F1 ===
precision0 <- if ((TN + FN) > 0) TN / (TN + FN) else 0
recall0    <- if ((TN + FP) > 0) TN / (TN + FP) else 0
f1_0       <- if ((precision0 + recall0) > 0) 2 * precision0 * recall0 / (precision0 + recall0) else 0

precision1 <- if ((TP + FP) > 0) TP / (TP + FP) else 0
recall1    <- if ((TP + FN) > 0) TP / (TP + FN) else 0
f1_1       <- if ((precision1 + recall1) > 0) 2 * precision1 * recall1 / (precision1 + recall1) else 0

# === Macro / Weighted Averages ===
macro_precision <- mean(c(precision0, precision1))
macro_recall    <- mean(c(recall0, recall1))
macro_f1        <- mean(c(f1_0, f1_1))

weighted_precision <- (support0 * precision0 + support1 * precision1) / total
weighted_recall    <- (support0 * recall0 + support1 * recall1) / total
weighted_f1        <- (support0 * f1_0 + support1 * f1_1) / total

# === Final Metrics Summary ===
metrics_summary <- data.frame(
  Class      = c("0", "1", "macro avg", "weighted avg"),
  Precision  = c(precision0, precision1, macro_precision, weighted_precision),
  Recall     = c(recall0, recall1, macro_recall, weighted_recall),
  F1_Score   = c(f1_0, f1_1, macro_f1, weighted_f1),
  Support    = c(support0, support1, total, total),
  Accuracy   = c(rep(accuracy, 4)),
  TP         = c(rep(TP, 4)),
  TN         = c(rep(TN, 4)),
  FP         = c(rep(FP, 4)),
  FN         = c(rep(FN, 4))
)

# === Save Metrics Table to PNG ===
tryCatch({
  png("metrics_summary.png", width = 1000, height = 400)
  grid.table(metrics_summary)
  dev.off()
}, error = function(e) {
  message("❌ Failed to save metrics_summary.png: ", e$message)
})




# === Required Libraries ===
library(pROC)
library(PRROC)
library(ggplotify)
library(ggplot2)

# === DEBUG: Input Check ===
cat("==== DEBUG: ROC/PR Input Check ====\n")
cat("Length of labels_flat:", length(labels_flat), "\n")
cat("Length of probs:", length(probs), "\n")
cat("labels_flat unique values:", paste(unique(labels_flat), collapse = ", "), "\n")
cat("probs summary:\n")
print(summary(probs))

# === Flatten and Coerce ===
labels_numeric <- suppressWarnings(as.numeric(labels_flat))
probs_numeric <- suppressWarnings(as.numeric(probs))

# === ROC Curve ===
tryCatch({
  if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
    roc_obj <- pROC::roc(response = labels_numeric,
                         predictor = probs_numeric,
                         quiet = TRUE)
    auc_value <- pROC::auc(roc_obj)
    auc_val_text <- paste("AUC =", round(as.numeric(auc_value), 4))
    cat("✅ AUC (ROC):", auc_val_text, "\n")
    
    roc_curve_plot <- ggplotify::as.ggplot(~{
      plot(roc_obj,
           col = "#1f77b4", lwd = 2,
           main = "ROC Curve - Neural Network")
      abline(a = 0, b = 1, lty = 2, col = "gray")
      text(0.6, 0.2, labels = auc_val_text, cex = 1.2)
    })
    ggsave("roc_curve.png", plot = roc_curve_plot, width = 6, height = 4, dpi = 300)
    while (!is.null(dev.list())) dev.off()
    print(roc_curve_plot)
  } else {
    cat("⚠️ ROC skipped: not enough unique label values or mismatched lengths.\n")
  }
}, error = function(e) {
  message("⚠️ ROC block failed gracefully: ", e$message)
})

# === PR Curve ===
tryCatch({
  if (length(unique(labels_numeric)) >= 2 && length(labels_numeric) == length(probs_numeric)) {
    pr_obj <- PRROC::pr.curve(
      scores.class0 = probs_numeric[labels_numeric == 1],
      scores.class1 = probs_numeric[labels_numeric == 0],
      curve = TRUE
    )
    cat("✅ AUPRC (PR Curve):", round(pr_obj$auc.integral, 4), "\n")
    
    pr_curve_plot <- ggplotify::as.ggplot(~{
      plot(pr_obj,
           main = "Precision-Recall Curve - Neural Network",
           col = "#d62728",
           lwd = 2)
    })
    ggsave("pr_curve.png", plot = pr_curve_plot, width = 6, height = 4, dpi = 300)
    while (!is.null(dev.list())) dev.off()
    print(pr_curve_plot)
  } else {
    cat("⚠️ PR skipped: not enough unique label values or mismatched lengths.\n")
  }
}, error = function(e) {
  message("⚠️ PR block failed gracefully: ", e$message)
})








library(ggplot2)
library(dplyr)
library(tidyr)
library(gridExtra)
library(reshape2)

# === Prep Predictions ===
Rdata_predictions <- Rdata_predictions %>%
  rename(prob = Predicted_Prob, label = Label) %>%
  mutate(
    prob = as.numeric(prob),
    label = as.numeric(label),
    label = ifelse(label >= 1, 1, 0)
  ) %>%
  filter(!is.na(prob), !is.na(label)) %>%
  mutate(prob_bin = ntile(prob, 10)) %>%
  group_by(prob_bin) %>%
  mutate(bin_mid = mean(prob, na.rm = TRUE)) %>%
  ungroup()

# === Bin Summary ===
bin_summary <- Rdata_predictions %>%
  group_by(prob_bin, bin_mid) %>%
  summarise(actual_death_rate = mean(label, na.rm = TRUE), .groups = "drop") %>%
  filter(
    !is.na(bin_mid),
    !is.na(actual_death_rate),
    is.finite(bin_mid),
    is.finite(actual_death_rate)
  ) %>%
  mutate(prob_bin = factor(prob_bin))

# === Plot 1: Bar of Actual Death Rate ===
tryCatch({
  plot1 <- ggplot(bin_summary, aes(x = prob_bin, y = actual_death_rate)) +
    geom_col(fill = "steelblue") +
    labs(
      title = "Observed Death Rate by Risk Bin",
      x = "Predicted Risk Decile Bin (1 = lowest, 10 = highest)",
      y = "Actual Death Rate"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  ggsave("plot1_bar_actual_death_rate.png", plot1, width = 6, height = 4)
  while (!is.null(dev.list())) dev.off()
  print(plot1)
}, error = function(e) {
  message("❌ Failed to generate plot1: ", e$message)
})

# === Plot 2: Calibration Curve ===
tryCatch({
  plot2 <- ggplot(bin_summary, aes(x = bin_mid, y = actual_death_rate)) +
    geom_line(color = "blue", size = 1.2) +
    geom_point(size = 3, color = "black") +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
    labs(
      title = "Calibration Curve: Predicted vs Actual",
      x = "Average Predicted Probability (bin_mid)",
      y = "Observed Death Rate"
    ) +
    theme_minimal() +
    theme(plot.title = element_text(face = "bold", hjust = 0.5))
  ggsave("plot2_calibration_curve.png", plot2, width = 6, height = 4)
  while (!is.null(dev.list())) dev.off()
  print(plot2)
}, error = function(e) {
  message("❌ Failed to generate plot2: ", e$message)
})

# === Plot 3: Overlay ===
tryCatch({
  plot_overlay <- ggplot(bin_summary, aes(x = prob_bin)) +
    geom_col(aes(y = actual_death_rate, fill = "Actual Death Rate")) +
    geom_point(aes(y = bin_mid, color = "Avg Predicted Prob"),
               size = 3, shape = 21, stroke = 1.2, fill = "white") +
    scale_fill_manual(values = c("Actual Death Rate" = "steelblue")) +
    scale_color_manual(values = c("Avg Predicted Prob" = "black")) +
    labs(
      title = "Overlay: Actual vs Predicted Death Rate",
      x = "Predicted Risk Decile Bin",
      y = "Rate",
      fill = NULL,
      color = NULL
    ) +
    theme_minimal() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", hjust = 0.5)
    )
  ggsave("plot_overlay_with_legend_below.png", plot_overlay, width = 6, height = 4)
  while (!is.null(dev.list())) dev.off()
  print(plot_overlay)
}, error = function(e) {
  message("❌ Failed to generate overlay plot: ", e$message)
})

# === Calibration Table PNG ===
tryCatch({
  calibration_table <- bin_summary %>%
    mutate(
      difference = round(actual_death_rate - bin_mid, 4),
      calibration = case_when(
        abs(difference) < 0.02 ~ "✅ Well Calibrated",
        difference > 0 ~ "⚠️ Underconfident",
        TRUE ~ "⚠️ Overconfident"
      )
    ) %>%
    select(
      Bin = prob_bin,
      `Avg Predicted Prob` = bin_mid,
      `Actual Death Rate` = actual_death_rate,
      `Difference (Actual - Predicted)` = difference,
      `Calibration Quality` = calibration
    )
  table_plot <- tableGrob(calibration_table, rows = NULL)
  ggsave("calibration_table.png", table_plot, width = 8, height = 4, dpi = 300)
}, error = function(e) {
  message("❌ Failed to generate calibration table PNG: ", e$message)
})

# === Commentary ===
tryCatch({
  commentary_text <- c()
  
  if (!exists("accuracy") || !is.numeric(accuracy)) accuracy <- NA
  if (!exists("metrics") || !is.list(metrics)) metrics <- list()
  if (is.null(metrics$F1)) metrics$F1 <- NA
  if (is.null(metrics$precision)) metrics$precision <- NA
  if (is.null(metrics$recall)) metrics$recall <- NA
  
  acc <- round(accuracy, 4)
  f1 <- round(metrics$F1, 4)
  prec <- round(metrics$precision, 4)
  rec <- round(metrics$recall, 4)
  
  if (!is.na(acc)) {
    commentary_text <- c(commentary_text, paste("Accuracy:", acc))
  }
  
  if (!is.na(f1)) {
    commentary_text <- c(commentary_text, paste("F1 Score:", f1))
  }
  
  if (!is.na(prec) && !is.na(rec)) {
    if (prec > rec + 0.05) {
      commentary_text <- c(commentary_text, "⚠️ Precision-dominant: Model is conservative and may miss true positives.")
    } else if (rec > prec + 0.05) {
      commentary_text <- c(commentary_text, "⚠️ Recall-dominant: Model identifies positives aggressively but may raise false alarms.")
    } else {
      commentary_text <- c(commentary_text, "✅ Balanced precision and recall.")
    }
  }
  
  commentary_df_metrics <- data.frame(Interpretation = commentary_text)
  
}, error = function(e) {
  message("❌ Failed to generate commentary_df_metrics: ", e$message)
  commentary_df_metrics <- data.frame(Interpretation = "⚠️ Metric interpretation unavailable.")
})




# === Write All to Excel ===
wb <- createWorkbook()

addWorksheet(wb, "Rdata_Predictions")
writeData(wb, "Rdata_Predictions", Rdata_predictions)

addWorksheet(wb, "Metrics_Summary")
writeData(wb, "Metrics_Summary", metrics_summary)
writeData(wb, "Metrics_Summary", commentary_df_metrics, startRow = 10)
writeData(wb, "Metrics_Summary", conf_matrix_df, startRow = 13, rowNames = TRUE)
insertImage(wb, "Metrics_Summary", heatmap_path, startRow = 18, startCol = 1, width = 6, height = 4)

# === Commentary on Prediction Means ===
if (!exists("commentary_df_means")) {
  if (exists("prediction_means") &&
      is.data.frame(prediction_means) &&
      "prob" %in% colnames(prediction_means)) {
    
    prob_values <- suppressWarnings(as.numeric(prediction_means$prob))
    mean_prob <- mean(prob_values, na.rm = TRUE)
    
    if (!is.na(mean_prob)) {
      if (mean_prob > 0.7) {
        commentary_text_means <- "The model shows a strong tendency to predict the positive class."
      } else if (mean_prob < 0.3) {
        commentary_text_means <- "The model shows a strong tendency to predict the negative class."
      } else {
        commentary_text_means <- "The model has balanced prediction tendencies across classes."
      }
    } else {
      commentary_text_means <- "Could not calculate mean prediction probability due to NA or non-numeric values."
    }
    
  } else {
    commentary_text_means <- "Prediction means data is missing or does not include a 'prob' column."
  }
  
  commentary_df_means <- data.frame(Commentary = commentary_text_means)
}



addWorksheet(wb, "Prediction_Means")
writeData(wb, "Prediction_Means", prediction_means)
writeData(wb, "Prediction_Means", commentary_df_means, startRow = 5)

addWorksheet(wb, "Misclassified")
writeData(wb, "Misclassified", misclassified_sorted)

addWorksheet(wb, "Misclass_Summary")
writeData(wb, "Misclass_Summary", summary_by_type)
insertImage(wb, "Misclass_Summary", "misclassification_heatmap.png", startRow = 10, startCol = 1, width = 6, height = 4)
insertImage(wb, "Misclass_Summary", "boxplot_serum_creatinine.png", startRow = 25, startCol = 1, width = 6, height = 4)

saveWorkbook(wb, "Rdata_predictions.xlsx", overwrite = TRUE)
cat("DEBUG >>> Final best_threshold before return:", best_threshold, "\n")

return(list(
  best_threshold = best_threshold,
  accuracy = accuracy,
  metrics = metrics,
  misclassified = misclassified
))

}
