suppressPackageStartupMessages({
  library(dplyr)
  library(tibble)
})

## ------------------------------------------------------------------
## 0. we assume these objects already exist in your session:
##    - SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025_175155
##    - SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026_111537
##    - SingleRun_Test_Metrics_500_seeds_20251025_175155
##    - SingleRun_Test_Metrics_500_seeds_20251026_111537
##
## If not, you'd do readRDS() here first.
## ------------------------------------------------------------------

## 1. combine the 2 train frames (now ~1000 rows total if each is 500 seeds)
train_all <- bind_rows(
  SingleRun_Train_Acc_Val_Metrics_500_seeds_20251025_175155,
  SingleRun_Train_Acc_Val_Metrics_500_seeds_20251026_111537
)

## 2. combine the 2 test frames (same idea)
test_all <- bind_rows(
  SingleRun_Test_Metrics_500_seeds_20251025_175155,
  SingleRun_Test_Metrics_500_seeds_20251026_111537
)

## 3. per-seed metrics from train_all
##    we keep the row with the highest best_val_acc for each seed
train_seed <- train_all %>%
  group_by(seed) %>%
  slice_max(order_by = best_val_acc, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed       = seed,
    train_acc  = best_train_acc,
    val_acc    = best_val_acc
  )

## 4. per-seed metrics from test_all
##    we keep the row with the highest accuracy for each seed
test_seed <- test_all %>%
  group_by(seed) %>%
  slice_max(order_by = accuracy, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  transmute(
    seed      = seed,
    test_acc  = accuracy
  )

## 5. merge train/val with test on seed
merged <- train_seed %>%
  inner_join(test_seed, by = "seed") %>%
  arrange(seed)

## 6. helper to compute stats for a numeric vector
summarize_column <- function(x) {
  x_sorted <- sort(x)
  n <- length(x)
  
  pct <- function(p) {
    # p as 0.25, 0.5, 0.75
    if (n == 1) return(x_sorted[1])
    stats::quantile(x, probs = p, names = FALSE, type = 7)
  }
  
  data.frame(
    count = n,
    mean  = mean(x),
    std   = ifelse(n > 1, sd(x), NA_real_),
    min   = min(x),
    `25%` = pct(0.25),
    `50%` = pct(0.50),
    `75%` = pct(0.75),
    max   = max(x),
    check.names = FALSE
  )
}

## 7. build summaries for train_acc / val_acc / test_acc
summary_train <- summarize_column(merged$train_acc)
summary_val   <- summarize_column(merged$val_acc)
summary_test  <- summarize_column(merged$test_acc)

## 8. stitch them into one table (rows = stats, cols = train_acc/val_acc/test_acc)
summary_all <- data.frame(
  rownames = c("count","mean","std","min","25%","50%","75%","max"),
  train_acc = c(summary_train$count,
                summary_train$mean,
                summary_train$std,
                summary_train$min,
                summary_train$`25%`,
                summary_train$`50%`,
                summary_train$`75%`,
                summary_train$max),
  val_acc = c(summary_val$count,
              summary_val$mean,
              summary_val$std,
              summary_val$min,
              summary_val$`25%`,
              summary_val$`50%`,
              summary_val$`75%`,
              summary_val$max),
  test_acc = c(summary_test$count,
               summary_test$mean,
               summary_test$std,
               summary_test$min,
               summary_test$`25%`,
               summary_test$`50%`,
               summary_test$`75%`,
               summary_test$max),
  check.names = FALSE
)

row.names(summary_all) <- summary_all$rownames
summary_all$rownames <- NULL

## 9. pretty print:
##    - round numeric columns to 4 decimals
round4 <- function(x) round(x, 4)

pretty_summary <- as.data.frame(
  lapply(summary_all, round4),
  row.names = row.names(summary_all)
)

## 10. output
cat("=== Summary across 1000 seeds ===\n\n")
print(pretty_summary, row.names = TRUE)

cat("\n=== Per-seed table ===\n\n")
print(as_tibble(merged))
