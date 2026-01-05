df1 <- SingleRun_Pretty_Test_Metrics_500_seeds_20251025_175155
df2 <- SingleRun_Pretty_Test_Metrics_500_seeds_20251026_111537

df_all <- dplyr::bind_rows(df1, df2)

library(dplyr)
library(pROC)

auc_by_seed <- df_all %>%
  group_by(seed) %>%
  summarise(
    auc = as.numeric(pROC::roc(y_true, y_prob, quiet = TRUE)$auc),
    .groups = "drop"
  )

mean_auc <- mean(auc_by_seed$auc, na.rm = TRUE)
sd_auc   <- sd(auc_by_seed$auc, na.rm = TRUE)

cat("Mean AUC:", mean_auc, "\nSD:", sd_auc)
