###############################################################
# Hold-Out Segmentation Dataset — Column Definitions
#
# The extended multiclass segmentation hold-out dataset
# contains the following columns:
#
# run_index            : Sequential experiment iteration identifier
# seed                 : Random seed used for model initialization
# model_slot           : Ensemble model index within the run
# slot                 : Sub-model / internal slot identifier
# y_true               : True observed class label
# y_prob               : Predicted probability for target class
# y_pred               : Final predicted class after thresholding
# split                : Dataset partition (train/test/validation)
# SPLIT                : Uppercase standardized split label
# .__split__           : Internal split marker
# CLASSIFICATION_MODE  : Binary or multiclass indicator
# RUN_INDEX            : Uppercase standardized run identifier
# SEED                 : Uppercase standardized seed value
# MODEL_SLOT           : Uppercase standardized model slot
# obs_index            : Original observation index
#
# NOTE:
# To comply with CRAN size guidelines, the full hold-out .rds
# artifacts used for extended facet-slice reporting are not
# distributed in this release.
#
# Full artifacts are available under GitHub tag v7.1.7.
###############################################################

library(dplyr)
library(pROC)

# ------------------------------------------------------------------
# Combine multi-seed experimental outputs
# ------------------------------------------------------------------

df1 <- SingleRun_Pretty_Test_Metrics_500_seeds_20251025_175155
df2 <- SingleRun_Pretty_Test_Metrics_500_seeds_20251026_111537

df_all <- bind_rows(df1, df2)

# ------------------------------------------------------------------
# Compute AUC per seed
# ------------------------------------------------------------------

auc_by_seed <- df_all %>%
  group_by(seed) %>%
  summarise(
    auc = as.numeric(
      pROC::roc(
        response = y_true,
        predictor = y_prob,
        quiet = TRUE
      )$auc
    ),
    .groups = "drop"
  )

mean_auc <- mean(auc_by_seed$auc, na.rm = TRUE)
sd_auc   <- sd(auc_by_seed$auc, na.rm = TRUE)

cat("Mean AUC:", mean_auc, "\n")
cat("SD AUC:", sd_auc, "\n")
