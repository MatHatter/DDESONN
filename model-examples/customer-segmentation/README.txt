Customer-segmentation model upload example — DDESONN 7.3.31

1. Download customer-segmentation-model-7.3.31.rds and customer-rows-to-predict.csv.
2. Open Shiny: 4. Model / Results > Saved models & predictions.
3. Under Save or restore, use Load trusted workbench model to upload the .rds.
4. Under Predict on new rows, upload customer-rows-to-predict.csv.
5. Open the prediction preview or choose Download new predictions.

Only the .rds goes into the model-upload control. The CSV goes into New data.
You do not need to prepare data or train again after loading this bundle.

This is an actually trained multiclass model for customer segments A, B, C and D.
It was trained with DDESONN 7.3.31 on the packaged 8,068-row labeled customer CSV.
ID is excluded. Saved preprocessing encodes categories, fills missing predictors,
and standardizes using training-only statistics. The example prediction CSV has
20 rows from the packaged unlabeled customer data, with the model's 9 predictors.

Architecture: 32,16 hidden units; Adam; learning rate .003; 100 epochs; seed 42.
Best validation checkpoint restored at epoch 76.
Rows: 4,834 training / 1,618 validation / 1,616 test.
Accuracy: training 54.096%; validation 50.865%; test 52.228%.
Groups of identical raw predictors are kept together; if fitted preprocessing
collapses distinct groups, they are combined and the split/preprocessing is
recomputed before training. Exact transformed inputs have zero pairwise overlap.
This is an example model, not a promise of accuracy on your own customer data.

expected-predictions.csv records expected outputs for the provided input CSV.
evaluation.csv contains held-out metrics; train-model.R records the training recipe.
The repository source code and existing dataset files were not changed.
