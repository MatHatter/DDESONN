# Customer-model evaluation example

Use `customer-segmentation-original-test-example.csv` with the downloadable
`customer-segmentation-model-7.3.31.rds` workbench bundle. It contains 20 rows from
that model's original test partition of the packaged labeled customer data, with
the nine predictors and the correct Segmentation labels. No labels are invented.

Choose **Use a saved model**, load the bundle, choose **Evaluate the model on
labeled data**, upload this CSV and click **Evaluate model**. Accuracy measures
agreement with those supplied labels. The original full-test score is 52.23%; a
20-row subset can have a different accuracy.

This is an interface demonstration reusing original test rows. It is not a new
independent benchmark. For an independent evaluation, supply held-out
labeled data that was not used to train or select the model. The optional matching
check should report 20 original-test input matches for this exact bundle; that
count is not its accuracy. The earlier customer-rows-to-predict.csv has no labels
and is for prediction-only mode.

[Matching model ZIP](https://raw.githubusercontent.com/MatHatter/DDESONN/artifacts/customer-segmentation-7.3.31/model-examples/customer-segmentation/customer-segmentation-model-7.3.31.zip)
