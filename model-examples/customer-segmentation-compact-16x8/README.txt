Customer Segmentation — Compact 16x8

1. Extract this ZIP.
2. In Shiny, choose Use a saved model.
3. Optionally click Clear saved-model workspace to start blank.
4. Upload customer-segmentation-compact-16x8.rds.
5. To measure accuracy, choose Evaluate the model on labeled data and upload
   compact-original-test-example.csv. Click Evaluate model.
6. To get unknown answers, choose Predict answers for unlabeled data and upload
   compact-unlabeled-prediction-example.csv. Click Run predictions.

This is a different trained model from the earlier 32x16 customer model.
Hidden layers: 16 and 8; Adam .003; 100 epochs; seed 42; selected checkpoint 90.
Fitted category encoding, missing-value rules and scaling are saved in the bundle.
It uses the same grouped split/preprocessing as the original model, with no exact
transformed-input overlap between training, validation and test.
Rows: 4,834 training / 1,618 validation / 1,616 test.
Recomputed accuracy: training 52.61%, validation 50.00%, test 51.05%.
The 20-row labeled demonstration scores 50.00%. It repeats original test rows,
so it is not a new independent evaluation or a replacement for full-test scores.
The unlabeled example cannot measure accuracy. Both CSVs contain the nine saved
predictors; the evaluation CSV additionally contains the known Segmentation labels.

Trained with DDESONN 7.3.35. The model was saved, reloaded and validated, and CSV
predictions matched after reload. The original model/downloads were not changed.
Reproduction uses train-compact.R and the original 7.3.31 model to reuse its exact
prepared partitions. No claim is made that the compact model is better.
