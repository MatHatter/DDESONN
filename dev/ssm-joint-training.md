# SSM joint training and NQ evaluation

SSM models now default to `ssm_training = "joint"` in `ddesonn_fit()`.
The network and encoder have separate weights and optimizer state, but use the
same final prediction error on each training epoch. The engineered-only path
(`sequence_encoder = "none"`) keeps its existing updates and RNG behavior.

## Implementation

- `SONN$learn()` returns per-sample gradients with respect to network inputs.
- Each member forwards its current encoder before learning, extracts the
  embedding part of the input gradient, and backpropagates through its SSM.
- Identical sequence windows share a forward pass. Their conditional prediction
  gradients are summed with the original sample mean preserved; different trade
  conditions and labels on one window are not discarded or averaged into a
  substitute label in joint mode.
- The encoder uses its own Adam state. `ssm_lr = NULL` follows the head's current
  learning rate; a non-negative override is available.
- Sequence scaling is fitted only on training data. Encoder-produced embedding
  columns are excluded from legacy static-feature normalization/batch norm, so
  refreshed training embeddings use the same representation as inference.
- Encoders stay paired with ensemble members during promotion, refitting,
  serialization, and best-epoch restoration. Transient training caches are
  removed even if training fails.
- `ssm_training = "pretrain"` retains the separately supervised, then frozen
  encoder for comparisons. Its earlier 1-to-10-epoch pretraining limit remains;
  joint mode updates the encoder on every requested training epoch.

For `ddesonn_run()`, pass the mode and optional encoder learning rate through
`training_overrides`. Per-member `ssm_training_history` attributes record completed
encoder epochs, gradient norms, parameter changes, and the selected Adam step.

## Verification

All 29 SSM tests pass with 230 assertions on R 4.5.0. Coverage includes numerical
finite differences through the final weighted binary prediction loss into every
SSM parameter, repeated windows with different trade inputs and labels, ML and SL
heads, input dropout, more than ten epochs, zero gradients through a disconnected
head, validation isolation, independent ensemble members, resuming each member's
Adam state, serialization, best-epoch pairing, and failure cleanup.

Run the suite with `Rscript -e 'devtools::test(filter = "ssm")'`.
A separate 300-epoch mtcars seed-1 check reproduced the previous non-SSM
probabilities bit for bit. A fresh R session reloaded the NQ joint model and
reproduced seven holdout predictions within 3.9e-16; scaler, parameter shapes,
Adam state, and member pairing were retained.

## Matched NQ experiment

Data: user-supplied `DDESONN_NQ_FeatureDataset.csv` and `NQ_5Minute_History.csv`.
The input data and full model artifacts are not committed here.
The existing `Run_NQ_DDESONN_EndToEnd_SSM_v4.R` preprocessing and chronological
splits were retained: 58,391 training rows (415 unique observations), 16,975
validation rows (121 observations), and 13,087 holdout rows (93 observations).

Controls: 39 static features, 48 bars by 13 sequence features, 16 SSM states,
convolution width 4, head hidden sizes 32 and 16, three epochs, LR 0.001,
dropout 0.10, and the release/7.2 legacy `optimizer = "adagrad"` implementation.
Optimizer PR #7 is not included. Model seeds 1 through 5 were paired, with seed
111 rerun separately for comparison with earlier experiments. Initial head
weights, initial encoder weights, training controls and holdout row keys matched
within each pair. Both modes use the same embedding-normalization handling;
these frozen-mode results are therefore not byte-for-byte reproductions of the
older pipeline that normalized embedding columns again inside the legacy head.

| Mode | Mean holdout AUC | Mean log loss | Mean Brier score |
| --- | ---: | ---: | ---: |
| Separate pretraining / frozen encoder | 0.484301 | 0.688598 | 0.247670 |
| Joint training | 0.484988 | 0.688113 | 0.247424 |

For seed 111, frozen versus joint AUC was 0.464176 versus 0.464071; log loss was
0.687028 versus 0.685659, and Brier score was 0.246585 versus 0.245965.
All three encoder epochs completed, with finite, nonzero prediction-derived
encoder gradients. All twelve evaluations (ten paired-seed fits plus seed 111's
two fits) completed and their metrics were independently recalculated from saved
labels and probabilities.

Aggregate and per-seed values are in [ssm-results](ssm-results/).
The average gain is small and seed 111 has mixed results. These are exploratory
reruns on a previously inspected holdout, with 93 independent observations rather
than 13,087 independent market windows. They do not establish reliable predictive
or trading improvement. No hyperparameters were selected using these results.
