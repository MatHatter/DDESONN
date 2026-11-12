# DDESONN 7.3.9

## Readable R examples and explicit saved-model instructions

- Separate code into logical blocks with blank lines, section comments and more
  comfortable line spacing in help, code panels and the full-script viewer.
- Label the saved-model example as R code for loading an existing model and
  predicting new rows without retraining. Explain what a bundle contains,
  which file paths to replace, predictor requirements and prediction outputs.
- Explain the original intention: a standalone script, predating the package
  API, for reloading a saved model with its original settings and using it again.
- Mark saving copies and exporting the standalone API model as optional steps.
  Existing completed-model recipes remain unchanged historical records.

# DDESONN 7.3.8

## Explain learning-rate multipliers in generated code

- State directly in the Learning rate help and newly generated R recipes that
  lr_decay_rate is a multiplier, not an amount subtracted. A value of 1 keeps
  100% of the starting rate, so it disables decay.
- Explain the task-specific default intervals, scheduler formula and minimum
  floor. Actual decay requires both a multiplier below 1 and a minimum below
  the starting rate; the workbench intentionally fixes both to prevent decay.
- Include before-floor examples for factors 0.5 and 0.9, and clarify the same
  semantics beside the scheduler implementation. Training behavior is unchanged.

# DDESONN 7.3.7

## Focused help and a separate full-code viewer

- Replace repeated preparation/training programs in help with short, distinct
  fragments specific to each title or setting.
- Add Show full R code at the end of the workbench. View and download a complete
  next-model script, a completed model's saved recipe, or saved-bundle usage code.
- Keep current prepared settings separate from completed-model snapshots and
  label the iris illustration used before preparation.
- Draw the help circle as a fixed SVG circle with a real lowercase i, keeping
  its outline round and unclipped across layouts and display scaling.
- Retain keyboard/touch access, scrolling, viewport placement and model immutability.

# DDESONN 7.3.6

## Centered help icons and equivalent R examples

- Optically center the lowercase i inside its circle while keeping the compact
  help controls accessible by keyboard, mouse and touch.
- Add equivalent public-API or base R code to every workbench help popover,
  covering data preparation, splits/scaling, training settings, checkpoint
  restoration, evaluation, model bundles, prediction and warm starts.
- Follow prepared settings and completed-model recipes where available; label
  iris/default illustrations and prerequisites explicitly before preparation.
- Add help beside split fractions, regularization and additional epochs.
- Keep expanded code readable in wider, scrollable popovers positioned within
  the viewport; inspection never starts training or changes model weights.

# DDESONN 7.3.5

## Training feedback and action states

- Enable Train only for prepared data while idle, Cancel only for an active job,
  and model downloads only for completed results. Gate warm starts, uploads and
  prediction downloads according to the current session state.
- Show checkpoint mode, selected epoch, validation criterion and score in
  Training and Model / Results. Final-epoch summaries describe the final tensors;
  best-validation summaries describe the restored checkpoint.
- Present actionable training errors with expandable technical details. Keep the
  previous completed model after failed or cancelled jobs.
- Inspect epoch and training loss by hovering near chart points or expanding a
  keyboard-accessible values table. Inspection never selects or restores weights.

## Explicit weight choice and switches

- Replace the checkpoint checkbox with a dropdown naming both choices: best
  validation checkpoint or final epoch (the last weights after the loop).
- Use right-aligned, keyboard-accessible switches for binary options such as
  standardization and initial self-organization, retaining their help icons.


## Checkpoint selection and equivalent R code

- Default new workbench fits to best-validation checkpoint selection, with a
  final-epoch alternative. Restore both weights and biases before evaluation,
  record the selected epoch and mark it on the loss chart. Both modes train for
  all requested epochs; selection is not early stopping and never uses Test.
- Add API examples under help icons, expandable equivalent-code panels across
  the workflow, and downloadable next-fit and completed-model R recipes. Keep
  completed recipes tied to saved settings rather than later UI changes.
- Retain final-epoch behavior for older model bundles without a checkpoint field.
- Restrict the legacy binary-only report tuning block to binary tasks so enabling
  validation reporting also works for multiclass and regression models.


## Workbench controls and layout

- Use smaller help circles with a plain lowercase i and explain training-only
  standardization beside its switch.
- Align the first content panels with their sidebar controls across the pages.
- Add an off-by-default initial self-organization control on Build & Train,
  forwarded to the existing API for new models. Save its setting; warm starts
  preserve the organized weights without repeating the phase.
- Show Test first in model performance while retaining validation and training
  views. Explain checkpoint selection separately from automatic early stopping.

# DDESONN 7.3.4

## Guided Shiny workflow

- Group the core workbench into four numbered steps: Data, Features, Build & Train,
  and Model / Results, with a shared color treatment and short next-step guidance.
- Separate dataset loading from feature selection. Group performance, saved
  models and predictions within Model / Results; retain Experiments separately.
- Add compact circled-i help beside section headings and important controls,
  including target, predictors, task, split, optimizer and model performance.
  Hover shows a description; click or keyboard activation opens inline help.
- Add next-step buttons and explain how earlier choices affect training and
  evaluation. Existing model training, cancellation and persistence are retained.
- Clarify callr's role as the optional background training process for Shiny.

# DDESONN 7.3.3

## Interactive model workbench

- Expand the optional Shiny app with Data, Build & train, Evaluate and Models
  pages using the public model, fit and prediction API. Support binary and
  multiclass classification and regression with finite numeric predictors.
- Fit preprocessing on training rows only and retain the split, class mapping
  and scaling in downloaded model bundles for evaluation and new predictions.
- Run training in an isolated callr worker with logged epoch progress, loss
  curves and cancellation. Failed/cancelled jobs retain the completed model;
  temporary files and active jobs are cleaned up when a session ends.
- Support additional fits from saved weights. Explain that optimizer accumulators
  and epoch schedules restart; this is a warm start, not an exact resume.
- Retain the benchmark viewer as Experiments. Explain legacy/standard AdaGrad and
  metric means, align its summary cards with the controls, and remove the visible
  artwork-enlargement caption while retaining the clickable image.
- Keep Shiny and callr optional. Advanced ensembles, sequence encoders and
  orchestration remain available through the R API rather than this first UI.

# DDESONN 7.3.2

## Optional experiment dashboard

- Add a native R Shiny explorer for paired seed metrics, with a compact 100-seed
  benchmark, CSV uploads, seed filters, metric summaries, plots and downloads.
- Include the project brain artwork locally, with a clickable enlarged view.
- Keep Shiny optional; no Python or external deep-learning backend is required.


## Documentation

- Keep detailed benchmark tables and figures in the separate research report;
  retain a concise result summary and link in the README.
- Add a path-by-path archive index for the ten files excluded from package builds
  starting in 7.3.0. Files remain in Git at the verified 7.1.11 archive.

# DDESONN 7.3.1

## Research documentation

- Add an exploratory heart-failure comparison of archived legacy AdaGrad and
  corrected standard AdaGrad across seeds 1–100, using 360 epochs per current
  run and the fixed historical reconstruction. The benchmark used unchanged
  7.3.0 source; this patch does not change an optimizer implementation.
- Record separate and combined summaries, sample SDs, paired differences,
  initialization uncertainty, arithmetic verification and artifact provenance
  under `dev/research/heart-failure-adagrad-seeds-1-100/`.
- Report mixed predictive findings: corrected mean test accuracy is 99.9680%
  versus 99.9427% legacy, while corrected log loss is higher overall. Original
  historical initialization is unverified, and 738/750 test rows match training
  inputs. These observations do not establish performance on new cases or a
  causal effect of the AdaGrad correction.
- Keep research documentation under the existing excluded `dev/` tree and raw
  models, probabilities and checkpoints outside the distributed package.

# DDESONN 7.3.0

This release focuses on the optimizer implementations and retains the SSM fixes
from 7.2.0. It does not introduce a new SSM algorithm or change the default
`sequence_encoder = "none"` setting.

## Optimizer maintenance

- Use standard AdaGrad as the default. Remove the former custom AdaGrad behavior,
  called `adagrad_x` on the development branch; the removed name raises a
  migration error rather than silently selecting another algorithm.
- Apply one update per parameter tensor using actual gradients and separate
  optimizer state for weights and biases, for single-layer and multilayer models.
- Correct and verify SGD, momentum, NAG, AdaGrad, RMSprop, Adam, AdaDelta, LAMB,
  FTRL and Lookahead updates, accumulator shapes and update counts.
- Add plain-language explanations of gradients and each optimizer, including
  FTRL's L1 rule that can set weights exactly to zero.

## Compatibility and verification

- Training trajectories and initialization RNG consumption can differ from
  earlier versions. Construct a new model for training; legacy optimizer state
  is not interchangeable with standard AdaGrad state.
- Add independent numerical fixtures and 40-step high-precision reference
  trajectories, state/resumption checks and end-to-end optimizer dispatch tests.
- Preserve the 7.2 SSM implementation and rerun its tests with the optimizer suite.
  Synchronize existing SSM API help signatures and argument descriptions.
- Keep exploratory mtcars accuracy experiments separate from arithmetic tests.
  Equal-budget tuning uses one seed and a small held-out set; it does not
  establish universal optimizer superiority or change global learning rates.

## Historical benchmark documentation

- Preserve the two 500-seed heart-failure runs and combined 1,000-seed Keras
  comparison as historical research evidence. Explain the legacy multilayer
  double-update path and why these results do not establish current standard
  AdaGrad superiority over Keras.
- Replace the unsupported categorical assertion that the original runs lacked a
  master seed with a source-based account: the earliest inspected script already
  contains master/per-run seed settings, while the exact original execution
  configuration cannot be reconstructed from the saved outputs.
- Retain a concise historical vignette with verified summary values and an
  immutable Git revision for retrieving raw results, workbook and scripts.

## Package-size reduction

- Exclude five generated historical RDS outputs and the historical workbook/
  experiment-script folder from CRAN source builds and installations, using
  scoped .Rbuildignore rules. Keep all ten files unchanged in Git and record
  their hashes and sizes in the development audit.
- Make the historical vignette self-contained so excluded research files are
  not installation or rendering dependencies. Preserve current example datasets,
  scripts and reporting functionality.

## Future benchmarking

- Recommend 10–30 matched seeds for exploration, consider 100 for larger stability
  studies, and increase the count when variability or precision requirements
  justify it. Record configurations and report distributions and uncertainty.
- Match data, splits, preprocessing, metrics and experimental conditions. Small
  seed counts do not establish statistical superiority.
- Parallel processing is deferred beyond 7.3.0; none is implemented in this release.

# DDESONN 7.2.0

This minor release focuses on ironing out the experimental SSM integration.

## SSM correctness and training

- Correct SSM gradient calculations and preserve parameter dimensions in Adam state.
- Train the SSM jointly from the final prediction error, with separate encoder
  weights and optimizer state. Retain the earlier frozen/pretraining mode for
  explicit comparisons.
- Fit sequence scaling on training data only and preserve encoder/network pairing
  across ensemble promotion, refitting, checkpoint selection and serialization.
- Keep SSM optional and off by default (`sequence_encoder = "none"`).

## Verification and evidence

- SSM verification covers numerical gradients, dimension and Adam-state invariants,
  repeated sequence windows, independent ensemble members, validation isolation,
  checkpoint restoration, save/reload behavior and failure cleanup.
- The SSM suite passed 29 tests with 230 assertions on R 4.5.0. A separate
  non-SSM, 300-epoch mtcars check reproduced the earlier predictions exactly.
- The exploratory NQ evaluation found only small, inconclusive changes. Current
  benchmarks do not establish a meaningful predictive benefit from adding SSM.
- Separate development benchmarks included an ETTh1 comparison with published
  TimeMachine scores and a one-seed Keras implementation check. TimeMachine's
  model was not run to generate those reference scores. The Keras check does
  not establish superiority of either framework. Benchmark code and its
  experimental optimizer changes are outside this release.

## Community participation and scope

Independent evaluations, reproducible examples of useful applications, bug fixes
and documentation improvements are strongly encouraged through
[GitHub issues](https://github.com/MatHatter/DDESONN/issues) and
[pull requests](https://github.com/MatHatter/DDESONN/pulls).
You can also [email the author](mailto:quiksilver67213@yahoo.com) or contact him
on LinkedIn, as described in the README.

Performance comparisons should use repeated seeds (10 or more where practical),
matched controls and reported variability. A single-seed check is exploratory.

Optimizer work remains separate in PR #7 and is not included in version 7.2.0.
