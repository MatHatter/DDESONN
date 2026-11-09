# Optimizer review

Reviewed the unmodified checkout at commit
`5dd541a7fe5c6fc56ddc5b7d4a3e8b0ec7d55373` (version 7.1.11).
This report concerns the native network optimizers. The optional SSM encoder
has a separate Adam implementation in `R/ssm_encoder.R`, which is unchanged.

## Default compatibility decision

Following review, the default is now **`adagrad_x`**, preserving the pre-fix
custom Adagrad behavior from the reviewed commit. The corrected standard
implementation is available explicitly as **`adagrad`**. Existing scripts with
an explicit `optimizer = "adagrad"` must use `"adagrad_x"` to request the
pre-review custom algorithm; omission now selects `adagrad_x` automatically.

`R/adagrad_x.R` retains the original Adagrad expressions in an isolated private
implementation. Only unreachable branches for other optimizers were removed.
It intentionally retains the multilayer manual-plus-adaptive updates, matrix
ordering, bias error sums, original gradient preparation, and initialization
RNG draws. These are documented compatibility semantics, not claims about
standard mathematical Adagrad. The standard fixes below remain available and
are not silently applied to `adagrad_x`.

The compatibility tests use fixtures generated from the **unmodified installed
package at the exact reviewed commit**, not from the new compatibility code.
They match parameter values/shapes, optimizer state, groups and RNG state after
three updates across 16 SL/ML, regularization and update-flag configurations.
Two complete three-epoch regression runs additionally match predictions,
parameters and RNG state exactly, including multilayer dropout. These establish
compatibility for the tested workflows at this commit; they do not certify
reproduction of unspecified older benchmark versions.

## Training flow

This is a map of the package's existing training pipeline, not a new pipeline
created for this review. The changes are the optimizer selection/default,
standard parameter-update implementation, isolated Adagrad X compatibility
route, and the prediction-result fallback noted below. The forward/backward
network architecture and optional self-organization phase are retained.


1. `ddesonn_run()` in `R/api.R` merges `training_overrides` into
   `ddesonn_training_defaults()` and calls `ddesonn_fit()` for MAIN and TEMP
   models. TEMP-specific overrides still take precedence for TEMP models.
2. `ddesonn_fit()` resolves controls and passes `optimizer`, `lr`,
   `beta1`, `beta2`, `epsilon`, `lookahead_step`, and update flags to
   the R6 model's `train()`.
3. `DDESONN$train()` in `R/DDESONN.R` calls each ensemble member's
   `SONN$train_network()`. Optional `self_org=TRUE` performs a separate,
   explicit self-organization phase before ordinary training; it is not an
   optimizer step and remains unchanged.
4. Each training epoch calls `SONN$learn()` for forward propagation and
   backpropagation. Output sigmoid/softmax classification uses the cross-entropy
   delta shortcut; other layers use activation derivatives and dropout masks.
   Weight gradients are `t(input) %*% delta`; bias gradients are
   `colMeans(delta)`. That existing sum/mean distinction is preserved.
5. For standard optimizers, `update_weights_block()` and `update_biases_block()`
   share `.update_parameter_block()`. Each enabled tensor passes through existing
   gradient-norm clipping, coupled regularization, and one optimizer call.
   Bias gradients come from `learn()`, including in single-layer mode.
   `adagrad_x` instead dispatches to its isolated compatibility implementation.
6. `apply_optimizer_update()` in `R/optimizers.R` binds state to the current
   model tensor, obtains a descent delta, subtracts it once, and returns the
   actual resulting tensor and its state. Lookahead returns its synchronized
   tensor through the same assignment route.

State remains local to one `train_network()` invocation and carries between
its epochs. Repeated calls to `ddesonn_fit()` still start fresh optimizer
state; this change does not introduce checkpointed optimizer resumption.

## Findings against the reviewed commit

"Manual update" in the original notes means a direct arithmetic formula in the
code, such as `weights <- weights - lr * gradient`. Nobody updates values by
hand: both it and the optimizer call run automatically. "Optimizer dispatch"
means choosing and calling the function for the selected optimizer. In the
original multilayer route, the direct formula ran first and the optimizer
then changed the same parameter again. Adagrad X deliberately preserves that
original custom rule; corrected standard optimizers use a single update.


The resolutions below describe the corrected standard optimizers. Adagrad X
retains its documented original behavior for compatibility.

| Lead | Verified finding and affected functions | Resolution |
| --- | --- | --- |
| Multiple updates | Confirmed for ML weights and biases in both update blocks: the code first subtracted a gradient-based change directly, then called the selected optimizer for a second change. The SL route did not have that same double update. | One shared update route; update flags still freeze their respective tensors. |
| Matrix ordering | Confirmed: `apply_optimizer_update()` reconstructed several weight updates with `byrow=TRUE`, rearranging column-major matrix values. | Keep element order and dimensions; reject incompatible shapes rather than repeat, pad, or truncate gradients/state. Equivalent bias row/column vectors preserve unit order. |
| Initialization | Confirmed: `initialize_optimizer_params()` drew random parameter matrices, even for optimizers that did not need them. | No random draws; bind actual parameters for FTRL, LAMB, and Lookahead; initialize only the accumulators each method needs. |
| Adagrad | Its core accumulator formula was sound. Integration had ML double updates, reordered weight deltas, and an ML bias override using `colSums(errors)` instead of the mean gradient. | Use the supplied bias gradient and one correctly ordered update in both modes. |
| Adam / RMSProp | Their basic moment formulas were sound. ML double updates, matrix ordering, and silent shape coercion affected their application. | Shared strict application route; Adam correction uses the tensor's update count, rather than assuming epoch always equals step. |
| SGD | Confirmed: `sgd_update()` returned dummy zero bias deltas. ML's preceding manual update obscured the problem. | Identical descent-delta rule for weights and biases. |
| Momentum | Confirmed: the weight velocity was negative but then subtracted, reversing descent. Biases received plain SGD instead of momentum. | Store velocity using the subtractive convention for both targets. |
| NAG | Confirmed: matrix state was indexed as a list of tensors, causing element/scalar reuse; biases used plain SGD. | Shape-matched velocity for each tensor and Nesterov momentum for both targets. |
| FTRL | Confirmed: the helper produced an absolute next weight, but the dispatcher subtracted it as a delta. `sigma` multiplied zero instead of the actual parameter; biases were zero and the L1 zero region was absent. | Apply the FTRL-Proximal recurrence with actual parameters, L1 thresholding, and return `old - next` as the delta for both targets. |
| LAMB | Confirmed with a qualification: the helper multiplied by LR for its private parameter copy, but returned an unscaled direction that the dispatcher applied. Biases were zero; trust norms used random/stale copies; correction used first-step denominators on every step. | Actual parameter norms, per-step correction, decay included before trust scaling, and exactly one LR factor in the applied delta for both targets. |
| Lookahead | Confirmed: fast values started from random copies and slow values from zero. Sync merely copied fast to slow without interpolation, and bias deltas defaulted to zero. | Start fast/slow from the model; Adam inner steps; every k steps interpolate slow toward fast with alpha=0.5 and reset fast to slow. |
| Adadelta | Confirmed: dispatcher referenced undefined `errors`. Its helper also used Adam-style correction and ignored LR. | Consume the supplied gradient; use Adadelta squared-gradient/squared-delta EMAs and apply LR. Advertise `adadelta` in the optimizer options list. |

The implementation consolidates duplicated per-optimizer dispatch and shape
handling, retaining existing function signatures, optimizer identifiers, and
helper return-field names. Internal state layout changes and is not a supported
migration format for previously hand-saved optimizer lists.

## How the corrected standard optimizers change parameters

The following corrections apply to the standard implementations. The default
`adagrad_x` retains its original compatibility semantics described above.

- Each helper returns a signed change called a delta. The caller subtracts it:
  `parameter <- parameter - delta`. A delta of 0.1 changes a weight from 0.5 to
  0.4; a delta of -0.1 changes it from 0.5 to 0.6. "Subtractive" does not mean
  every number in the delta is positive.
  Each layer's weights and biases keep separate optimizer memory (for example,
  past squared gradients), with the matching matrix dimensions. This is not
  ensemble aggregation or averaging predictions.
- Momentum uses `v <- beta1*v + lr*g`; NAG uses
  `delta <- lr*g + beta1*v`. This is the usual current-gradient Nesterov
  velocity form, without a second forward/backward pass at a lookahead point.
- Adam and LAMB correct moments with the actual update count.
- Adadelta uses `beta2` as rho for both EMAs, no Adam correction, and
  `lr*sqrt(E[delta^2]+epsilon)/sqrt(E[g^2]+epsilon)*g`.
  Its historical helper arguments remain accepted.
- FTRL uses training LR as alpha unless the internal dispatcher receives an
  explicit alpha. Beta remains 1; its existing internal L1/L2 defaults remain
  0.01. LAMB's existing internal decay default remains 0.01. These internal
  defaults are separate from the training `reg_type` penalty.
- Coupled regularization is added to the gradient before the sole optimizer
  step. Previously ML applied an additional, unscaled-by-LR penalty with its
  direct-formula update, while SL optimizer paths could omit that penalty.
- Existing gradient-norm clipping (including its minimum norm) is preserved.
  Hidden optimizer-specific clipping of the resulting deltas is removed so
  optimizer calculations follow their stated recurrences.
- Wrong shapes/nonfinite gradients now fail explicitly rather than silently
  recycling, truncating, or filling elements. Bias row/column representations
  with the same unit count are explicitly supported.
- Removing initialization RNG draws changes subsequent random-number
  consumption (for example, dropout). Corrected updates also change model
  trajectories. An old seed alone cannot reproduce pre-fix results.

Corrected Adagrad also exposed a latent training-result fallback bug. With the
documented binary mtcars example and seed 1, post-update accuracy need not exceed
the pre-update accuracy, so no best snapshot is saved. Previously the no-snapshot
path could return `learn_result` (which has `learn_output`) where callers
expected `predicted_output`, causing `compute_error(NULL, ...)`.
`SONN$train_network()` now returns the already evaluated last prediction when
that happens. The unmodified commit's example passed; the corrected update
exposed this path. A regression test covers it.

## Validation

These tests were newly written and actually executed during this review. They
are not results recalled from earlier sessions. The compatibility fixtures
were generated in this session by running the unmodified installed package at
the stated commit.

The tests train real model instances through `ddesonn_run()`/`ddesonn_fit()`
and call `ddesonn_predict()`. Training itself computes errors/loss against the
supplied labels. However, the assertions primarily verify optimizer formulas,
state, output shapes/finiteness, and exact reproduction of baseline model
predictions. Matching a previous model's predictions is different from checking
those predictions against observed outcomes.

No high classification accuracy, low holdout error, or superiority over another
optimizer has been established by these tests. The mtcars classification check
uses its training data and does not impose an accuracy threshold. The large
`inst/scripts/TestDDESONN.R`, 1,000-seed comparison, and NQ research scripts have
not been executed as performance benchmarks in this review.

The original script selections now explicitly use `adagrad_x`, each with a
comment explaining the preserved custom behavior and pending separate evaluation
of standard `adagrad`. This changes the optimizer identifier, not their other
training or evaluation controls. Dedicated standard-Adagrad tests continue to
select `adagrad` intentionally.


A subsequent accuracy comparison now runs the repository's existing small
mtcars classification example for standard Adagrad and Adagrad X across five
paired model seeds, with the same initial weights and data splits. It evaluates
predictions against observed labels on training, validation and held-out test
rows. Mean test accuracies were 74.3% and 71.4%, respectively, on only seven test
rows per seed. This is exploratory evidence, not a general ranking or a
reproduction of the large research benchmarks. See
[the comparison report](optimizer-comparison-mtcars.md) and the runnable
[comparison driver](compare-adagrad-mtcars.R) for setup and measured errors.

The recovered historical script was subsequently run with 300 epochs and its
0.5-every-20-epochs LR decay schedule. With the original split-seed-42 RNG flow,
Adagrad X achieved 100% on train/validation/test; standard Adagrad achieved
100%/100%/71.4%. Five additional matched model seeds averaged test accuracy
82.9% for Adagrad X and 85.7% for standard Adagrad. These are current
implementations under recovered v6.6.1 settings, not an exact rerun of the
historical v6.6.0 package. See the
[historical-settings comparison](optimizer-comparison-historical-mtcars.md).

`tests/testthat/test-optimizers.R` contains 18 cases with 785 passing
expectations, no failures or skips:

- Independently calculated fixed two-step results for every optimizer, applied
  through the real blocks to SL and ML weights and biases.
- Distinct signed elements in 2x3 and 3x2 matrices; state and parameter shape,
  binding, update counts, Adagrad accumulation, and Lookahead synchronization.
- RNG preservation, invalid-shape/nonfinite rejection, update flags, coupled L2,
  bias batch-size independence, row/column/vector bias storage, FTRL thresholding,
  and LAMB LR scaling.
- Two-epoch `ddesonn_run()` workflows for all ten optimizers in both modes,
  spying on dispatch to verify `training_overrides` selection, and checking
  finite predictions with the expected dimensions.
- The documented default Adagrad classification example without a best snapshot.

Full testthat run: 35 cases, 33 passed; 909 passing expectations, one failed
expectation and one error. Both remaining failures were observed before optimizer
changes at the reviewed commit:

1. `test-ssm-encoder.R:19`: convolution gradient 0.447 versus numerical 0.537.
2. `test-ssm-encoder.R:61`: `expect_length(..., info=nm)` uses an unsupported
   argument with testthat 3.3.2.

All four SSM bridge cases pass. These unrelated SSM files were not changed.

`tests/testthat/test-adagrad-x.R` adds 3 compatibility cases with 61 passing
expectations and no failures or skips. Default controls select `adagrad_x` in
binary, multiclass and regression modes; `adagrad` remains explicitly selectable.

Build/install/load pass under R 4.5.3. Build and package-check commands:

```sh
cd /workspace/ddesonn-setup/optimizer-check
/workspace/.ddesonn-r/bin/R CMD build --no-build-vignettes /workspace/DDESONN
R_RD4PDF=times,hyper /workspace/.ddesonn-r/bin/R CMD check --no-vignettes --output=final-check DDESONN_7.1.11.tar.gz
```

The long research-vignette executions/rendering are excluded. The initial check
started executing them and was stopped; it is not counted as a completed check.
The standard manual font configuration requires missing `inconsolata.sty`;
`R_RD4PDF=times,hyper` uses the supported font configuration without that optional
font package.

The completed check reports **1 ERROR and 4 WARNINGs**:

- The error is the two baseline SSM test failures listed above.
- Warnings concern existing SSM code/documentation signature mismatches,
  undocumented SSM arguments, and two checks for missing rendered vignette
  files. The relevant SSM documentation and vignette source files are unchanged.
- Syntax, dependency use, load/unload, static-code analysis, examples, and PDF
  manual generation pass. Example failure seen during development was resolved
  by the prediction fallback fix and its regression test.
- Remote CRAN/Bioconductor index probes were unavailable under the current
  network policy; local installed-dependency validation completed successfully.

Detailed completed-check log:
`/workspace/ddesonn-setup/optimizer-check/final-check/DDESONN.Rcheck/00check.log`.

Logs and machine-readable test results are outside the checkout under
`/workspace/ddesonn-setup/`.

## Implications for optimizer comparisons

At the reviewed commit, ML comparisons measured composite manual-plus-optimizer
updates, with optimizer-specific bias and matrix-ordering defects. Those runs
cannot straightforwardly establish comparisons between the standard named
algorithms. Adagrad's mathematical accumulator itself was not the source of all
these problems; its integration and batch treatment were affected.

This does not establish that any particular past benchmark is invalid. No exact
historical benchmark commit, data split, preprocessing, seed list, or saved
configuration was supplied, so earlier benchmark implementations/results were
not adjudicated. The baseline source extracted for diagnosis is exactly the
reviewed HEAD, not an assumed historical benchmark version.

For a particular comparison, first recover that provenance and verify the
actual code path at its commit, including ML/SL mode, self-organization,
regularization, dropout, and update flags. Preserve the historical result and
rerun matched baselines with these corrections, reporting seed distributions
and uncertainty rather than assuming either universal invalidity or unchanged
performance. Existing weight-gradient sums and bias-gradient means, clipping,
and scheduler controls still matter when designing a fair comparison.
