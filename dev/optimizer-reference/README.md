# Independent optimizer arithmetic checks

These tests check calculations and internal state, not prediction accuracy.
`generate.py` is a separate Python mathematical oracle using 60-digit decimal
arithmetic. It never imports DDESONN, reads its source, or invokes its R helpers.
Where possible it calculates moment/momentum accumulators from the entire
weighted gradient history rather than the incremental production recurrence.
The checked-in CSV lets the R suite run without Python or extra dependencies.

Coverage for all ten optimizers:

- Forty successive updates, checked at every step, for rectangular tensors and
  zero-initialized scalars; weight and bias storage orientations are both tested.
- Positive, negative, zero and tiny gradients; identical predetermined inputs.
- Three learning-rate segments, with moments retained across rate changes.
- Parameters and applicable moments, squared-gradient sums, velocities, FTRL
  accumulators, Lookahead slow weights and synchronization counter.
- Update count distinct from epoch, and serialization/resumption after step 19.

Reproduce from the repository root:

```sh
python dev/optimizer-reference/generate.py
Rscript -e 'pkgload::load_all("."); testthat::test_dir("tests/testthat", filter="optimizer-reference", stop_on_failure=TRUE)'
```

The generator is a mathematical reference, not an installed third-party optimizer
library. A shared misunderstanding of an equation is still possible; passing
these checks does not prove every possible configuration or the network's
back-propagation is correct. Existing two-step literal fixtures remain a separate
check. Framework comparisons must explicitly match algorithm conventions.

Conventions: momentum stores the velocity including the learning rate; NAG uses
that velocity convention. AdaDelta stores squared **unscaled** deltas and applies
the learning-rate multiplier afterward, using beta2 as rho. FTRL uses power -1/2,
beta=1, L1=L2=.01, alpha=current learning rate, initial z=n=0, and nonzero warm
parameters in the rectangular case. LAMB uses tensor-wide norms and decay .01.
Lookahead uses Adam, interpolation .5 and synchronization every three steps.
Epsilon=1e-6, beta1=.8 and beta2=.9; comparisons use tolerance 1e-10.

Algorithm references:

- AdaGrad: https://jmlr.org/papers/v12/duchi11a.html
- Adam: https://arxiv.org/abs/1412.6980
- AdaDelta: https://arxiv.org/abs/1212.5701
- FTRL-Proximal: https://research.google/pubs/ad-click-prediction-a-view-from-the-trenches/
- LAMB: https://arxiv.org/abs/1904.00962
- Lookahead: https://arxiv.org/abs/1907.08610

## Recorded validation

All ten optimizer reference cases passed: 11,360 expectations, covering both
parameter storage orientations and both initial-value scenarios over 40 steps.
The combined suite passed 58 test cases / 12,392 expectations with zero failures,
errors or skips, including the existing network and SSM cases. No optimizer
formula changes were needed in response to these new checks.
