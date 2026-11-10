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
