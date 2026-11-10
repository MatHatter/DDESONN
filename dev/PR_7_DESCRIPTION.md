This prepares the optimizer implementation for the future 7.3 release. Standard optimizers now apply one update per parameter tensor using the actual gradients and per-tensor state. The former custom AdaGrad behavior is removed: standard `adagrad` is the default, and `adagrad_x` is rejected with migration guidance rather than retained as a compatibility algorithm. Existing legacy optimizer state must not be reused to resume standard AdaGrad training.

Changes:

- Retain the corrected SGD, momentum, NAG, AdaGrad, RMSprop, Adam, AdaDelta, LAMB, FTRL and Lookahead implementations for single-layer and multilayer networks, including weights and biases.
- Explain gradients and each optimizer in short plain-language comments. Explain FTRL's L1 threshold: weak accumulated signals produce exact zero weights, which can become nonzero after stronger signals. These comment additions leave the parsed optimizer code unchanged.
- Integrate `release/7.2` at `0f4cfcef937c131c21735e9f6f7bac44b3c3be45`. This includes the SSM gradient, dimension, scaler and joint-training fixes. Resolve the README conflict while retaining standard AdaGrad and the 7.2 SSM documentation. The SSM implementation and its existing tests match that release branch.
- Synchronize the SSM help signatures and argument descriptions with the existing model/prediction APIs; no SSM behavior changes.
- Include independent two-step fixtures, a separate 60-digit mathematical oracle over 40 updates per optimizer, state/shape/resumption checks, and end-to-end training dispatch tests. The oracle is independent mathematical code, not a third-party optimizer-library comparison.
- Keep empirical mtcars performance experiments separate from arithmetic tests. The equal-budget comparison uses seed 1, 300 epochs and four candidate rates per optimizer, selects rates by validation log loss, and evaluates restored validation checkpoints. Earlier common-rate and two-optimizer-only tables are labeled separately. One small split is not evidence of universal optimizer superiority, and tuned rates do not change package defaults.

Validation:

- Optimizer, network and SSM suite: 58 cases, 12,392 passing expectations; no failures, errors or skips.
- Comment-only optimizer edits and API/SSM documentation edits preserve parsed R code exactly.
- Independent reference fixture regeneration is byte-for-byte reproducible.
- Package build and `R CMD check --no-manual --ignore-vignettes`: passed, with zero errors, warnings or notes.
- `git diff --check` passes against `release/7.2`.

Scope: this is development toward 7.3. DESCRIPTION remains 7.2.0; no release or PR merge is performed. PDF-manual generation and vignette rendering were not checked in this run (LaTeX is not installed; vignette experiments were excluded).
