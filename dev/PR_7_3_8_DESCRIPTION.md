# DDESONN 7.3.8: Explain learning-rate multipliers in code comments

The Learning rate example previously said “fixed-rate arguments” without explaining why `lr_decay_rate = 1` keeps a rate such as 0.01 unchanged. Shared comments now appear directly in that focused fragment and all newly generated fit recipes/full scripts: the setting multiplies rather than subtracts, `1` retains 100%, and factors 0.5/0.9 halve/retain 90% before applying the minimum floor.

The comments include the scheduler formula, explain that `lr_min = lr` also prevents decay, and state that API decay requires both `0 < lr_decay_rate < 1` and `lr_min < lr`. They identify the actual API default intervals: one epoch for classification and 20 for regression. The same semantics are noted beside the scheduler implementation, and the workbench help/guide explain its intentional fixed-rate configuration. Existing saved recipes remain historical snapshots.

Validation: installation of 7.3.8 succeeds; the actual scheduler agrees with the documented multiplier/interval/floor examples and API defaults. The scheduler source, 36 generated new/warm fit scripts and 18 learning-rate help variants have identical executable R syntax trees before/after the update. Installed-app browser checks verify the comments in help and the full-script display/download, including a changed learning-rate example, with no job/model creation or JavaScript errors. Full numerical, research-vignette and PDF suites were not rerun for this documentation-only change.

No scheduler values, training behavior, existing tags, merges or release publication are changed.
