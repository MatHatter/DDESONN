# Proposed 7.3.1 research documentation note

Add an exploratory heart-failure report comparing archived legacy AdaGrad results
against corrected standard AdaGrad for seeds 1–100 under a fixed 360-epoch
protocol. Include separate 1–50 and 51–100 summaries, combined sample SDs and
paired differences, probability metrics, initialization/checkpoint audits and
independent arithmetic verification. The benchmark ran the unchanged 7.3.0
implementation; this documentation does not change an optimizer.

Keep conclusions descriptive: original historical initialization is unverified,
and 738/750 test rows match training inputs. Store raw models and prediction
archives outside the package payload. NEWS for 7.3.1 includes a concise version of this note. No release has been
published.

The 100-seed reconstruction reports higher mean test accuracy with standard
AdaGrad (99.9680% versus 99.9427%, +0.0253 percentage points), lower overall Brier
score and higher overall log loss. Continue with the standard single-update
recurrence; the legacy multilayer double-update path was removed in 7.3.0.
A possible remaining 900-seed extension is deferred. The historical 1,000-seed
Keras comparison has not been rerun, so any improvement there is untested.

The README surfaces the existing paired-results figure and lists the exact ten
historical files excluded from package builds starting in 7.3.0, with immutable
links to their verified 7.1.11 archive. Files remain in Git; these exclusions are
recorded once as a versioned change, not repeated as new removals in later releases.
