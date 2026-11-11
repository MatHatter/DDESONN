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
