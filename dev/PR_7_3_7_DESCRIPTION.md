# DDESONN 7.3.7: Focused help snippets and a full-code viewer

Each help title now shows a short fragment for its own operation or setting. Epochs shows the epoch budget, Optimizer the update-rule argument, and Regularization strength the model penalty. All 31 topics have distinct snippets; they no longer repeat complete preparation and training programs.

A separate **Show full R code** button at the bottom of the workbench opens a read-only viewer with matching script downloads. Choose a next-model script using prepared data/current settings, the completed model's saved recipe, or saved-bundle load/predict code. Before preparation, the next-model example is explicitly labeled as iris. Saved recipes retain their original settings, and invalid next-model settings disable the script download.

The help icon now uses a fixed SVG circle and a real lowercase i, with an unclipped outline and centered glyph. Keyboard/touch access, scrolling, mobile placement and Escape dismissal remain available. Viewing help or full code never trains a model or changes weights.

Validation: 16 relevant workbench/dashboard tests and 398 assertions passed without failures, errors, warnings or skips. Tests guard distinct short fragments and replay complete scripts across binary/multiclass/regression and both checkpoint modes. Installed-app browser checks cover all 31 topics, the full viewer and its downloads, illustration/current/saved contexts, invalid settings, model immutability, keyboard/mobile interaction, circle geometry and glyph alignment. Both downloaded next-model and completed-model scripts reproduce the browser model's epoch, predictions and metrics. Installation succeeds. Full research-vignette/PDF package checks were not repeated for this UI patch; prior full-check results remain historical evidence.

This is version 7.3.7, based on released 7.3.6. No training mathematics, optimizer code, existing tags, merges or release publication are changed.
