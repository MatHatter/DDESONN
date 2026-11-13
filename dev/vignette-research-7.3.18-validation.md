# 7.3.18 research documentation and vignette links

Based on main at c37d998 (7.3.17), preserving its workbench changes. This update
adds research documentation and guide navigation; model/optimizer source is unchanged.

Validated in the prepared R 4.5.3 environment:

- R CMD INSTALL: 7.3.18 installs and loads successfully.
- Existing dashboard tests and new vignette-link tests pass. The link tests cover
  installed HTML preference, online-source fallback and safe new-tab attributes.
- The new static benchmark vignette renders through rmarkdown without training.
- A fresh installed-package Shiny process serves the Vignettes tab, all five
  guide links, and the actual rendered benchmark HTML with HTTP 200.
- R CMD build --no-build-vignettes succeeds. Archive inspection confirms the new
  vignette source is included and dev/research/raw experiment artifacts are excluded.
  The existing computational demonstration vignettes were not rebuilt here.
- Research-report relative links resolve; git diff --check passes.

The separate frozen 7.3.0 benchmark completed 1,000 fresh production seeds and
12,162 arithmetic/reference assertions. Its validated bundle, protocol and source
are unchanged by this 7.3.18 documentation update. The full bundle remains outside
Git and the distributed package; ARTIFACTS.json records its local retrieval and SHA256.

The historical overlap-heavy split is intentionally retained. Documentation
explains that correcting standard AdaGrad does not repair the data overlap or
establish a fully matched Keras comparison. The no-overlap split fix and matched
rerun are explicitly separate future work. Beginner-example/model-loading changes
are deferred following the discussion.
