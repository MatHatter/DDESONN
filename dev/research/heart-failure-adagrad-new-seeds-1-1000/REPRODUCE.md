# Reproduce or resume this new reconstructed benchmark

This is a new run of seeds 1–1000. Earlier 100-seed records and historical artifacts are separate. Do not substitute earlier outputs for seeds in this experiment. The original package checkout was not reset, and no release or optimizer change was made.

## Current cloud workspace

Experiment root: `/workspace/ddesonn-heart-failure-new-1000`.
R executable: `/workspace/.ddesonn-r/bin/Rscript`.
The extracted, checksum-verified 7.3.0 source is `source/`; the isolated installed package is `library/`. Runtime dependency versions and explicit conda package records are under `software/`.

Every R process must set `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`. The supervisor does this automatically. `R_LIBS_USER=/workspace/.ddesonn-library` supplies existing dependencies in this cloud image; frozen DDESONN is prepended to `.libPaths()` inside the R harness. Use the existing isolated checkout if doing repository work; do not create a Git worktree unless explicitly requested.

```sh
cd /workspace/ddesonn-heart-failure-new-1000
python harness/supervise.py --verify-only
python harness/supervise.py --workers 4
```

The second command resumes this new run: every checksum-valid completed seed is preserved. Invalid output directories are moved to `quarantine/`; missing or invalid seeds rerun. `progress.json`, `run.log`, per-seed `status.json`/`FAILURE.json`, SHA256 `manifest.json` and atomically written `COMPLETE.json` distinguish completion from failure. The freeze hash must match. A failure does not count as complete; rerun the same resume command after diagnosing it. Do not run two supervisors concurrently.

The cgroup has four CPUs and 32 GiB; four single-threaded workers fit measured pilot memory. Each seed starts a fresh R process and then a separate saved-model reload process. The validated pilot is a production prerequisite. Production includes seeds 1 and above from scratch; pilot directories are excluded from production counts.

## Fresh reproduction from the bundle

1. Verify the adjacent archive `.sha256` file, extract the archive, then run `sha256sum -c MANIFEST.sha256` from its root. BUNDLE.json and COMPLETION.json are external post-archive sidecars: an archive cannot contain its own final checksum. The internal manifest covers every included file except itself; bundle.py separately verifies its member too.
2. Install the dependency environment recorded in `software/conda-explicit.txt` (same-platform packages, MD5 records retained) or `software/environment.yml` using a trusted conda/micromamba installation. Keep package checksums and TLS verification enabled. The dependency environment is not embedded; frozen source, dataset and raw model artifacts are embedded. Set `DDESONN_RSCRIPT` to the resulting absolute Rscript path if different from the cloud path.
3. Install the frozen source, without editing it: `R CMD INSTALL --library=library source`. Set `R_LIBS_USER` to your dependency library if needed. The harness prepends the experiment-relative library, so it cannot silently select current DDESONN 7.3.11 instead of frozen 7.3.0.
4. To run a genuinely new experiment, copy the bundle to a new directory and move existing `seeds/`, `pilot-a/`, `pilot-b/`, `report/`, `quarantine/`, progress/status/completion files and logs to a separate preservation directory. Keep all original records intact. Never overwrite them or count them as fresh results. Verify `freeze.json` against `freeze.sha256` and all frozen inputs. Relocation changes no frozen input bytes.
5. Run `python harness/supervise.py --pilot a` and `--pilot b`, then `Rscript harness/compare-pilot.R "$PWD"`. Run `python harness/supervise.py --tamper-test`. Rerun the frozen optimizer suites with the command in evidence/arithmetic-command.txt. If anything changes in frozen inputs, make a new explicit freeze, retain the old freeze and repeat both pilots; never alter an in-progress experiment to meet a score target.
6. Run the supervisor with suitable concurrency until all 1000 seeds verify. Use `--verify-only` to independently check final completion.
7. Run `python report.py` and `python bundle.py`. Report generation requires all 1000 valid seeds. It recomputes confusion counts from probabilities, summaries, row errors and historical comparisons. It runs graphs.R and creates a standalone HTML report through pandoc. Bundle generation rechecks the seeds, hashes every included file, reads back every archive member and computes the whole-archive SHA256.

The helper scripts expect Python 3 and R dependencies from the supplied environment. `report.py` and `bundle.py` resolve their experiment root relative to themselves. The supervisor accepts `DDESONN_RSCRIPT` for relocation. `seed.R`/`reload.R` accept an explicit experiment root. `compare-pilot.R` also accepts that root.

## Protocol and interpretation

See PROTOCOL.md, config.json and the full report. The split, preprocessing, RNG seeding, architecture, native 2/1 class weighting, 360 epochs, fixed rate and threshold are frozen. The trainer exit trace reads history only. Initialization, RNG, final and selected checkpoints, native results, exact RDS probabilities and 17-digit CSVs are retained. Both weights and biases are restored before inference. No test-driven tuning, early stopping or seed selection is permitted.

The training and test feature overlap is substantial and unexplained. No patient identities are inferred. These repeated fixed-split predictions do not establish clinical generalization. The matched no-overlap Keras/DDESONN experiment is future work and is not run by these commands.

## Retrieval and storage

`BUNDLE.json` records the exact absolute archive path, size and checksum after verification. The full report is `report/REPORT.html` (self-contained), with Markdown, SVG/PNG graphs and CSVs alongside it. No external hosting/upload capability is available in this session. Local paths and cloud snapshot persistence are not proof of durable external storage; copy/download the bundle with your client's file access before discarding the environment.
