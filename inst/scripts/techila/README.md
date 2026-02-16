# Techila Distributed Execution

> This directory contains optional external execution scripts.  
> Techila integration is not required for normal package usage and is not a CRAN dependency.

This directory provides helper scripts for running DDESONN experiments in a distributed Techila environment.

These scripts:

- Use the installed **DDESONN** package API (no `source()` and no direct `R/` file loading)
- Require a configured **Techila** backend
- Are **not required** for standard package usage

DDESONN behaves identically in local execution mode. Techila support is provided solely to reduce wall-clock time for large experiment runs.

---

## Why Parallel Processing Is Ideal Here

DDESONN workflows frequently involve **many independent runs** that are naturally parallelizable:

- Randomized seed sweeps (e.g., 100–1,000 seeds)
- Ensemble and temporary ensemble runs
- Longer training schedules (e.g., 360+ epochs)

Each run is typically **independent** (different seed, ensemble member, or iteration), meaning it can be dispatched to its own worker without affecting correctness. Sequential execution compounds wall-clock time quickly when individual training jobs take meaningful time.

As a development benchmark reference: a 1,000-seed Keras comparison sweep required approximately **~2 days** end-to-end (roughly **~1 hour per 100 seeds**, give or take), primarily due to sequential execution. The same scaling principle applies to DDESONN. Once epoch counts increase (e.g., >10, and especially ~360), per-run wait time becomes significant, and parallel execution becomes the practical option for large sweeps.

Techila is most beneficial when:

- Running **many seeds** (hundreds or thousands)
- Running **ensembles** (multiple candidate models)
- Using **higher epoch counts** (e.g., 360+)
- Needing throughput without modifying model logic

Parallelization does not change model behavior or results — it reduces wall-clock time by distributing independent runs across workers.

---

## Execution Runners

Two execution modes are provided for parity and validation:

- `single_runner_local_mvp.R`  
  Local execution using the installed DDESONN package. Useful for baseline validation and reproducibility.

- `single_runner_techila_mvp.R`  
  Distributed execution via Techila. Intended for large-scale or computationally intensive runs.

Both runners call the same DDESONN package API and are designed to produce comparable outputs.

---

## Requirements

To run the Techila scripts, you must have:

- The **DDESONN** package installed
- The **foreach** package installed
- The **techila** package installed
- A working Techila configuration on the submitting machine

Techila support is optional and must be installed separately if used.

---

## Alternative Parallel Infrastructure

While these scripts focus on Techila, the same distributed-run pattern can be implemented using other parallel compute environments.

Examples include:

- Microsoft Azure virtual machines or batch compute services
- Amazon Web Services (EC2, Batch, or similar)
- Any multi-core or multi-node cluster environment

DDESONN’s seed-based and ensemble-based workflows are naturally parallelizable because individual runs are independent.

If you implement and validate an alternative parallel backend that preserves output parity with the local runner, contributions are welcome via pull request.
