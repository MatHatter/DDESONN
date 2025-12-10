# DDESONN: Deep Dynamic Ensemble Self-Organizing Neural Network

Mathew William Armitage Fok (<quiksilver67213@yahoo.com>)

---

## Table of contents
1. [Project overview](#project-overview)
2. [Key capabilities](#key-capabilities)
3. [Architecture](#architecture)
4. [Getting started](#getting-started)
5. [Running the examples](#running-the-examples)
6. [Project layout](#project-layout)
7. [Datasets](#datasets)
8. [Roadmap](#roadmap)
9. [Contributing](#contributing)
10. [License](#license)
11. [Contact](#contact)

---

## Project overview

**DDESONN (Deep Dynamic Ensemble Self-Organizing Neural Network)** is an R-based research framework for adaptive neural network experimentation.  
It blends self-organizing principles with modern deep-learning practices to support:

- Configurable single-layer or multi-layer architectures  
- Dynamic ensemble learning with pruning and add-back mechanisms  
- Full control of optimizer, regularization, and activation flows  
- Reproducible evaluation and artifact reporting  

The codebase is intentionally transparent—implemented in plain R for maximum inspectability and experimentation.  
It is being prepared for public release on **CRAN**.

---

## Key capabilities

- **Flexible architecture selection** — toggle between SL (single-layer) and ML (multi-layer) setups with independent activations, dropout, and initialization.  
- **Manual training loop** — exposes full forward and backward propagation control.  
- **Optimizer suite** — includes Adam, SGD, RMSProp (refactor in progress).  
- **Regularization** — supports L1, L2, and mixed penalties on both weights and biases.  
- **F1 threshold tuning** — automatic F1-optimized threshold search with precision/recall scoring.  
- **Dynamic ensemble** — orchestrates multiple learners (`ensembles$main_ensemble`) with metadata tracking.  
- **Excel and Plot Reporting** — builds `writexl`/`openxlsx` output with optional ggplot visualizations.  
- **High-level API shim** — modern helpers in `R/api.R` for easy integration into new projects.  

---

## Architecture

Core structure:

- `R/DDESONN.R` — central R6 class defining SONN core logic and training.  
- `R/activation_functions.R` — activation library (ReLU, sigmoid, bent, etc.).  
- `R/optimizers.R` — Adam and other optimizers with `apply_optimizer_update()`.  
- `R/update_weights_block.R`, `R/update_biases_block.R` — modular update helpers.  
- `R/performance_relevance_metrics.R` — accuracy, precision, recall, F1, and advanced metrics.  
- `R/reports/` — Excel + Plot report builders.  
- `R/api.R` — lazy-load entry point for API-style integration.  

Formal R **vignettes** for guided exploration and reproducible demonstrations are available in the `vignettes/` folder.

---

## Getting started

### Prerequisites
- R ≥ 4.1 (RStudio project `DDESONN.Rproj` included)
- Dependencies listed in `DESCRIPTION`  
  (`R6`, `tidyverse`, `ggplot2`, `plotly`, `writexl`, `openxlsx`, `pROC`, etc.)

### Installation

```bash
git clone https://github.com/<your-org>/DDESONN.git
cd DDESONN
```

Inside R:
```r
required_pkgs <- c(
  "R6","cluster","fpc","tibble","dplyr","tidyverse","ggplot2","plotly",
  "gridExtra","rlist","writexl","readxl","tidyr","purrr","pracma","openxlsx",
  "pROC","ggplotify"
)
missing <- setdiff(required_pkgs, rownames(installed.packages()))
if (length(missing)) install.packages(missing)
invisible(lapply(required_pkgs, library, character.only = TRUE))
```

To load manually:
```r
source("R/DDESONN.R")
```

---

## Running the examples

Under `inst/scripts/` you’ll find ready-to-run demos:

- `DDESONN_mtcars_example.R` — basic binary classification  
- `DDESONN_mtcars_A-D_examples*.R` — architecture variants (A–D)  
- `Heart_failure_ScenarioA.R` — real dataset demonstration  
- `LoadandPredict.R` — artifact reloading example  
- `TestDDESONN.R` — full integration test  

Run directly:

```r
source("inst/scripts/DDESONN_mtcars_example.R")
```

Artifacts (predictions, confusion matrices, plots) are written to `artifacts/` and `artifacts_runs/`.

---

## Benchmark highlight: 1,000-seed Keras comparison

Empirical testing versus the **Keras (Python)** industry standard neural network shows that  
**DDESONN achieved competitive and in some runs superior accuracy** — often surpassing  
the benchmark across 1,000 randomized seeds.

Benchmark data and analysis are stored in:

```
C:/Users/wfky1/Desktop/DDESONN/helpfulFiles/vKeras/1000SEEDSRESULTSvkeras
```

Details, graphs, and discussion are available in the accompanying **vignette** under  
`vignettes/DDESONN_vs_Keras_Performance.Rmd`.

---

## Project layout

```
├── R/                  # Core code
│   ├── reports/        # Excel + Plot output
├── inst/scripts/       # Example scenarios
├── vignettes/          # Guided R vignettes and performance studies
├── data/               # Datasets
├── artifacts/          # Output workbooks and plots
├── artifacts_runs/     # Saved experiment runs
├── plots/              # ggplot/plotly visuals
├── README_v*.md        # Legacy drafts
└── LICENSE/            # Research-only license
```

---

## Datasets

Bundled sample data in `data/`:

- `heart_failure_clinical_records.csv` — UCI Heart Failure dataset  
- `WMT_1970-10-01_2025-03-15.csv` — Walmart stock dataset  
- `train_multiclass_customer_segmentation.csv` / `test_multiclass_customer_segmentation.csv` — segmentation toy set  

Use only for demonstration; verify original licensing if repurposed.

---

## Roadmap

- Unify non-Adam optimizers into a common API  
- Extend metrics for regression & multiclass  
- CRAN packaging with full documentation  
- Add support for attention & autoencoder modes  

---

## Contributing

1. Fork and branch from `main`.  
2. Run demos to confirm no regressions.  
3. Submit pull requests with description and tests.  

---

## License

DDESONN is released for **personal, educational, and research use only**  
under the terms in [`LICENSE/LICENSE`](LICENSE/LICENSE).  
Commercial use requires written authorization.

---

## Contact

For collaboration, research partnerships, or licensing inquiries,  
contact **Mathew William Armitage Fok** at <quiksilver67213@yahoo.com>.
