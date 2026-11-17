"""Build the installed-app download from frozen sources, without training."""
from pathlib import Path
import zipfile

STUDY = Path(__file__).resolve().parent
REPO = STUDY.parents[2]
TARGET = REPO / 'inst/shiny/benchmark/reproduction.zip'
PREPARE = '''from pathlib import Path
import shutil, sys, json
root = Path(sys.argv[1]).resolve()
study = Path(__file__).resolve().parent / 'dev/research/keras-clean-reference'
sys.path.insert(0, str(study))
from community.extension import canonical
protocol = canonical()  # fail if any frozen training source differs
import importlib.metadata, os
for package, version in protocol['software']['python'].items():
    assert importlib.metadata.version(package.replace('_','-')) == version, package
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from audit_split import audit
if root.exists():
    audit(root)
    assert all((root / f'initial-{s}.json').exists() for s in range(1, 101))
else:
    root.mkdir(parents=True)
    shutil.copyfile(study/'split-rows.csv', root/'split-rows.csv')
    from reference import prepare
    prepare(root)
    assert audit(root) == json.loads((study/'protocol.json').read_text())['audit']
print('Original seeds 1-100 prepared; no fitting in preparation.')
'''
RUNNER = '''# Run from the extracted bundle root; requires the recorded R/Python environment.
# See dev/research/keras-clean-reference/CONTRIBUTING_SEEDS.md.
run_study <- function(framework, reduction, output = "study-results", first = 1L,
                      last = 100L, python = Sys.getenv("PYTHON", "python")) {
  stopifnot(framework %in% c("keras", "ddesonn"), reduction %in% c("sum", "mean"),
            length(first)==1, length(last)==1, first %% 1==0, last %% 1==0,
            first >= 1, last >= first, last <= 100)
  scripts <- "dev/research/keras-clean-reference"
  execute <- function(command, args) {
    status <- system2(command, vapply(args, shQuote, character(1)))
    if (!identical(as.integer(status), 0L)) stop("Command failed; see output above")
  }
  execute(python, c("prepare_original.py", output))
  output <- normalizePath(output)
  # Portable independent numerical checks before training.
  execute(python, c(file.path(scripts,"community/check_tensorflow.py"), output))
  rscript <- file.path(R.home("bin"), "Rscript")
  execute(rscript, c(file.path(scripts,"community/check_native.R"), output))
  if (framework == "keras") {
    execute(python, c(file.path(scripts,"run_keras.py"), output, reduction, first, last))
  } else {
    execute(rscript, c(file.path(scripts,"community/run_native.R"), output,
                        if (reduction == "mean") "TRUE" else "FALSE", first, last))
  }
}
'''
README = '''# Reproduce the current clean matched study

This download contains frozen sources and inputs for the 100-seed, four-condition
comparison. It contains no trained models. Extract it and set your working
directory to this folder. Install the recorded Python and R dependencies first;
see dev/research/keras-clean-reference/CONTRIBUTING_SEEDS.md and protocol.json.
Use CPU execution. Set PYTHON to your Python executable if needed, and set
DDESONN_REFERENCE_BLAS to a library exporting cblas_sgemm (original Linux path:
/usr/lib/x86_64-linux-gnu/libblas.so.3). Non-Linux execution is untested.

In R, source any of the four named launcher scripts. Each defaults to seeds
1-100, 360 epochs and the original settings; it prepares inputs, runs independent
numerical checks, then trains its condition. All four use the same study-results
folder and skip completed fit files. Run launchers sequentially. Keras fitting and
shared initialization require Python even when launched from R. For a shorter
range, source RUN_STUDY.R then run_study('keras','mean', first=1, last=10).

For seeds above 100 use the community extension workflow instead. Keep new results
separate from the frozen report. The original matched native runner is preserved
for provenance; launchers use the portable community native runner with identical
training and metric code. The unchanged historical compile.R contains original
workspace paths and is not the portable launcher.

The Shiny Show full R code button reproduces the displayed analysis of saved
results. These launchers rerun training, which can take hours. Neither is run
merely by viewing or downloading it. DDESONN here is the experimental native
float32 adapter; it is not the unchanged default trainer.
'''

def build():
    files = [REPO/'R/keras_clean_reference.R',
             REPO/'inst/extdata/heart_failure_clinical_records.csv',
             REPO/'inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py']
    files += [p for p in STUDY.rglob('*') if p.is_file() and '__pycache__' not in p.parts
              and p.suffix in ('.py','.R','.cpp','.json','.csv','.md','.txt','.png','.svg')]
    with zipfile.ZipFile(TARGET, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(files):
            info=zipfile.ZipInfo(p.relative_to(REPO).as_posix(),(2026,1,1,0,0,0))
            z.writestr(info,p.read_bytes(),compress_type=zipfile.ZIP_DEFLATED)
        extras={'prepare_original.py':PREPARE,'RUN_STUDY.R':RUNNER,'README.md':README}
        for f,r in [('keras','mean'),('ddesonn','sum'),('ddesonn','mean'),('keras','sum')]:
            extras[f'{f}_{r}.R']=f'# Run from the extracted reproduction bundle root.\nsource("RUN_STUDY.R")\nrun_study("{f}", "{r}")\n'
        for name,body in extras.items():
            z.writestr(zipfile.ZipInfo(name,(2026,1,1,0,0,0)),body,compress_type=zipfile.ZIP_DEFLATED)
    print(TARGET)

if __name__=='__main__':build()
