"""Verify exclusions and measure real built/installed sizes.
Usage: python dev/research/verify-package.py BEFORE_TAR AFTER_TAR BEFORE_LIB_PACKAGE AFTER_LIB_PACKAGE
Library arguments point to the installed DDESONN directory, not its parent.
"""
import csv, hashlib, json, sys, tarfile
from pathlib import Path
if len(sys.argv)!=5: raise SystemExit(__doc__)
before, after, before_lib, after_lib=map(Path,sys.argv[1:])
repo=Path(__file__).resolve().parents[2]
inventory=list(csv.DictReader((repo/'dev/research/legacy-benchmark-inventory.csv').open()))
def members(path):
 with tarfile.open(path) as archive:
  return {m.name:archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
b,a=members(before),members(after)
for r in inventory:
 path=r['path'];data=(repo/path).read_bytes()
 assert hashlib.sha256(data).hexdigest()==r['sha256'],path
 assert b['DDESONN/'+path]==data,path
 assert 'DDESONN/'+path not in a,path
 installed=path.removeprefix('inst/')
 assert (before_lib/installed).is_file(),installed
 assert not (after_lib/installed).exists(),installed
# Protect all existing input CSVs and nonhistorical supported examples/helpers.
for path in ['inst/extdata/heart_failure_clinical_records.csv',
 'inst/extdata/WMT_1970-10-01_2025-03-15.csv',
 'inst/extdata/train_multiclass_customer_segmentation.csv',
 'inst/extdata/test_multiclass_customer_segmentation.csv',
 'inst/scripts/DDESONN_mtcars_A-D_examples.R',
 'inst/scripts/DDESONN_mtcars_A-D_examples_regression.R',
 'inst/scripts/Heart_failure_ScenarioA.R',
 'inst/scripts/vsKeras/TablesPerSeedMostRecentRunResults.R']:
 assert a['DDESONN/'+path]==b['DDESONN/'+path],path
 assert (after_lib/path.removeprefix('inst/')).is_file(),path
# This cleanup must not change implementations, public exports or test fixtures.
for name,data in b.items():
 if name.startswith(('DDESONN/R/','DDESONN/tests/')) or name=='DDESONN/NAMESPACE':
  assert a.get(name)==data,name
vignettes=[p for p in a if p.startswith('DDESONN/inst/doc/') and p.endswith('.html')]
assert len(vignettes)==4,vignettes
current=(repo/'vignettes/DDESONNvsKeras_1000Seeds.Rmd').read_bytes()
for name in ['DDESONN/vignettes/DDESONNvsKeras_1000Seeds.Rmd',
             'DDESONN/inst/doc/DDESONNvsKeras_1000Seeds.Rmd']:
 assert a[name]==current,name
html=a['DDESONN/inst/doc/DDESONNvsKeras_1000Seeds.html'].decode()
for revision in ['1a41c9fe0c8e74190137370a2d507a7865b6bcae',
                 '4bc994462612ce868b90bc70885eb1635eddcdee',
                 '0d1b557098b56b39ed2357b30716dcddc25871b3']:
 assert revision in html,revision

assert not any(p.startswith(('DDESONN/inst/extdata/heart_failure_runs/',
 'DDESONN/inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/')) for p in a)
def size(path): return sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
result={'source_tar_bytes':{'before':before.stat().st_size,'after':after.stat().st_size},
 'installed_regular_file_bytes':{'before':size(before_lib),'after':size(after_lib)},
 'excluded_tracked_bytes':sum(int(r['bytes']) for r in inventory),
 'excluded_files':len(inventory),'rendered_vignettes':len(vignettes)}
for key in ['source_tar_bytes','installed_regular_file_bytes']:
 r=result[key];r['saved']=r['before']-r['after'];r['saved_percent']=100*r['saved']/r['before']
print(json.dumps(result,indent=2))
