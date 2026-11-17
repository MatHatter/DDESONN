"""Descriptive seed-block analysis of the frozen, clean-split 400-fit cohort."""
import csv
import hashlib
import math
from pathlib import Path
import statistics as st
from scipy.stats import t

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'per-seed-test-accuracy.csv'
rows = list(csv.DictReader(SOURCE.open()))
data = {(int(r['seed']), r['framework'], r['reduction']): r for r in rows}
assert len(rows) == len(data) == 400
assert set(data) == {(s, f, r) for s in range(1, 101) for f in ('ddesonn','keras') for r in ('sum','mean')}
assert all(int(r['test_rows']) == 802 and int(r['epochs']) == 360 and int(r['optimizer_updates']) == 78840 and r['checkpoint'] == 'final' for r in rows)

def stats(seeds, reduction):
    a = [100 * int(data[s,'ddesonn',reduction]['correct']) / 802 for s in seeds]
    b = [100 * int(data[s,'keras',reduction]['correct']) / 802 for s in seeds]
    d = [x-y for x,y in zip(a,b)]
    n = len(d)
    avg = st.mean(d)
    sd = st.stdev(d)
    half = float(t.ppf(.975,n-1))*sd/math.sqrt(n)
    return dict(n=n,ddesonn_accuracy=st.mean(a),keras_accuracy=st.mean(b),difference_pp=avg,sd_difference_pp=sd,ci95_low_pp=avg-half,ci95_high_pp=avg+half,ci95_halfwidth_pp=half,wins=sum(x>0 for x in d),ties=sum(x==0 for x in d),losses=sum(x<0 for x in d))

records=[]
for reduction in ('sum','mean'):
    for end in range(10,101,10):
        for kind,seeds in [('block',range(end-9,end+1)),('cumulative',range(1,end+1))]:
            records.append(dict(reduction=reduction,kind=kind,first_seed=min(seeds),last_seed=end,**stats(seeds,reduction)))
with (ROOT/'seed-deciles.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)

lines=['# Seed-block and cumulative analysis','',f'Source: `per-seed-test-accuracy.csv`, SHA-256 `{hashlib.sha256(SOURCE.read_bytes()).hexdigest()}`. All 400 records validated: four conditions, seeds 1–100 exactly once, 802 test rows, 360 epochs, 78,840 optimizer updates, final checkpoint. No training runs added.','', 'Positive differences favor the experimental DDESONN float32 matched-protocol adapter. Sum is compared with sum (FALSE), mean with mean (TRUE). Same frozen overlap-free split and original training settings in every condition.','', '## Separate blocks of ten','', '| Seeds | Sum advantage (pp) | Mean advantage (pp) | Sum W/T/L | Mean W/T/L |','|---|---:|---:|---:|---:|']
for end in range(10,101,10):
    a,b=[stats(range(end-9,end+1),r) for r in ('sum','mean')]
    lines.append(f"| {end-9}–{end} | {a['difference_pp']:+.3f} | {b['difference_pp']:+.3f} | {a['wins']}/{a['ties']}/{a['losses']} | {b['wins']}/{b['ties']}/{b['losses']} |")
lines += ['', '## Cumulative paired differences','', '| First N seeds | Sum advantage, 95% CI (pp) | Mean advantage, 95% CI (pp) |','|---|---:|---:|']
for end in range(10,101,10):
    a,b=[stats(range(1,end+1),r) for r in ('sum','mean')]
    fmt=lambda x:f"{x['difference_pp']:+.3f} [{x['ci95_low_pp']:+.3f}, {x['ci95_high_pp']:+.3f}]"
    lines.append(f'| {end} | {fmt(a)} | {fmt(b)} |')
lines += ['', '## Precision planning','', 'Approximate total seeds per condition needed for a two-sided 95% paired-t interval with the stated half-width, holding the observed paired-difference SD fixed. These are planning estimates, not guarantees or a sequential stopping rule.','', '| Reduction | Observed SD (pp) | Current half-width (pp) | Projected half-width at 250 (pp) | Total N for ±1 pp | Total N for ±0.5 pp |','|---|---:|---:|---:|---:|---:|']
for r in ('sum','mean'):
    x=stats(range(1,101),r)
    def need(margin):
        n=2
        while float(t.ppf(.975,n-1))*x['sd_difference_pp']/math.sqrt(n)>margin:n+=1
        return n
    projected=float(t.ppf(.975,249))*x['sd_difference_pp']/math.sqrt(250)
    lines.append(f"| {r} | {x['sd_difference_pp']:.3f} | {x['ci95_halfwidth_pp']:.3f} | {projected:.3f} | {need(1)} | {need(.5)} |")
    print(r,x,'N_for_1pp',need(1),'N_for_halfpp',need(.5),'250_halfwidth',projected)
lines += ['', '## Interpretation limits','', 'Intervals are ordinary pointwise paired Student-t intervals across training seeds, approximate for these bounded differences. They are descriptive after inspecting the completed cohort; they are not time-uniform confidence sequences and have no adjustment for repeated looks or multiple comparisons. Matching seed labels gives matched initial tensors, but framework-native shuffling/dropout streams differ. Blocks follow the original numerical seed order; no reordering or favorable selection was used. Ten-seed blocks can be noisy and are not independent datasets.','', 'All runs share one training/validation/test split. These intervals describe training randomness conditional on that split; they exclude uncertainty from sampling new test cases, new splits, datasets, or patients. More seeds cannot resolve those uncertainties. A stable cumulative curve does not prove convergence or universal superiority. Precision targets should be chosen based on useful accuracy differences before any further extension.','']
(ROOT/'SEED_DECILES.md').write_text('\n'.join(lines))
print('\n'.join(lines[6:33]))
