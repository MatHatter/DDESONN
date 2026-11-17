"""Plot observed seed precision and a labeled synthetic uncertainty example."""
import os
os.environ.setdefault('MPLCONFIGDIR','/tmp/ddesonn-matplotlib')
from pathlib import Path
import csv
import numpy as np
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
rows=list(csv.DictReader((ROOT/'per-seed-test-accuracy.csv').open()))
data={(int(r['seed']),r['framework'],r['reduction']):int(r['correct']) for r in rows}
ns=np.arange(10,101,10)
def curve(d):
    means=np.array([np.mean(d[:n]) for n in ns])
    halves=np.array([t.ppf(.975,n-1)*np.std(d[:n],ddof=1)/np.sqrt(n) for n in ns])
    return means,halves
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
colors={'sum':'#1769aa','mean':'#b44722'}
widths={}
for ax,red in zip(axes[0],('sum','mean')):
    d=np.array([(data[s,'ddesonn',red]-data[s,'keras',red])*100/802 for s in range(1,101)])
    m,h=curve(d);widths[red]=(d.std(ddof=1),h)
    ax.fill_between(ns,m-h,m+h,color=colors[red],alpha=.18,label='Pointwise 95% paired t interval')
    ax.plot(ns,m,'o-',color=colors[red],label='Cumulative mean advantage')
    ax.axhline(0,color='black',lw=1)
    ax.set_title(f'Observed: {red} vs. {red}')
    ax.set_ylabel('DDESONN − Keras accuracy (pp)')
    ax.legend(fontsize=8)
# Illustrative generated data, never included in the real experiment.
rng=np.random.default_rng(20261117)
synthetic=rng.normal(.2,7,100)
m,h=curve(synthetic)
ax=axes[1,0]
ax.fill_between(ns,m-h,m+h,color='#76549b',alpha=.18)
ax.plot(ns,m,'o-',color='#76549b')
ax.axhline(0,color='black',lw=1)
ax.set_title('HYPOTHETICAL: small effect, greater seed variation')
ax.set_ylabel('Synthetic accuracy difference (pp)')
ax.text(.03,.03,'Synthetic normal draws: mean 0.2 pp, SD 7 pp\nIllustration only; not benchmark results',transform=ax.transAxes,fontsize=8)
ax=axes[1,1]
future=np.arange(10,301)
for red,(sd,h) in widths.items():
    ax.plot(ns,h,'o',color=colors[red])
    ax.plot(future,t.ppf(.975,future-1)*sd/np.sqrt(future),color=colors[red],label=f'{red}: fixed observed SD')
ax.axhline(1,color='black',ls='--',label='Illustrative precision target: ±1 pp')
ax.axhline(.5,color='gray',ls=':',label='Stricter target: ±0.5 pp')
ax.axvline(100,color='gray',lw=.8)
ax.set_title('Precision planning: approximately proportional to 1/√N')
ax.set_ylabel('95% interval half-width (pp)')
ax.legend(fontsize=8)
for ax in axes.flat:
    ax.set_xlabel('Training seeds per condition')
    ax.grid(alpha=.2)
fig.suptitle('Seed-count diagnostics on one fixed, overlap-free split',fontsize=15)
fig.savefig(ROOT/'seed-precision.png',dpi=180)
fig.savefig(ROOT/'seed-precision.svg')
print('Saved seed-precision.png and seed-precision.svg')
