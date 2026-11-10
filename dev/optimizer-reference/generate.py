"""Independent high-precision mathematical oracle; never imports DDESONN.
Uses full-history weighted sums rather than the production EMA recurrences.
Regenerate from repository root: python dev/optimizer-reference/generate.py
"""
from decimal import Decimal as D, getcontext
from pathlib import Path
import csv
getcontext().prec = 60
B1, B2, EPS = D('.8'), D('.9'), D('1e-6')
METHODS = 'sgd sgd_momentum nag adagrad rmsprop adam adadelta lamb ftrl lookahead'.split()
def ema(history, rho, t, j):
    return (1-rho)*sum((rho**(t-1-k)*v[j] for k,v in enumerate(history)),D(0))
def sq(v): return [x*x for x in v]
rows=[]
for case, initial in [('rectangular',[D(x) for x in ['.2','-.3','.4','-.5','.6','-.7']]),('zero_scalar',[D(0)])]:
 for method in METHODS:
    p=initial[:]; slow=p[:]; gh=[]; gh2=[]; steps=[]; deltas=[]; ph=[]; sigma_history=[]
    for t in range(1,41):
        lr=D('.03') if t<=13 else D('.007') if t<=26 else D('.02')
        # Whole zero-gradient steps, exact sign reversals, tiny and unequal values.
        g=[D(0) if t%7==0 else D(((t*(j+3))%17)-8)/D(13+j) for j in range(len(p))]
        if t%11==0: g[0]=D('1e-12')
        gh.append(g);gh2.append(sq(g));steps.append(lr);ph.append(p[:])
        m=[ema(gh,B1,t,j) for j in range(len(p))]
        v=[ema(gh2,B2,t,j) for j in range(len(p))]
        r=[sum((x[j] for x in gh2),D(0)) for j in range(len(p))]
        momentum=[sum((B1**(t-1-k)*steps[k]*x[j] for k,x in enumerate(gh)),D(0)) for j in range(len(p))]
        state={}
        if method=='sgd': u=[lr*x for x in g]
        elif method in ['sgd_momentum','nag']:
            u=momentum[:] if method=='sgd_momentum' else [lr*g[j]+B1*momentum[j] for j in range(len(p))]
            state['momentum']=momentum
        elif method=='adagrad':
            u=[lr*g[j]/(r[j].sqrt()+EPS) for j in range(len(p))];state['r']=r
        elif method=='rmsprop':
            u=[lr*g[j]/(v[j].sqrt()+EPS) for j in range(len(p))];state['v']=v
        elif method in ['adam','lamb','lookahead']:
            direction=[(m[j]/(1-B1**t))/((v[j]/(1-B2**t)).sqrt()+EPS) for j in range(len(p))]
            if method=='lamb':
                direction=[direction[j]+D('.01')*p[j] for j in range(len(p))]
                pn=sum(sq(p),D(0)).sqrt();un=sum(sq(direction),D(0)).sqrt()
                ratio=pn/un if pn and un else D(1)
                direction=[ratio*x for x in direction]
            u=[lr*x for x in direction];state.update(m=m,v=v)
        elif method=='adadelta':
            old=[ema(deltas,B2,t-1,j) if t>1 else D(0) for j in range(len(p))]
            delta=[(old[j]+EPS).sqrt()/(v[j]+EPS).sqrt()*g[j] for j in range(len(p))]
            deltas.append(sq(delta));state.update(v=v,m=[ema(deltas,B2,t,j) for j in range(len(p))]);u=[lr*x for x in delta]
        elif method=='ftrl':
            before=[sum((x[j] for x in gh2[:-1]),D(0)) for j in range(len(p))]
            sigma_history.append([(r[j].sqrt()-before[j].sqrt())/lr for j in range(len(p))])
            z=[sum((gh[k][j]-sigma_history[k][j]*ph[k][j] for k in range(t)),D(0)) for j in range(len(p))]
            nxt=[D(0) if abs(z[j])<=D('.01') else -(z[j]-(D('.01') if z[j]>0 else -D('.01')))/((1+r[j].sqrt())/lr+D('.01')) for j in range(len(p))]
            u=[p[j]-nxt[j] for j in range(len(p))];state.update(z=z,n=r)
        p=[p[j]-u[j] for j in range(len(p))]
        if method=='lookahead':
            if t%3==0: slow=[(slow[j]+p[j])/2 for j in range(len(p))];p=slow[:]
            state['slow']=slow
        for j in range(len(p)):
            row=dict(method=method,case=case,step=t,coordinate=j+1,initial=initial[j],gradient=g[j],lr=lr,parameter=p[j],counter=t%3 if method=='lookahead' else '')
            row.update({k:state[k][j] if k in state else '' for k in ['m','v','r','momentum','z','n','slow']})
            rows.append({k:format(v,'.17g') if isinstance(v,D) else v for k,v in row.items()})
path=Path(__file__).resolve().parents[2]/'tests/testthat/fixtures/optimizer-reference.csv'
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator="\n");writer.writeheader();writer.writerows(rows)
print(f'Wrote {len(rows)} rows to {path}')
