"""Prepare, run and validate additional seeds without modifying the frozen 100."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
STUDY = HERE.parent
REPO = STUDY.parents[2]
CONDITIONS = [('keras', 'mean'), ('ddesonn', 'sum'), ('ddesonn', 'mean'), ('keras', 'sum')]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def canonical():
    p = json.loads((STUDY/'protocol.json').read_text())
    for name, expected in p['adapter_sha256'].items():
        if sha(REPO/name) != expected:
            raise ValueError(f'Frozen protocol code differs: {name}')
    if sha(REPO/'inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py') != p['reference_sha256']:
        raise ValueError('Archived Keras source differs')
    return p

def prepare(root, first, last):
    p = canonical()
    if first < 101 or last < first or last > 2147483647:
        raise ValueError('Extra seed range must be 101 <= FIRST <= LAST <= 2147483647')
    if root.exists():
        raise ValueError('Use a fresh output directory; preparation never overwrites a run')
    sys.path.insert(0, str(STUDY))
    import importlib.metadata
    for package, version in p['software']['python'].items():
        if importlib.metadata.version(package.replace('_','-')) != version:
            raise ValueError(f'Expected {package}=={version}')
    from reference import inputs, build, parameters, moving, np
    root.mkdir(parents=True)
    shutil.copyfile(STUDY/'split-rows.csv', root/'split-rows.csv')
    x,y,_,audit = inputs(root)
    if audit != p['audit']:
        raise ValueError('Inputs differ from frozen cohort')
    np.asarray(x,dtype='<f4').ravel(order='F').tofile(root/'x-float32.bin')
    np.asarray(y,dtype='<f4').tofile(root/'y-float32.bin')
    (root/'protocol.json').write_text(json.dumps(p,indent=2))
    for seed in range(first,last+1):
        m=build(seed,'mean')
        (root/f'initial-{seed}.json').write_text(json.dumps(dict(parameters=parameters(m),moving=moving(m))))
    extension=dict(first_seed=first,last_seed=last,base_protocol_sha256=sha(STUDY/'protocol.json'),
                   initial_sha256={str(s):sha(root/f'initial-{s}.json') for s in range(first,last+1)},
                   extension_code_sha256={f.name:sha(f) for f in sorted(HERE.iterdir()) if f.suffix in ('.py','.R')})
    (root/'extension.json').write_text(json.dumps(extension,indent=2))
    (root/'python-session.txt').write_text(sys.version+'\n'+subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    print(f'Prepared {first}–{last}; no training started.')

def verify(root):
    p=canonical()
    e=json.loads((root/'extension.json').read_text())
    if e['base_protocol_sha256'] != sha(STUDY/'protocol.json') or json.loads((root/'protocol.json').read_text()) != p:
        raise ValueError('Protocol mismatch')
    for name,expected in e['extension_code_sha256'].items():
        if sha(HERE/name)!=expected:raise ValueError(f'Extension code changed: {name}')
    for s,expected in e['initial_sha256'].items():
        if sha(root/f'initial-{s}.json')!=expected:raise ValueError(f'Initialization changed: seed {s}')
    if not (101 <= e['first_seed'] <= e['last_seed'] <= 2147483647):
        raise ValueError('Invalid extension range')
    if set(e['initial_sha256'])!={str(s) for s in range(e['first_seed'],e['last_seed']+1)}:
        raise ValueError('Initialization range mismatch')
    sys.path.insert(0,str(STUDY))
    from audit_split import audit
    audit(root)
    return p,e

def run(framework,reduction):
    parser=argparse.ArgumentParser(description=f'Run additional {framework}/{reduction} seeds using the frozen protocol')
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    root=args.output.resolve()
    _,e=verify(root)
    if framework=='ddesonn':
        check=json.loads((root/'native-equivalence-check.json').read_text())
        if check.get('passed') is not True or len(check.get('checks',[]))!=146 or any(c['maximum_relative_scaled_error']>2e-4 for c in check['checks']):
            raise ValueError('Native equivalence checks must pass before training')
    import fcntl
    with (root/f'{framework}-{reduction}.worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        tail=[str(root),reduction if framework=='keras' else ('TRUE' if reduction=='mean' else 'FALSE'),str(e['first_seed']),str(e['last_seed'])]
        if framework=='keras':cmd=[sys.executable,str(STUDY/'run_keras.py'),*tail]
        else:cmd=[os.environ.get('RSCRIPT','Rscript'),str(HERE/'run_native.R'),*tail]
        subprocess.run(cmd,cwd=REPO,check=True)

def export(root,destination):
    p,e=verify(root)
    records=[]
    for s in range(e['first_seed'],e['last_seed']+1):
        for f,r in CONDITIONS:
            path=root/f'{f}-{r}-{s}.json'
            x=json.loads(path.read_text())
            assert (x['seed'],x['framework'],x['reduction'])==(s,f,r),path
            assert x['epochs']==360 and x['optimizer_updates']==78840 and x['checkpoint']=='final',path
            assert x['split_sha256']==p['split_sha256'] and x['dataset_sha256']==p['dataset_sha256'],path
            assert len(x['metrics'])==3 and {m['split'] for m in x['metrics']}==set(p['rows']),path
            for m in x['metrics']:
                assert m['rows']==p['rows'][m['split']] and type(m['correct']) is int and 0<=m['correct']<=m['rows'],path
                assert abs(m['accuracy']-m['correct']/m['rows'])<1e-12,path
            if f=='ddesonn':
                assert x['average_gradients']==(r=='mean') and x['validation_epochs']==360 and x['validation_steps']==15840,path
                assert x['validation_batch_size']==16 and x['evaluation_batch_size']==32,path
                assert x['implementation']=='opt-in native original-Keras clean-split protocol adapter',path
            records.append((path,x))
    if destination.exists():raise ValueError('Export destination already exists')
    check=json.loads((root/'native-equivalence-check.json').read_text())
    assert check['passed'] is True and len(check['checks'])==146
    assert all(c['maximum_relative_scaled_error']<=2e-4 for c in check['checks'])
    names=['protocol.json','extension.json','python-session.txt','native-session.txt','native-equivalence-check.json','tensorflow-checks.json']
    for name in names:
        if not (root/name).is_file():raise ValueError(f'Missing evidence: {name}')
    destination.mkdir(parents=True)
    for name in names:
        shutil.copyfile(root/name,destination/name)
    for path,_ in records:shutil.copyfile(path,destination/path.name)
    with (destination/'per-seed-test-accuracy.csv').open('w',newline='') as stream:
        w=csv.writer(stream);w.writerow(['seed','framework','reduction','test_rows','correct','accuracy','epochs','optimizer_updates','checkpoint'])
        for _,x in records:
            m=next(m for m in x['metrics'] if m['split']=='test')
            w.writerow([x['seed'],x['framework'],x['reduction'],m['rows'],m['correct'],m['accuracy'],360,78840,'final'])
    print(f'Validated and exported {len(records)} fits. Original 100-seed cohort unchanged.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare');a.add_argument('output',type=Path);a.add_argument('first',type=int);a.add_argument('last',type=int)
    a=sub.add_parser('export');a.add_argument('output',type=Path);a.add_argument('destination',type=Path)
    a=parser.parse_args()
    if a.command=='prepare':prepare(a.output.resolve(),a.first,a.last)
    else:export(a.output.resolve(),a.destination.resolve())
