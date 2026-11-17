import csv
import fcntl
import json
from pathlib import Path
import sys
from reference import build, inputs, SCOPE, FIT, ast, np, tf
from audit_split import audit as verify_split

root, reduction = Path(sys.argv[1]), sys.argv[2]
first, last = map(int, sys.argv[3:5])
verified = verify_split(root)
x, y, _, audit = inputs(root)
assert audit == verified
protocol = json.loads((root / 'protocol.json').read_text())
assert audit == protocol['audit']
n_train, n_val = audit['rows']['train'], audit['rows']['validation']
assert audit['standardized_float32_sha256'] == __import__('hashlib').sha256((root / 'x-float32.bin').read_bytes()).hexdigest()
def run_seed(seed):
    out = root / f'keras-{reduction}-{seed}.json'
    if out.exists():
        return
    model = build(seed, reduction)
    SCOPE.update(model=model, X_train=x[:n_train], y_train=y[:n_train],
                 X_val=x[n_train:n_train+n_val], y_val=y[n_train:n_train+n_val])
    print(f'Keras {reduction} seed {seed}: starting 360 epochs', flush=True)
    exec(compile(ast.Module(body=[FIT], type_ignores=[]), '<archived-fit>', 'exec'), SCOPE)
    assert len(model.history.history['loss']) == len(model.history.history['val_loss']) == 360
    updates = int(model.optimizer.iterations)
    assert updates == 78840
    metrics = []
    for name, start, end in [('train', 0, n_train), ('validation', n_train, n_train+n_val), ('test', n_train+n_val, 5000)]:
        probability = model.predict(x[start:end], verbose=0).ravel()
        reported_loss, reported_accuracy = model.evaluate(x[start:end], y[start:end], verbose=0)
        prediction = probability > .5
        correct = int(np.sum(prediction == y[start:end]))
        assert abs(reported_accuracy - correct / (end-start)) < 1e-6
        assert np.isfinite(probability).all()
        metrics.append(dict(split=name, rows=end-start, correct=correct,
                            accuracy=correct/(end-start), keras_reported_accuracy=float(reported_accuracy)))
    payload = dict(seed=seed, framework='keras', reduction=reduction, epochs=360,
                   optimizer_updates=updates, checkpoint='final', metrics=metrics, split_sha256=audit['split_sha256'], dataset_sha256=audit['dataset_sha256'])
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(out)
    print(f'Keras {reduction} seed {seed}: test {metrics[-1]["accuracy"]:.6f}', flush=True)

for seed in range(first, last + 1):
    with (root / f'keras-{reduction}-{seed}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        run_seed(seed)
