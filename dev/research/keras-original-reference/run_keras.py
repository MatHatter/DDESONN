import csv
import fcntl
import json
from pathlib import Path
import sys
from reference import build, inputs, SCOPE, FIT, ast, np, tf

root, reduction = Path(sys.argv[1]), sys.argv[2]
first, last = map(int, sys.argv[3:5])
x, y, _ = inputs()
def run_seed(seed):
    out = root / f'keras-{reduction}-{seed}.json'
    if out.exists():
        return
    model = build(seed, reduction)
    SCOPE.update(model=model, X_train=x[:3500], y_train=y[:3500],
                 X_val=x[3500:4250], y_val=y[3500:4250])
    print(f'Keras {reduction} seed {seed}: starting 360 epochs', flush=True)
    exec(compile(ast.Module(body=[FIT], type_ignores=[]), '<archived-fit>', 'exec'), SCOPE)
    assert len(model.history.history['loss']) == 360
    updates = int(model.optimizer.iterations)
    assert updates == 78840
    metrics = []
    for name, start, end in [('train', 0, 3500), ('validation', 3500, 4250), ('test', 4250, 5000)]:
        probability = model.predict(x[start:end], verbose=0).ravel()
        reported_loss, reported_accuracy = model.evaluate(x[start:end], y[start:end], verbose=0)
        prediction = probability > .5
        correct = int(np.sum(prediction == y[start:end]))
        assert abs(reported_accuracy - correct / (end-start)) < 1e-6
        assert np.isfinite(probability).all()
        metrics.append(dict(split=name, rows=end-start, correct=correct,
                            accuracy=correct/(end-start), keras_reported_accuracy=float(reported_accuracy)))
    payload = dict(seed=seed, framework='keras', reduction=reduction, epochs=360,
                   optimizer_updates=updates, checkpoint='final', metrics=metrics)
    tmp = out.with_suffix('.tmp'); tmp.write_text(json.dumps(payload, indent=2)); tmp.replace(out)
    print(f'Keras {reduction} seed {seed}: test {metrics[-1]["accuracy"]:.6f}', flush=True)

for seed in range(first, last + 1):
    with (root / f'keras-{reduction}-{seed}.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            continue
        run_seed(seed)
