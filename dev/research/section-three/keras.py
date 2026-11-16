"""Section Three: execute the archived builder/compile/fit, with clean row IDs."""
import os
os.environ.update(TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1')
import ast, pathlib, random, sys
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter

# Limit worker resources without changing the archived NumPy adapter's shuffle/batches.
original_dataset = ArrayDataAdapter.get_tf_dataset
def limited_dataset(self):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return original_dataset(self).with_options(options)
ArrayDataAdapter.get_tf_dataset = limited_dataset

if len(sys.argv) not in (3,5):
    raise SystemExit('Usage: python keras.py OUTPUT mean|sum [FIRST LAST]')
root = pathlib.Path(sys.argv[1])
mode = sys.argv[2]
first,last = map(int,sys.argv[3:5]) if len(sys.argv)==5 else (1,10)
assert mode in ('mean','sum') and first >= 1 and last >= first
out = root / f'keras-{mode}-{first}-{last}.csv'
assert not out.exists(), 'Output exists; choose a new range/directory.'
archive = pathlib.Path(__file__).resolve().parents[3] / 'inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py'
tree = ast.parse(archive.read_text())
scope = {}
# Use original imports and constants, excluding paths/data loading and empty seed loop.
for node in tree.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)) or (
        isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets)
        and node.targets[0].id in {'LR','L1_LAMBDA','EPOCHS','BATCH_SIZE','BN_MOMENTUM','BN_EPS',
                                  'BN_BETA_INIT','BN_GAMMA_INIT','CUSTOM_SCALE','feature_cols','target_col'}):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<archived>', 'exec'), scope)
loop = next(n for n in tree.body if isinstance(n, ast.For))
builder = [n for n in loop.body if (
    isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)
    and n.targets[0].id in {'KERNEL_INIT','model'}) or (
    isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'compile')]
assert len(builder) == 3
# Preserve the archived optimizer, architecture, metrics and regularizer.
# Only the data-loss reduction changes. Kernel L1 remains added once per batch.
if mode == 'sum':
    compile_call = builder[-1].value
    loss_arg = next(k for k in compile_call.keywords if k.arg == 'loss')
    loss_arg.value = ast.parse("tf.keras.losses.BinaryCrossentropy(reduction='sum')", mode='eval').body
    ast.fix_missing_locations(builder[-1])
fit = next(n for n in loop.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
           and isinstance(n.value.func, ast.Attribute) and n.value.func.attr == 'fit')
data = pd.read_csv(root / 'data.csv')
assert not data.isna().any().any()
rows = pd.read_csv(root / 'split-rows.csv')
parts = {name: rows.loc[rows['split'] == name, 'row'].to_numpy()-1
         for name in ('train','validation','test')}
x = data[scope['feature_cols']].to_numpy()
y = data[scope['target_col']].to_numpy().astype(int)
scaler = StandardScaler().fit(x[parts['train']])
xs = {name: scaler.transform(x[idx]) for name, idx in parts.items()}
results = []
for seed in range(first,last+1):
    tf.keras.backend.clear_session()
    random.seed(seed); np.random.seed(seed); tf.keras.utils.set_random_seed(seed)
    scope.update(seed=seed, X_train=xs['train'], y_train=y[parts['train']],
                 X_val=xs['validation'], y_val=y[parts['validation']])
    exec(compile(ast.Module(body=builder, type_ignores=[]), '<archived-builder>', 'exec'), scope)
    model = scope['model']
    print(f'Section Three Keras {mode} seed {seed}: 360 epochs', flush=True)
    exec(compile(ast.Module(body=[fit], type_ignores=[]), '<archived-fit>', 'exec'), scope)
    assert len(model.history.history['loss']) == 360
    assert int(model.optimizer.iterations) == 360 * int(np.ceil(len(parts['train'])/16))
    for name, idx in parts.items():
        prob = model.predict(xs[name], verbose=0).ravel().astype(np.float64)
        accuracy = float(np.mean((prob >= .5) == y[idx]))
        results.append(dict(seed=seed, split=name, accuracy=accuracy, selected_epoch=360,
                            optimizer_updates=int(model.optimizer.iterations)))
    pd.DataFrame(results).to_csv(out, index=False)
    print(results[-1], flush=True)
