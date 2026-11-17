"""Execute the original archived Keras protocol; only loss reduction may vary."""
import os
os.environ.update(TF_NUM_INTRAOP_THREADS='1', TF_NUM_INTEROP_THREADS='1', OMP_NUM_THREADS='1')
import ast
import hashlib
import json
from pathlib import Path
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.src.trainers.data_adapters.array_data_adapter import ArrayDataAdapter

REPO = Path(__file__).resolve().parents[3]
ARCHIVE = REPO / 'inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py'
original_dataset = ArrayDataAdapter.get_tf_dataset

def limited_dataset(self):
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return original_dataset(self).with_options(options)

ArrayDataAdapter.get_tf_dataset = limited_dataset
TREE = ast.parse(ARCHIVE.read_text())
SCOPE = {}
CONSTANTS = {'LR', 'L1_LAMBDA', 'EPOCHS', 'BATCH_SIZE', 'BN_MOMENTUM', 'BN_EPS',
             'BN_BETA_INIT', 'BN_GAMMA_INIT', 'CUSTOM_SCALE', 'feature_cols', 'target_col'}
for node in TREE.body:
    if isinstance(node, (ast.Import, ast.ImportFrom)) or (
        isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id in CONSTANTS):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<archived>', 'exec'), SCOPE)
LOOP = next(node for node in TREE.body if isinstance(node, ast.For))
BUILDER = [node for node in LOOP.body if (
    isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
    and node.targets[0].id in {'KERNEL_INIT', 'model'}) or (
    isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'compile')]
assert len(BUILDER) == 3
FIT = next(node for node in LOOP.body if isinstance(node, ast.Expr)
           and isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute)
           and node.value.func.attr == 'fit')

def build(seed, reduction):
    assert reduction in ('mean', 'sum')
    tf.keras.backend.clear_session()
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    SCOPE['seed'] = seed
    SCOPE.setdefault('X_train', np.empty((0, 12)))
    exec(compile(ast.Module(body=BUILDER, type_ignores=[]), '<archived-builder>', 'exec'), SCOPE)
    model = SCOPE['model']
    if reduction == 'sum':
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=SCOPE['LR']),
                      loss=tf.keras.losses.BinaryCrossentropy(reduction='sum'), metrics=['accuracy'])
    return model

def inputs():
    source = REPO / 'inst/extdata/heart_failure_clinical_records.csv'
    data = pd.read_csv(source).dropna()
    assert len(data) == 5000
    x = data[SCOPE['feature_cols']].to_numpy()
    y = data[SCOPE['target_col']].to_numpy().astype(int)
    scaler = StandardScaler().fit(x[:3500])
    return scaler.transform(x), y, source

def tensor(value):
    a = np.asarray(value, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    return dict(shape=list(a.shape), values=a.ravel(order='F').tolist())

def parameters(model):
    return [tensor(v.numpy()) for v in model.trainable_variables]

def moving(model):
    return [tensor(v.numpy()) for layer in (model.layers[1], model.layers[4])
            for v in (layer.moving_mean, layer.moving_variance)]

def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    x, y, source = inputs()
    np.asarray(x, dtype='<f4').ravel(order='F').tofile(root / 'x-float32.bin')
    np.asarray(y, dtype='<f4').tofile(root / 'y-float32.bin')
    for seed in range(1, 101):
        model = build(seed, 'mean')
        payload = dict(parameters=parameters(model), moving=moving(model))
        (root / f'initial-{seed}.json').write_text(json.dumps(payload))
    metadata = dict(reference_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    rows=dict(train=3500, validation=750, test=750), batch_size=16,
                    epochs=360, optimizer_updates=78840, tensorflow=tf.__version__,
                    optimizer=build(1, 'mean').optimizer.get_config(),
                    initialization='Shared original Keras per-seed initial tensors',
                    native_precision='float32',
                    stochastic_draws='Framework-native shuffled batches and dropout; same distributions, different RNG engines')
    (root / 'protocol.json').write_text(json.dumps(metadata, indent=2))
