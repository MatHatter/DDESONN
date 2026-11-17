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

def inputs(root):
    source = REPO / 'inst/extdata/heart_failure_clinical_records.csv'
    data = pd.read_csv(source).dropna()
    assert len(data) == 5000
    rows = pd.read_csv(root / 'split-rows.csv')
    assert len(rows) == 5000 and sorted(rows['row']) == list(range(1, 5001))
    assert set(rows['split']) == {'train', 'validation', 'test'}
    indices = {name: rows.loc[rows['split'] == name, 'row'].to_numpy(dtype=int) - 1
               for name in ('train', 'validation', 'test')}
    raw = data[SCOPE['feature_cols']].to_numpy()
    scaler = StandardScaler().fit(raw[indices['train']])
    transformed = scaler.transform(raw)
    raw_keys = {name: {tuple(r) for r in raw[idx]} for name, idx in indices.items()}
    model_keys = {name: {tuple(r) for r in transformed[idx].astype(np.float32)}
                  for name, idx in indices.items()}
    pairs = [('train', 'validation'), ('train', 'test'), ('validation', 'test')]
    overlap = {f'{a}_{b}': len(raw_keys[a] & raw_keys[b]) for a, b in pairs}
    model_overlap = {f'{a}_{b}': len(model_keys[a] & model_keys[b]) for a, b in pairs}
    assert not any(overlap.values()), f'Raw predictor overlap: {overlap}'
    assert not any(model_overlap.values()), f'Float32 input overlap: {model_overlap}'
    ordered = np.concatenate(list(indices.values()))
    x = transformed[ordered]
    y = data[SCOPE['target_col']].to_numpy().astype(int)[ordered]
    sizes = {name: len(idx) for name, idx in indices.items()}
    audit = dict(rows=sizes, raw_predictor_overlap=overlap, float32_input_overlap=model_overlap,
                 predictor_groups={name: len(keys) for name, keys in raw_keys.items()},
                 split_sha256=hashlib.sha256((root / 'split-rows.csv').read_bytes()).hexdigest(),
                 dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                 standardized_float32_sha256=hashlib.sha256(np.asarray(x, dtype='<f4').ravel(order='F').tobytes()).hexdigest(),
                 labels_sha256=hashlib.sha256(np.asarray(y, dtype='<f4').tobytes()).hexdigest())
    return x, y, source, audit


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
    x, y, source, audit = inputs(root)
    np.asarray(x, dtype='<f4').ravel(order='F').tofile(root / 'x-float32.bin')
    np.asarray(y, dtype='<f4').tofile(root / 'y-float32.bin')
    for seed in range(1, 101):
        model = build(seed, 'mean')
        payload = dict(parameters=parameters(model), moving=moving(model))
        (root / f'initial-{seed}.json').write_text(json.dumps(payload))
    metadata = dict(reference_sha256=hashlib.sha256(ARCHIVE.read_bytes()).hexdigest(),
                    dataset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    rows=audit['rows'], split_sha256=audit['split_sha256'], audit=audit, batch_size=16,
                    epochs=360, optimizer_updates=78840, tensorflow=tf.__version__,
                    optimizer=build(1, 'mean').optimizer.get_config(),
                    initialization='Shared original Keras per-seed initial tensors',
                    native_precision='float32',
                    stochastic_draws='Framework-native shuffled batches and dropout; same distributions, different RNG engines')
    (root / 'overlap-audit.json').write_text(json.dumps(audit, indent=2))
    (root / 'protocol.json').write_text(json.dumps(metadata, indent=2))
