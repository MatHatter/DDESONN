"""Validate the corrected original-protocol run; never mix in Section Three results."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
from audit_split import audit as verify_split

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CONDITIONS = [('keras', 'mean'), ('ddesonn', 'sum'), ('ddesonn', 'mean'), ('keras', 'sum')]


def collect(root):
    protocol = json.loads((root / 'protocol.json').read_text())
    audit = verify_split(root)
    reference = REPO / 'inst/scripts/vsKeras/1000SEEDSRESULTSvsKeras/TestKeras_1000seeds.py'
    dataset = REPO / 'inst/extdata/heart_failure_clinical_records.csv'
    assert protocol['reference_sha256'] == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert protocol['dataset_sha256'] == hashlib.sha256(dataset.read_bytes()).hexdigest()
    metadata = json.loads((HERE / 'protocol.json').read_text())
    for name, expected in metadata['adapter_sha256'].items():
        assert hashlib.sha256((REPO / name).read_bytes()).hexdigest() == expected, name
    assert hashlib.sha256((root / 'tensorflow-checks.json').read_bytes()).hexdigest() == metadata['tensorflow_fixture_sha256']
    check = json.loads((root / 'native-equivalence-check.json').read_text())
    assert check['passed'] is True and len(check['checks']) == 146
    assert max(c['maximum_relative_scaled_error'] for c in check['checks']) <= check['tolerance']
    result = {}
    for framework, reduction in CONDITIONS:
        for path in sorted(root.glob(f'{framework}-{reduction}-*.json')):
            record = json.loads(path.read_text())
            seed = record['seed']
            assert record['framework'] == framework and record['reduction'] == reduction
            assert 1 <= seed <= 100 and path.name == f'{framework}-{reduction}-{seed}.json'
            assert record['epochs'] == 360 and record['optimizer_updates'] == 78840
            assert record['split_sha256'] == audit['split_sha256']
            assert record['dataset_sha256'] == audit['dataset_sha256']
            assert record['checkpoint'] == 'final'
            assert len(record['metrics']) == 3
            assert {r['split'] for r in record['metrics']} == {'train', 'validation', 'test'}
            for row in record['metrics']:
                assert row['rows'] == audit['rows'][row['split']]
                assert 0 <= row['correct'] <= row['rows']
                assert abs(row['accuracy'] - row['correct'] / row['rows']) < 1e-12
            if framework == 'ddesonn':
                assert record['validation_epochs'] == 360 and record['validation_steps'] == 15840
                assert record['validation_batch_size'] == 16 and record['evaluation_batch_size'] == 32
                assert record['average_gradients'] == (reduction == 'mean')
                assert record['implementation'] == 'opt-in native original-Keras clean-split protocol adapter'
            key = (seed, framework, reduction)
            assert key not in result
            result[key] = record
    return result


def write(root, result, complete=False):
    if complete:
        assert len(result) == 400
        assert all((seed, f, r) in result for seed in range(1, 101) for f, r in CONDITIONS)
    rows = []
    for key in sorted(result):
        record = result[key]
        test = next(m for m in record['metrics'] if m['split'] == 'test')
        rows.append(dict(seed=record['seed'], framework=record['framework'], reduction=record['reduction'],
                         test_rows=test['rows'], correct=test['correct'], accuracy=test['accuracy'],
                         epochs=record['epochs'], optimizer_updates=record['optimizer_updates'], checkpoint='final'))
    target = HERE / 'per-seed-test-accuracy.csv'
    tmp = target.with_suffix(f'.tmp.{os.getpid()}')
    with tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['seed', 'framework', 'reduction', 'test_rows', 'correct',
                                                   'accuracy', 'epochs', 'optimizer_updates', 'checkpoint'])
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(target)
    if not complete:
        return
    lines = ['# Original training settings, zero predictor overlap — 100 seeds per condition', '',
             'All 400 fresh fits completed 360 epochs and 78,840 batch-16 updates each.',
             'Only the split is changed: shared group-disjoint partitions of 3,500 / 698 / 802 rows.',
             'Raw predictor overlap and actual float32 model-input overlap are zero for all three partition pairs.', '',
             '| Seed | Original Keras mean | DDESONN FALSE (sum) | DDESONN TRUE (mean) | Keras sum |',
             '|---|---:|---:|---:|---:|']
    values = {c: [] for c in CONDITIONS}
    for seed in range(1, 101):
        formatted = []
        for condition in CONDITIONS:
            test = next(m for m in result[(seed, *condition)]['metrics'] if m['split'] == 'test')
            value = 100 * test['accuracy']; values[condition].append(value); formatted.append(f'{value:.2f}%')
        lines.append(f'| {seed} | ' + ' | '.join(formatted) + ' |')
    lines += ['| **Mean** | ' + ' | '.join(f'{statistics.mean(values[c]):.2f}%' for c in CONDITIONS) + ' |',
              '| SD (percentage points) | ' + ' | '.join(f'{statistics.stdev(values[c]):.2f}' for c in CONDITIONS) + ' |', '',
              'DDESONN uses the new opt-in native float32 original-Keras protocol adapter,',
              'not its unchanged default training path. All four conditions share original',
              'Keras per-seed initial tensors. Shuffling and dropout retain the same',
              'distributions but use framework-native RNG engines; the runs are not bitwise identical.', '',
              'Original batch size 16, normalization, architecture, initialization, BN, dropout,',
              'AdaGrad, regularization, 360 epochs and final-checkpoint selection are retained.',
              'The only protocol change is the shared split required to remove identical predictor overlap.',
              'No overlapping-split or differently configured earlier results are reused.',
              'These are 100 training seeds on one fixed clean partition, not independent patient-cohort replications.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = collect(args.root)
    write(args.root, result, complete=not args.partial)
    print(f'Validated {len(result)}/400 original-protocol fits')
