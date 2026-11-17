"""Validate and summarize Section Three, retaining seeds 1–10 from the prior PR.

Usage: python dev/research/section-three/summarize.py OUTPUT_ROOT [--partial]
Only compact accuracy results are written into the repository.
"""
import argparse
import csv
import math
from pathlib import Path
import statistics

CONDITIONS = ('ddesonn_sum', 'ddesonn_mean', 'keras_mean', 'keras_sum')
SIZES = {'train': 3500, 'validation': 698, 'test': 802}
HERE = Path(__file__).resolve().parent


def read_csv(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def collect(root):
    data = read_csv(root / 'data.csv')
    rows = read_csv(root / 'split-rows.csv')
    assert len(data) == len(rows) == 5000
    assert sorted(int(row['row']) for row in rows) == list(range(1, 5001))
    groups = {}
    for name, size in SIZES.items():
        selected = [data[int(row['row']) - 1] for row in rows if row['split'] == name]
        assert len(selected) == size
        groups[name] = {tuple(value for key, value in row.items() if key != 'DEATH_EVENT')
                        for row in selected}
    assert all(not groups[a] & groups[b] for a, b in
               [('train', 'validation'), ('train', 'test'), ('validation', 'test')])
    result = {}
    for row in read_csv(HERE / 'seed-1-10-retained.csv'):
        seed = int(row['seed'])
        condition = row['condition']
        correct = int(row['correct'])
        assert 1 <= seed <= 10 and condition in CONDITIONS
        assert int(row['test_rows']) == 802
        # With 802 test rows, each published two-decimal percentage identifies
        # exactly one integer correct count. No original model/epoch is inferred.
        matches = [count for count in range(803)
                   if f'{100 * count / 802:.2f}' == row['reported_accuracy_percent']]
        assert matches == [correct]
        key = (seed, condition)
        assert key not in result
        result[key] = dict(seed=seed, condition=condition, accuracy=correct / 802,
                           test_rows=802, correct=correct, requested_epochs=360,
                           selected_epoch='', source='retained ten-seed table')
    assert len(result) == 40
    paths = list(root.glob('ddesonn-*-*/metrics.csv')) + list(root.glob('keras-*-*.csv'))
    for path in sorted(paths):
        records = read_csv(path)
        seeds = sorted({int(row['seed']) for row in records})
        for seed in seeds:
            batch = [row for row in records if int(row['seed']) == seed]
            assert 11 <= seed <= 100
            assert sorted(row['split'] for row in batch) == sorted(SIZES)
            if path.name == 'metrics.csv':
                flags = {row['average_gradients'] for row in batch}
                assert len(flags) == 1 and flags <= {'TRUE', 'FALSE'}
                condition = 'ddesonn_mean' if flags == {'TRUE'} else 'ddesonn_sum'
                for row in batch:
                    assert int(row['requested_epochs']) == 360
                    assert 1 <= int(row['selected_epoch']) <= 360
                    assert int(row['rows']) == SIZES[row['split']]
                    for metric in ['accuracy', 'log_loss', 'brier', 'group_weighted_accuracy']:
                        assert math.isfinite(float(row[metric]))
                    assert sum(int(row[k]) for k in ['TP', 'TN', 'FP', 'FN']) == int(row['rows'])
            else:
                condition = 'keras_mean' if path.name.startswith('keras-mean-') else 'keras_sum'
                assert path.name.startswith(('keras-mean-', 'keras-sum-'))
                for row in batch:
                    assert int(row['selected_epoch']) == 360
                    assert int(row['optimizer_updates']) == 360 * math.ceil(3500 / 16)
            test = next(row for row in batch if row['split'] == 'test')
            accuracy = float(test['accuracy'])
            assert math.isfinite(accuracy) and 0 <= accuracy <= 1
            correct = round(accuracy * 802)
            assert math.isclose(accuracy, correct / 802, abs_tol=1e-12)
            key = (seed, condition)
            assert key not in result, f'Duplicate result: {key}'
            result[key] = dict(seed=seed, condition=condition, accuracy=accuracy,
                               test_rows=802, correct=correct, requested_epochs=360,
                               selected_epoch=int(test['selected_epoch']),
                               source='new 360-epoch fit')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--partial', action='store_true')
    args = parser.parse_args()
    result = collect(args.root)
    complete = [seed for seed in range(1, 101)
                if all((seed, condition) in result for condition in CONDITIONS)]
    if not args.partial:
        assert len(result) == 400 and complete == list(range(1, 101)), \
            f'Only {len(result)}/400 fits and {len(complete)}/100 complete seed quartets'
    with (HERE / 'per-seed-test-accuracy.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=next(iter(result.values())).keys())
        writer.writeheader()
        writer.writerows(result[key] for key in sorted(result, key=lambda k: (k[0], CONDITIONS.index(k[1]))))
    if len(complete) != 100:
        print(f'{len(result)}/400 fits retained; {len(complete)}/100 complete seed quartets')
        return
    lines = ['# Section Three — 100 seeds, sum and mean', '',
             '360 epochs per fit; fixed .5 test threshold; shared clean partition.', '',
             '| Seed | DDESONN FALSE: sum | DDESONN TRUE: mean | Keras mean | Keras sum |',
             '|---|---:|---:|---:|---:|']
    for seed in complete:
        values = ' | '.join(f'{100 * result[seed, c]["accuracy"]:.2f}%' for c in CONDITIONS)
        lines.append(f'| {seed} | {values} |')
    values = [[100 * result[seed, c]['accuracy'] for seed in complete] for c in CONDITIONS]
    lines += ['| **Mean** | ' + ' | '.join(f'{statistics.mean(v):.2f}%' for v in values) + ' |',
              '| SD (percentage points) | ' + ' | '.join(f'{statistics.stdev(v):.2f}' for v in values) + ' |', '',
              'Seeds 1–10 are retained unchanged from the prior ten-seed table; seeds 11–100',
              'are 360 new fits (90 per condition). The retained two-decimal percentages',
              'uniquely identify integer correct counts out of 802 test rows; those counts',
              'recover their exact accuracies for aggregation. Original selected epochs for',
              'seeds 1–10 are unavailable and are left blank in the CSV.', '',
              'All 400 fits completed 360 epochs. Exact predictor overlap across splits is zero.',
              'Keras retains batches of 16 and its final checkpoint; DDESONN uses all 3,500',
              'training rows and its best validation checkpoint. Other archived framework',
              'differences remain. These are 100 training seeds on one clean partition,',
              'not 100 independent data splits or a full framework-parity study.', '']
    (HERE / 'RESULTS.md').write_text('\n'.join(lines))
    print('Validated all 400 fits; wrote 100-seed table and accuracy CSV')


if __name__ == '__main__':
    main()
