"""Independent, fail-closed raw-row and actual model-input overlap audit."""
import csv
import hashlib
import json
from pathlib import Path
import struct

PARTS = ('train', 'validation', 'test')
REPO = Path(__file__).resolve().parents[3]

def audit(root):
    source = REPO / 'inst/extdata/heart_failure_clinical_records.csv'
    with source.open(newline='') as f:
        data = list(csv.DictReader(f))
    with (root / 'split-rows.csv').open(newline='') as f:
        assignment = list(csv.DictReader(f))
    assert len(data) == len(assignment) == 5000
    assert sorted(int(r['row']) for r in assignment) == list(range(1, 5001))
    assert {r['split'] for r in assignment} == set(PARTS)
    features = [key for key in data[0] if key != 'DEATH_EVENT']
    assert len(features) == 12
    ids = {name: [int(r['row']) - 1 for r in assignment if r['split'] == name] for name in PARTS}
    raw = {name: {tuple(float(data[i][key]) for key in features) for i in indices}
           for name, indices in ids.items()}
    values = struct.unpack('<60000f', (root / 'x-float32.bin').read_bytes())
    labels = struct.unpack('<5000f', (root / 'y-float32.bin').read_bytes())
    model = {}; offset = 0
    for name in PARTS:
        n = len(ids[name])
        model[name] = {tuple(values[i + 5000*j] for j in range(12)) for i in range(offset, offset+n)}
        assert [float(data[i]['DEATH_EVENT']) for i in ids[name]] == list(labels[offset:offset+n])
        offset += n
    pairs = [('train', 'validation'), ('train', 'test'), ('validation', 'test')]
    overlap = {f'{a}_{b}': len(raw[a] & raw[b]) for a, b in pairs}
    model_overlap = {f'{a}_{b}': len(model[a] & model[b]) for a, b in pairs}
    assert not any(overlap.values()), f'Raw predictor overlap: {overlap}'
    assert not any(model_overlap.values()), f'Model-input overlap: {model_overlap}'
    expected = json.loads((root / 'protocol.json').read_text())['audit']
    assert expected['rows'] == {name: len(ids[name]) for name in PARTS}
    assert expected['raw_predictor_overlap'] == overlap
    assert expected['float32_input_overlap'] == model_overlap
    for filename, field in [('split-rows.csv', 'split_sha256'), ('x-float32.bin', 'standardized_float32_sha256'),
                            ('y-float32.bin', 'labels_sha256')]:
        assert hashlib.sha256((root / filename).read_bytes()).hexdigest() == expected[field], filename
    assert hashlib.sha256(source.read_bytes()).hexdigest() == expected['dataset_sha256']
    return expected

if __name__ == '__main__':
    import sys
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
