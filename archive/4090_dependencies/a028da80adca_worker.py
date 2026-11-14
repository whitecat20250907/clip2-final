"""Extend the pinned Y2-TF feature worker to public EVAL, with VALID controls."""
import argparse
import hashlib
import json
from pathlib import Path
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--role', required=True)
    a = ap.parse_args()
    base = Path(a.base)
    cfg = json.loads((base/'protocol.json').read_text())
    source = base/'original_worker.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == cfg['original_worker_SHA']
    for path in cfg.get('runtime_paths', []):
        sys.path.insert(0, path)
    rows = json.loads((base/'selection.json').read_text())
    assert len(rows) == 2191 and [r['split'] for r in rows[:2]] == ['valid']*2
    assert all(r['split'] == 'eval' for r in rows[2:])
    expected = {r['signal_id']: r for r in json.loads((base/'VALID_expected.json').read_text())}
    assert set(expected) == {r['signal_id'] for r in rows[:2]}
    proof = []

    def validate(row, features, activity):
        import numpy as np
        if row['split'] == 'eval':
            assert len(proof) == 2
            return
        old = expected[row['signal_id']]
        old_x = np.asarray(old['x'])
        assert old_x.shape == features.shape
        gap = float(np.abs(old_x-features).max())
        proof.append(dict(signal_id=row['signal_id'], maxgap=gap,
                          activity_gap=abs(old['active_fraction']-activity)))
        out = base/'tasks'/a.role
        (out/'VALID_control.json').write_text(json.dumps(proof, indent=2))
        assert gap <= 1e-6 and proof[-1]['activity_gap'] == 0., proof[-1]
        if len(proof) == 2:
            (out/'VALID_control_passed.json').write_text(json.dumps(dict(
                controls=proof, labels_read=False, training_updates=0), indent=2))

    text = source.read_text()
    patches = [
        ("Path(p['data_root'])/r['split']", "Path(p['data_roots'][r['split']])/r['split']"),
        ("if i%5==0:save(", "validate(r,feat,activity)\n        if i%5==0:save("),
        ("(out/'evidence.jsonl').open('w')", "(out/'evidence.jsonl').open('x')"),
    ]
    for old, new in patches:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    out = base/'tasks'/a.role
    out.mkdir(parents=True, exist_ok=True)
    (out/'engineering_patch.json').write_text(json.dumps(dict(
        original_SHA=cfg['original_worker_SHA'], patches=patches,
        adapted_SHA=hashlib.sha256(text.encode()).hexdigest(), model_math_changed=False), indent=2))
    exec(compile(text, str(source), 'exec'), dict(__name__='__main__', __file__=str(source), validate=validate))
    assert len(proof) == 2
    done = json.loads((out/'complete.json').read_text())
    done.update(EVAL_rows=2189, VALID_control_rows=2, training_updates=0)
    (out/'complete.json').write_text(json.dumps(done, indent=2))


if __name__ == '__main__':
    main()
