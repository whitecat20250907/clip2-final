"""Original Y2 signal math; EVAL routing and two exact VALID controls only."""
import argparse, hashlib, json
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args()
    base=Path(a.base);cfg=json.loads((base/'protocol.json').read_text());source=base/'original_worker.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['original_worker_SHA']
    expected={r['signal_id']:r for r in json.loads((base/'VALID_expected.json').read_text())};proof=[]
    rows=json.loads((base/'selection.json').read_text())
    assert len(rows)==2191 and set(expected)=={r['signal_id']for r in rows[:2]}
    assert [r['split']for r in rows[:2]]==['valid']*2 and all(r['split']=='eval'for r in rows[2:])
    def validate(row,wp):
        import numpy as np
        if row['split']=='eval':assert len(proof)==2;return
        x=np.asarray(wp);old=np.asarray(expected[row['signal_id']]['x']);assert x.shape==old.shape
        gap=float(abs(x-old).max());proof.append(dict(signal_id=row['signal_id'],maxgap=gap))
        out=base/'tasks'/a.role;(out/'VALID_control.json').write_text(json.dumps(proof,indent=2))
        assert gap<=1e-6,proof[-1]
        if len(proof)==2:(out/'VALID_control_passed.json').write_text(json.dumps(dict(cases=proof,training_updates=0,labels_read=False),indent=2))
    text=source.read_text();patches=[("Path(p['data_root'])/r['split']","Path(p['data_roots'][r['split']])/r['split']"),("if i%10==0:save(","validate(r,wp)\n        if i%10==0:save("),("(out/'evidence.jsonl').open('w')","(out/'evidence.jsonl').open('x')")]
    for old,new in patches:assert text.count(old)==1; text=text.replace(old,new)
    out=base/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True)
    (out/'engineering_patch.json').write_text(json.dumps(dict(original_SHA=cfg['original_worker_SHA'],patches=patches,adapted_SHA=hashlib.sha256(text.encode()).hexdigest(),model_math_changed=False),indent=2))
    exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),validate=validate))
    assert len(proof)==2
    r=json.loads((out/'complete.json').read_text());r.update(EVAL_rows=2189,VALID_controls=2,training_updates=0);(out/'complete.json').write_text(json.dumps(r,indent=2))

if __name__=='__main__':main()
