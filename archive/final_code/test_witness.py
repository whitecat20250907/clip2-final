"""Extend exact original blind MF/MOSS workers to fixed EVAL IDs only.

The source edits below change input/output routing and insert VALID controls.
The original prompts, decoding, native EOS and token scores are untouched.
"""
import argparse, hashlib, json, sys
from pathlib import Path

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base',required=True);ap.add_argument('--role',required=True);a=ap.parse_args()
    base=Path(a.base);p=json.loads((base/'protocol.json').read_text());source=base/'original_worker.py'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==p['original_worker_SHA']
    text=source.read_text();patches=[]
    def replace(old,new):
        nonlocal text
        assert text.count(old)==1,old
        text=text.replace(old,new);patches.append(dict(before=old,after=new))
    (base/'tasks').mkdir(exist_ok=True)
    replace('d=b/a.role;',"d=b/'tasks'/a.role;")
    if p['family']=='MF':
        replace("row['split'] in ['train','valid']","row['split'] in ['train','valid','eval']")
        replace("root=Path(p['data_root'])/row['split']","root=Path(p['data_roots'][row['split']])/row['split']")
        replace('completed+=1;save(', 'validate_row(d,row);completed+=1;save(')
    else:
        assert p['family']=='MOSS'
        replace("Path(p['data_root'])/row.get('split','train')","Path(p['data_roots'][row['split']])/row['split']")
        replace("  save(d/'progress.json'","  validate_row(d,row)\n  save(d/'progress.json'")
    original=json.loads((base/'VALID_expected.json').read_text());expected={(r['signal_id'],r['stream']):r for r in original}
    rows=p['parts'][a.role];assert [r['split']for r in rows[:2]]==['valid']*2
    assert len(rows)==2191 and all(r['split']=='eval'for r in rows[2:])
    assert len({r['signal_id']for r in rows})==2191
    controls=[]
    def validate_row(out,row):
        if row['split']=='eval':
            assert len(controls)==8
            return
        import numpy as np
        records=[json.loads(line)for line in(out/'evidence.jsonl').read_text().splitlines()]
        current=[r for r in records if r['signal_id']==row['signal_id']];assert len(current)==4
        for record in current:
            old=expected[record['signal_id'],record['stream']]
            identical={k:record[k]==old[k]for k in ['transcription','token_ids','recognized_words','word_token_counts','generation_capped']}
            gaps={k:float(np.max(np.abs(np.asarray(record[k])-np.asarray(old[k]))))if len(old[k])else 0. for k in ['token_logprob','word_logprob_sum','word_logprob_mean']}
            proof=dict(signal=row['signal_id'],stream=record['stream'],identical=identical,maxgaps=gaps)
            controls.append(proof);(out/'VALID_control.json').write_text(json.dumps(controls,indent=2))
            assert all(identical.values())and max(gaps.values())<=1e-6,proof
        if len(controls)==8:(out/'VALID_control_passed.json').write_text(json.dumps(dict(cases=controls,training_updates=0,labels_read=False),indent=2))
    job=base/'tasks'/a.role;job.mkdir(exist_ok=True)
    (job/'engineering_patch.json').write_text(json.dumps(dict(original_worker_SHA=p['original_worker_SHA'],patches=patches,adapted_source_SHA=hashlib.sha256(text.encode()).hexdigest(),model_math_changed=False),indent=2))
    scope=dict(__name__='__main__',__file__=str(source),validate_row=validate_row)
    exec(compile(text,str(source),'exec'),scope)
    assert len(controls)==8
    receipt=json.loads((job/'complete.json').read_text());receipt.update(EVAL_rows=2189,VALID_controls=2,training_updates=0,original_worker_SHA=p['original_worker_SHA'])
    (job/'complete.json').write_text(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
