"""Label-free export contracts. This module has no model or network dependency."""
import csv
import hashlib
import json
import re
from pathlib import Path
import numpy as np

SIDE = ('confidence_v3', 'confidence_v2', 'wer', 'extra_wer', 'candidate', 'paired')
DIMS = (12, 12, 1, 4, 1, 24)
ARMS = ('baseline', 'score_only', 'word_context')

def require(ok, message):
    if not ok: raise ValueError(message)

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
        separators=(',', ':')).encode()).hexdigest()

def write(path, value):
    path = Path(path); tmp = path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)

def public_records(records, count=2117):
    require(isinstance(records, list) and len(records)==count, 'Wrong official validation row count')
    forbidden = {'correctness', 'phoneme_correctness', 'response', 'original_response', 'hits', 'phoneme_hits'}
    ids=[]
    for row in records:
        require(not forbidden.intersection(row), 'Validation metadata contains listener targets')
        sid=row.get('signal'); ids.append(sid)
        require(isinstance(sid,str) and bool(re.fullmatch(r'[A-Za-z0-9_-]+',sid)), 'Unsafe/invalid signal ID')
        for field in ('prompt','original_prompt'):
            require(isinstance(row.get(field),str) and bool(row[field].strip()), 'Missing public '+field+'; do not invent lyrics')
    require(len(set(ids))==len(ids), 'Duplicate validation IDs')
    return ids

def normalized_side(raw, norms):
    require(set(norms)=={'confidence_v3','confidence_v2','candidate','paired'}, 'Unexpected FIT normalizer keys')
    n=len(raw['confidence_v3']); parts=[]
    for key, width in zip(SIDE,DIMS):
        x=np.asarray(raw[key],np.float32)
        require(x.shape==(n,width) and np.isfinite(x).all(), 'Bad feature '+key)
        if key in norms:
            mean=np.asarray(norms[key]['mean'],np.float32);std=np.asarray(norms[key]['std'],np.float32)
            require(mean.shape==std.shape==(width,) and np.isfinite(mean).all() and np.isfinite(std).all() and (std>0).all(), 'Invalid saved normalizer')
            x=(x-mean)/std
        parts.append(x)
    result=np.concatenate(parts,1)
    require(np.isfinite(result).all(), 'Nonfinite normalized input')
    return result

def table(path, field, width, ids):
    found={}; required=set(ids)
    with Path(path).open(encoding='utf-8-sig') as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); sid=r['signal']
            if sid not in required: continue
            x=np.asarray(r[field],np.float32).reshape(-1)
            require(sid not in found and x.shape==(width,) and np.isfinite(x).all(), 'Duplicate/bad cached feature '+str(path))
            found[sid]=x
    require(set(found)==required, 'Incomplete cached feature '+str(path))
    return np.stack([found[s] for s in ids])

def assemble_paired(acoustic, strict, likelihood):
    n=len(acoustic)
    require(acoustic.shape==(n,20) and strict.shape==(n,12) and likelihood.shape==(n,4), 'Paired input shape mismatch')
    result=np.zeros((n,24),np.float32)
    level=[0,1,2,3,10,11,12,13]; other=[4,5,6,7,8,9,14,15,16,17,18,19]
    result[:,level]=acoustic[:,level]; result[:,other]=strict; result[:,20:]=likelihood
    return result

def export_csv(path, ids, pred):
    pred=np.asarray(pred,np.float64)
    require(pred.shape==(len(ids),) and len(set(ids))==len(ids), 'CSV ID/count mismatch')
    require(np.isfinite(pred).all() and ((pred>=0)&(pred<=1)).all(), 'Invalid prediction; never silently clip')
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('w',encoding='utf-8',newline='') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerows((s,format(float(p)*100,'.10f')) for s,p in zip(ids,pred))
    with tmp.open(newline='',encoding='utf-8') as f: rows=list(csv.reader(f))
    require([r[0] for r in rows]==ids and all(len(r)==2 for r in rows), 'CSV replay failed')
    require(np.max(np.abs(np.array([float(r[1]) for r in rows])/100-pred))<=1e-11, 'Percentage conversion failed')
    if path.exists():
        require(path.read_bytes()==tmp.read_bytes(),'Refusing to overwrite different CSV')
        tmp.unlink()
    else: tmp.replace(path)
    return dict(path=str(path),sha256=sha(path),rows=len(ids),header=False,scale='0_to_100')
