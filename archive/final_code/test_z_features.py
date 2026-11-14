"""Routing and original-output controls for frozen Z2/Z8/Z9 extractors."""
import argparse,hashlib,json,sys
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();b=Path(a.base);cfg=json.loads((b/'protocol.json').read_text());source=b/'original_worker.py';assert hashlib.sha256(source.read_bytes()).hexdigest()==cfg['original_worker_SHA'];expected=json.loads((b/'VALID_expected.json').read_text());route=cfg['route'];key='audio_key'if route=='Z9'else'text_key';expected={r[key]:r for r in expected};proof=[];out=b/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True)
 def validate(row,actual):
  import numpy as np
  if row[key]not in expected:assert len(proof)==2;return
  old=expected[row[key]];gaps={}
  fields={'Z2':['x','surface'],'Z8':['surface_target_logp','sentence_PLL'],'Z9':['x','singing','ear_difference']}[route]
  for field in fields:
   x=np.asarray(actual[field]);y=np.asarray(old[field]);assert x.shape==y.shape;gaps[field]=float(np.max(abs(x-y)))
  if route=='Z8':
   assert actual['pieces']==old['pieces']
   # Original get_vocab() order is randomized. Equal-score top-k ties may
   # reorder word identities or exchange a tied token at rank 64. Preserve
   # the original worker; require its complete numerical score list exactly.
   from collections import Counter
   tie_changes=[]
   for position,(aa,bb) in enumerate(zip(actual['alternatives'],old['alternatives'])):
    assert len(aa)==len(bb)
    gaps['alternative_logp_'+str(position)]=max([abs(x[1]-y[1])for x,y in zip(aa,bb)]+[0.])
    ca=Counter(map(tuple,aa));cb=Counter(map(tuple,bb));added=list((ca-cb).elements());removed=list((cb-ca).elements())
    assert sorted(p for w,p in added)==sorted(p for w,p in removed)
    if added:
     # Set membership may only change at the exact 64th-score cutoff.
     assert all(p==aa[-1][1] for w,p in added+removed)
    tie_changes.append(dict(position=position,reordered=aa!=bb,added=added,removed=removed))
   (out/('tie_audit_'+row[key]+'.json')).write_text(json.dumps(tie_changes,indent=2))
  proof.append(dict(key=row[key],maxgaps=gaps));(out/'VALID_control.json').write_text(json.dumps(proof,indent=2));assert max(gaps.values())<=1e-6,proof[-1]
  if len(proof)==2:(out/'VALID_control_passed.json').write_text(json.dumps(dict(cases=proof,training_updates=0,labels_read=False),indent=2))
 text=source.read_text();payload={'Z2':"dict(x=mapped.tolist(),surface=raw.tolist())",'Z8':'ev','Z9':"dict(x=x.tolist(),singing=float(q[4]),ear_difference=float(np.max(abs(ear[0]-ear[-1]))))"}[route]
 needle='if i%10==0:save('if route=='Z2'else'if i%5==0:save('
 patches=[(needle,'validate(r,'+payload+')\n            '+needle),("(out/'evidence.jsonl').open('w')","(out/'evidence.jsonl').open('x')")]
 if route=='Z9':patches.append(("sid=r['signal_id'];path=data/r['split']","sid=r['signal_id'];data=Path(p['data_roots'][r['split']]);path=data/r['split']"))
 for old,new in patches:assert text.count(old)==1;text=text.replace(old,new)
 (out/'engineering_patch.json').write_text(json.dumps(dict(source_SHA=cfg['original_worker_SHA'],patches=patches,model_math_changed=False),indent=2))
 exec(compile(text,str(source),'exec'),dict(__name__='__main__',__file__=str(source),validate=validate));assert len(proof)==2
 p=out/'complete.json';r=json.loads(p.read_text());r.update(VALID_controls=2,training_updates=0,EVAL_unique_inputs=r['rows']-2);p.write_text(json.dumps(r,indent=2))

if __name__=='__main__':main()
