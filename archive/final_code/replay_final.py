"""Replay the frozen final CSV from cached interfaces; no fitting or submission.

This verifies the delivered prediction interfaces. It does not claim arbitrary
raw-audio inference or whole-system nested OOF certification.
"""
import sys, argparse, json, hashlib, csv, io
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
import torch
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def payload(ids,p):
 s=io.StringIO(newline='');csv.writer(s,lineterminator='\r\n').writerows(zip(ids,[format(float(v),'.10f')for v in p]));return s.getvalue().encode('utf-8')
def csvpred(p):
 with p.open(encoding='utf-8-sig',newline='')as f:rows=list(csv.reader(f))
 assert len(rows)==2117 and all(len(r)==2 for r in rows)
 ids=[r[0]for r in rows];q=np.asarray([float(r[1])for r in rows],dtype=np.float64)
 assert len(set(ids))==2117 and np.isfinite(q).all() and ((q>=0)&(q<=100)).all()
 return ids,q
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--bundle',required=True);ap.add_argument('--output');a=ap.parse_args()
 base=Path(a.bundle).resolve();contract=json.loads((base/'replay_contract.json').read_text(encoding='utf-8'))
 assert contract['schema']=='CLIP2.preserved_final.cached_replay.v1'
 assert contract['four_route_SIDs']==[587835,587836,587634,587632]
 def checked(key):
  r=contract['files'][key];p=(base/r['relative_path']).resolve();assert p.is_relative_to(base) and sha(p)==r['SHA'],key;return p
 files={k:checked(k)for k in contract['files']}
 members=[csvpred(files[k])for k in ['MF_VALID','Phi_VALID','MOSS_VALID','Omni_VALID']]
 ids=members[0][0];assert all(x[0]==ids for x in members)
 ownbytes=payload(ids,np.mean([m[1]for m in members],axis=0))
 assert hashlib.sha256(ownbytes).hexdigest()==contract['files']['OWN_VALID']['SHA']
 oi,own=csvpred(files['OWN_VALID']);li,lmy=csvpred(files['LMY_VALID']);zi,zero=csvpred(files['ZERO_JOINT_VALID'])
 assert oi==li==zi==ids;own=own/100;lmy=lmy/100;zero=zero/100
 w0=contract['original_own_weight'];assert w0==.5609719773532506
 assert np.max(abs(zero-(w0*own+(1-w0)*lmy)))<=5.1e-13
 with np.load(files['VALID_DATA'],allow_pickle=False)as z:
  assert 'target'not in z.files and z['signal_ids'].tolist()==ids
  assert np.array_equal(own,z['own'])and np.array_equal(lmy,z['LMY'])
  audio=z['audio8'].copy();sev=z['severity'].copy();count=z['counts'].copy()
 assert audio.shape==(2117,8)and np.isfinite(audio).all()and np.isin(sev,[0,1,2]).all()and(count>0).all()
 x=np.column_stack([audio,np.eye(3)[sev.astype(int)],np.log1p(count.astype(float))])
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 saved=torch.load(files['ADAPTER_MODEL'],map_location='cpu',weights_only=True)
 assert saved['parameters']==13 and saved['fold']==-1
 norm=saved['normalization'];raw=(x-np.asarray(norm['mean']))/np.asarray(norm['std'])
 assert (np.asarray(norm['std'])>=.01).all()
 clipped=np.clip(raw,-5,5);ood=(np.abs(raw)>6).any(1)
 linear=torch.nn.Linear(12,1,dtype=torch.float64)
 linear.load_state_dict({k.removeprefix('linear.'):v for k,v in saved['state'].items()},strict=True);linear.eval()
 assert sum(p.numel()for p in linear.parameters())==13
 with torch.inference_mode():
  t=torch.as_tensor(clipped);delta=torch.where(torch.as_tensor(ood),torch.zeros(len(ids),dtype=t.dtype),.05*torch.tanh(linear(t).squeeze(-1))).numpy()
 assert np.isfinite(delta).all()and np.max(abs(delta))<=.0500000000001
 assert ((w0+delta>=0)&(w0+delta<=1)).all()and np.count_nonzero(delta[ood])==0
 p=zero+delta*(own-lmy)
 jointbytes=files['ZERO_JOINT_VALID'].read_bytes()if np.count_nonzero(delta)==0 else payload(ids,p*100)
 actual=hashlib.sha256(jointbytes).hexdigest();assert actual==contract['files']['FINAL_JOINT_VALID']['SHA']
 if a.output:
  op=Path(a.output).resolve();assert op!=files['FINAL_JOINT_VALID']
  if op.exists():assert op.read_bytes()==jointbytes
  else:
   with op.open('xb')as f:f.write(jointbytes)
 print(json.dumps(dict(cached_interface_replay_exact=True,own_SHA=hashlib.sha256(ownbytes).hexdigest(),joint_SHA=actual,rows=2117,adapter_parameters=13,original_weight=w0,actual_weight_min=float((w0+delta).min()),actual_weight_max=float((w0+delta).max()),OOD_rows=int(ood.sum()),fits=0,POSTs=0,strict_whole_system_OOF=False,raw_audio_infer_all=False,environment=dict(numpy=np.__version__,torch=torch.__version__))))
if __name__=='__main__':main()
