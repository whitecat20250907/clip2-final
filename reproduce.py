"""Exact frozen interface replay. No raw-audio extraction, fitting or submission."""
from pathlib import Path
import csv,io,json,hashlib
import numpy as np
import torch

ROOT=Path(__file__).resolve().parent
def rounded(x):return np.asarray([float(format(float(v),'.10f')) for v in x])
def payload(ids,p,header):
 f=io.StringIO(newline='');w=csv.writer(f,lineterminator='\r\n')
 if header:w.writerow(['signal_id','intelligibility_score'])
 w.writerows(zip(ids,[format(float(v),'.10f') for v in p]));return f.getvalue().encode()
def main():
 torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
 b=ROOT/'replay/interfaces';m=torch.load(b/'adapter.pt',map_location='cpu',weights_only=True)
 net=torch.nn.Linear(12,1,dtype=torch.float64);net.load_state_dict({k.removeprefix('linear.'):v for k,v in m['state'].items()});net.eval().requires_grad_(False)
 out=ROOT/'generated';out.mkdir(exist_ok=True);report={}
 for split,n in [('valid',2117),('eval',2189)]:
  with np.load(b/(split+'_inputs.npz'),allow_pickle=False) as z:d={k:z[k].copy() for k in z.files}
  ids=d['signal_ids'].tolist();assert len(ids)==len(set(ids))==n
  own=rounded(np.stack([rounded(np.clip(d['pf']+inc,0,100)) for inc in d['increments']]).mean(0));assert np.max(np.abs(own-d['gold_own']))<1e-9
  x=np.column_stack([d['audio8'],np.eye(3)[d['severity']],np.log1p(d['counts'])]);norm=m['normalization'];raw=(x-np.asarray(norm['mean']))/np.asarray(norm['std'])
  with torch.inference_mode():delta=.05*torch.tanh(net(torch.as_tensor(np.clip(raw,-5,5))).squeeze(-1)).numpy()
  delta[(abs(raw)>6).any(1)]=0
  a=own/100;c=d['lmy']/100;base=.5609719773532506*a+.4390280226467494*c
  if split=='valid':base=rounded(base*100)/100
  pred=rounded((base+delta*(a-c))*100);assert np.array_equal(pred,d['gold_joint'])
  blob=payload(ids,pred,split=='eval');name='JOINT_'+('VALID2117' if split=='valid' else 'EVAL2189')+'.csv'
  assert blob==(ROOT/'predictions'/name).read_bytes()
  (out/name).write_bytes(blob);report[split]={'rows':n,'byte_identical':True,'sha256':hashlib.sha256(blob).hexdigest()}
 report['scope']='frozen interface replay; raw audio frontend is not executed';report['fits']=0
 (out/'verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()

