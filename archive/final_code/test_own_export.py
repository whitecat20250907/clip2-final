"""Assemble the preserved OWN readout from measured EVAL inputs only."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from test_fixed_point_readout import fixed_increment

def main():
 b=Path(__file__).resolve().parent;c=b/'outputs/TEST_components';v=b.parents[2];proof=json.loads((b/'outputs/TEST_POINT_READOUT_QUALIFICATION.json').read_text());sources={}
 def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
 def load(p,expected):
  assert sha(p)==expected,(str(p),sha(p));sources[str(p)]=expected
  with np.load(p,allow_pickle=False)as z:return{k:z[k].copy()for k in z.files}
 parent=load(c/'PRIOR_FIXED_EVAL_intermediate.npz','f9a8d75b8e06e9a3853e8dc8f7e09dda75abc004a1196911af104332dad62eae');ids=parent['signal_ids'];pf=parent['prediction']*100
 raw={};increments={}
 for judge,label,pin in [('MF','MF','ef260cd87293fa0c2c9dc963a7d2e67bc2c309c88cfed39acbead59d0d0f69b9'),('Phi4MM','Phi','4280f77128be4089ed61804456e8f89cab03673d23ef3d3d80db59e0e4522855')]:
  z=load(b/'outputs/TEST_point_exports'/judge/'raw_fields.npz',pin);assert np.array_equal(z['signal_ids'],ids);fp=Path(proof['cases'][label]['fit_path']);assert sha(fp)==proof['cases'][label]['fit_SHA'];fit=json.loads(fp.read_text());increments[label]=fixed_increment(z['raw_prompt_fields'],fit);sources[str(fp)]=sha(fp)
 mo=load(c/'MOSS_Omni_fixed_increments.npz',proof['output_SHA']);assert np.array_equal(mo['signal_ids'],ids)
 for k in ['MOSS','Omni']:increments[k]=mo[k+'_increment_points']
 # Each historical candidate crossed a ten-decimal CSV boundary before OWN.
 candidates=np.stack([[float(format(float(q),'.10f'))for q in np.clip(pf+increments[k],0,100)]for k in ['MF','Phi','MOSS','Omni']]);prediction=candidates.mean(0);assert len(ids)==len(set(ids.tolist()))==2189 and np.isfinite(prediction).all()and((prediction>=0)&(prediction<=100)).all()
 p=c/'OWN_EVAL_intermediate.csv'
 with p.open('w',newline='')as f:
  w=csv.writer(f,lineterminator='\r\n');w.writerows(zip(ids,[f'{q:.10f}'for q in prediction]))
 rec=dict(rows=2189,CSV_SHA=sha(p),range_points=[float(prediction.min()),float(prediction.max())],formula='mean(clip(PF + frozen_positive_increment_j,0,100), j=MF,Phi,MOSS,Omni)',training_updates=0,model_changes=0,source_files=sources,final_joint_CSV_complete=False,AB_source_review_pending=True,not_for_submission=True)
 (c/'OWN_EVAL_intermediate.json').write_text(json.dumps(rec,indent=2));print(json.dumps({k:v for k,v in rec.items()if k!='source_files'}))

if __name__=='__main__':main()
