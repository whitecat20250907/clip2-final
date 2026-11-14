"""Frozen 36 temporal fields from identity-checked native word profiles."""
import ast,hashlib,json,pathlib,time
import numpy as np
B=pathlib.Path(__file__).parent;V=B.parents[2];O=B/'outputs'
def read(p):return json.loads(p.read_text())
def arr(p):
 with np.load(p,allow_pickle=False)as z:return {k:z[k].copy() for k in z.files}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
source=pathlib.Path('E:/icassp/zhaojiahao/jobs/v10_ER_expanded_features_20261022.py')
assert sha(source)=='eb82ed8d53a6526e094fd8882ad3228c95bd6dd25dd02ce16fe579f76b7407b2'
env={'np':np,'STREAMS':['A1L','A1R','A2L','A2R']};nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
def row(gd,pd,sid):
 gj=read(gd/(sid+'.json'));pj=read(pd/(sid+'.json'));assert gj['prompt']==pj['prompt']
 return env['row_features'](arr(gd/(sid+'.npz')),arr(pd/(sid+'.npz')),gj,pj)[0]
def main():
 old=V/'evidence_relations_20261022/expanded';sel=read(old/'selection.json');valid=arr(old/'ER02/valid.npz');proof=[]
 for i,r in [(i,r)for i,r in enumerate(sel)if r['split']=='valid'][:2]:
  sid=r['signal_id'];x=row(old/'results'/('Granite_part_'+str(i%6)),old/'results'/('Parakeet_part_'+str(i%6)),sid);j=valid['signal_ids'].tolist().index(sid);gap=float(abs(x-valid['x'][j,:len(x),38:]).max());assert gap==0.;proof.append(dict(signal=sid,maxgap=gap))
 rows=read(O/'TEST_LMY_NATIVE/selection.json')[2:];x=np.zeros((2189,12,36),np.float32);mask=np.zeros((2189,12),bool);start=time.monotonic()
 for i,r in enumerate(rows):
  q=row(O/'TEST_native_EVAL/Granite/collected',O/'TEST_native_EVAL_3090/Parakeet/collected',r['signal']);assert len(q)==r['n_words'];x[i,:len(q)]=q;mask[i,:len(q)]=True
  if i%250==0:print('ER02',i,'/2189',round(time.monotonic()-start,1),flush=True)
 dest=O/'TEST_components/ER02_temporal36.npz'
 with dest.open('xb')as f:np.savez_compressed(f,signal_ids=np.asarray([r['signal']for r in rows]),x=x,mask=mask)
 report=dict(rows=2189,VALID_controls=proof,source_SHA=sha(source),output_SHA=sha(dest),training_updates=0,PCM_identity_checked=True,seconds=time.monotonic()-start);(dest.with_suffix('.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
