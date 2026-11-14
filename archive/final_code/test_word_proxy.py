"""Reproduce the original Z word proxy, including capped-stream exclusion."""
import ast,collections,hashlib,json,pathlib,re
import numpy as np
from scipy.special import expit
B=pathlib.Path(__file__).parent;V=B.parents[2];J=pathlib.Path('E:/icassp/zhaojiahao/jobs');O=B/'outputs'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def defs(path,names,scope):
 nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
env=dict(re=re,collections=collections);af=V.parent.parent/'research/personal_clip2/alignment.py';defs(af,['_words','align_words'],env);defs(J/'v10_human_channel_common_20261022.py',['words'],env)
def load(paths,ids):
 records={};excluded=collections.Counter()
 for fam,path in paths:
  for line in path.read_text().splitlines():
   r=json.loads(line);sid=r['signal_id']
   if sid not in ids:continue
   key=(sid,fam,r['stream']);assert key not in records
   if r.get('generation_capped',False):excluded[fam+'_capped']+=1;continue
   words=env['words'](r['transcription'])
   if len(words)>64:excluded[fam+'_over64']+=1;continue
   records[key]=words
 return records,dict(excluded)
def proxy(rows,records,width):
 weights=np.asarray(read(V/'final15h_20261022/Y3/lexical_table.json')['word_base_coefficients']);p=np.full((len(rows),width),.5);mask=np.zeros_like(p,bool)
 for i,r in enumerate(rows):
  words=r['prompt'].split();n=len(words);a=[]
  for fam in ['MF','MOSS']:
   for stream in ['A1L','A1R','A2L','A2R']:
    key=(r['signal_id'],fam,stream);a.append(np.asarray(env['align_words'](r['prompt'],' '.join(records[key]))['labels']) if key in records else np.zeros(n))
  x=np.column_stack([np.ones(n),np.stack(a,1),np.full(n,r['severity']==1),np.full(n,r['severity']==2)]);p[i,:n]=expit(x@weights);mask[i,:n]=True
 return p,mask
def main():
 with np.load(V/'fourth_round_20261023/V11_replay/joint_fit_inputs_conditional.npz')as z:ids=z['valid_signal_ids'].tolist();expected=z['valid_proxy'].copy();emask=z['valid_mask'].copy()
 meta={r['signal_id']:r for r in read(V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json')};paths=[(r['family'],pathlib.Path(r['path']))for r in read(V/'whole_human_mishearing_20261022/input_audit.json')['source_files']if r['family']in ['MF','MOSS']]
 records,excluded=load(paths,set(ids));p,m=proxy([meta[s]for s in ids],records,11);gap=float(abs(p-expected).max());assert gap==0. and np.array_equal(m,emask)
 rows=[dict(signal_id=r['signal'],prompt=r['prompt'],severity=r['severity'])for r in read(O/'TEST_LMY_NATIVE/selection.json')[2:]];paths=[(f,O/'TEST_WORD_WITNESS'/f/'collected/evidence.jsonl')for f in ['MF','MOSS']];records,ex=load(paths,{r['signal_id']for r in rows});p,m=proxy(rows,records,12)
 target=O/'TEST_components/word_proxy.npz'
 with target.open('xb')as f:np.savez_compressed(f,signal_ids=np.asarray([r['signal_id']for r in rows]),proxy=p,mask=m)
 report=dict(VALID2117_maxgap=gap,VALID_excluded_streams=excluded,EVAL_excluded_streams=ex,rows=2189,source_policy='Original generation_capped or normalized transcription length>64 streams omitted before proxy; missing hits neutral0; padded words probability.5 with false mask.',output_SHA=sha(target),training_updates=0,sources={str(p):sha(p)for p in [af,J/'v10_human_channel_common_20261022.py',J/'v10_human_channel_prepare_20261022.py',J/'v10_Z_readout_20261023.py']});target.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
