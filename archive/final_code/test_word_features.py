"""Build original 55/72 word inputs, then apply the original four music heads."""
import argparse,ast,collections,hashlib,json,pathlib,types
import numpy as np

B=pathlib.Path(__file__).parent;V=B.parents[2];J=pathlib.Path('E:/icassp/zhaojiahao/jobs')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def defs(path,names,scope):
 text=path.read_text();nodes=[n for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in names];assert {n.name for n in nodes}==set(names);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)

def features(rows,records,align,rules,scope):
 x=np.zeros((len(rows),12,55),np.float32);extra=np.zeros((len(rows),12,17),np.float32);mask=np.zeros((len(rows),12),bool)
 for i,row in enumerate(rows):
  sid=row['signal_id'];prompt=row['prompt'];words=prompt.split();n=len(words);mask[i,:n]=True;blocks=[];surface=[]
  for stream in ['A1L','A1R','A2L','A2R']:
   r=records[sid,stream];al=align.align_words(prompt,' '.join(r['recognized_words']));block=np.zeros((n,9),np.float32)
   for j,k in enumerate(al['hyp_indices']):
    block[j,0]=al['labels'][j];block[j,1]=k is not None and not al['labels'][j];block[j,2]=k is None;block[j,6]=al['mapping_mask'][j]
    if k is not None:block[j,3:6]=[r['word_logprob_sum'][k],r['word_logprob_mean'][k],r['word_token_counts'][k]]
   block[:,7]=min(2.,al['distance']/n);block[:,8]=r['generation_capped'];blocks.append(block);surface.append(scope['norm_fields'](prompt,r['recognized_words'],rules,align)[0])
  context=np.concatenate([np.tile(np.eye(3,dtype=np.float32)[row['severity']],(n,1)),np.arange(n,dtype=np.float32)[:,None]/max(1,n-1),np.asarray([len(w)/12 for w in words],np.float32)[:,None]],1)
  x[i,:n]=np.concatenate(blocks+[blocks[0][:,:7]-blocks[2][:,:7],blocks[1][:,:7]-blocks[3][:,:7],context],1)
  extra[i,:n]=np.concatenate(surface+[np.asarray([len(rules.get(w,w).split())>1 for w in words],np.float32)[:,None]],1)
 return dict(x55=x,x72=np.concatenate([x,extra],2),mask=mask,signal_ids=np.asarray([r['signal_id']for r in rows]))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--family',choices=['MF','MOSS'],required=True);a=ap.parse_args();folder=B/'outputs/TEST_WORD_WITNESS'/a.family/'collected';assert (folder/'complete.json').exists()and(folder/'VALID_control_passed.json').exists()
 af=V.parent.parent/'research/personal_clip2/alignment.py';expected=json.loads((V/'new_angles_20261021/music_witness_full_train/readout/inputs_report.json').read_text())['machine_alignment_code_SHA'];assert sha(af)==expected
 scope={'np':np,'collections':collections};defs(af,['_words','align_words'],scope);align=types.SimpleNamespace(align_words=scope['align_words']);p=J/'v10_new_information_20261022.py';assert sha(p)=='b466cddacf3211b08df00c79aa20daa10c4cc30cb940bd420ae8228da822c75f';defs(p,['expand','norm_fields'],scope)
 rulesfile=V/'next_optimization_20261021/contraction_audit.json';rules=json.loads(rulesfile.read_text())['expansion'];assert len(rules)==28
 records={};splits={}
 for line in (folder/'evidence.jsonl').read_text().splitlines():
  r=json.loads(line);key=r['signal_id'],r['stream'];assert key not in records;records[key]=r;splits[r['signal_id']]=r['split']
 meta={r['signal_id']:r for r in json.loads((V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json').read_text())};controls=[meta[s]for s,v in splits.items()if v=='valid'];assert len(controls)==2
 erows=json.loads((B/'outputs/TEST_LMY_NATIVE/selection.json').read_text())[2:];erows=[dict(signal_id=r['signal'],prompt=r['prompt'],severity=r['severity'])for r in erows];assert {r['signal_id']for r in erows}=={s for s,v in splits.items()if v=='eval'}
 # The original 587135 source and CSV establish G01 MOSS72 ancestry.
 # The cleanup handoff incorrectly pinned G02 MOSS55 (different SID587114).
 # This engineering correction restores the measured frozen graph; no refit.
 z=features(controls+erows,records,align,rules,scope);cpfile=V/('next_optimization_20261021/G01_MF/readout/valid.npz'if a.family=='MF'else'next_optimization_20261021/G01_G02/G01_valid.npz');dim=72
 with np.load(cpfile,allow_pickle=False)as old:ids=old['signal_ids'].tolist();ii=[ids.index(r['signal_id'])for r in controls];gap=float(abs(z['x'+str(dim)][:2]-old['x'][ii]).max());assert gap==0,(a.family,gap);vraw=old['x'].copy();vmask=old['mask'].copy()
 import torch
 torch.set_num_threads(2);proof=[];pred=[];idx=json.loads((B/'original_assets/ASSET_INDEX.json').read_text());assets={r['original_path']:r for r in idx['copied_files']};group='next_optimization_20261021/G01_MF/readout/results/witness_full_s'if a.family=='MF'else'next_optimization_20261021/G01_G02/results/G01_s'
 if a.family=='MOSS':
  receipt=json.loads((B/'outputs/TEST_MOSS_FROZEN_ANCESTRY_CONTRADICTION.json').read_text());assert receipt['original_G01_replay_max_probability_gap']<1e-12
  assets.update({r['path']:r for r in receipt['heads']if r['arm']=='G01'})
 for seed in [142,242,342,442]:
  path=V/(group+str(seed))/'model.pt';asset=assets[str(path)];assert sha(path)==asset['SHA'];cp=torch.load(path,map_location='cpu',weights_only=False);assert cp['dim']==dim
  net=torch.nn.Sequential(torch.nn.Linear(dim,8),torch.nn.Tanh(),torch.nn.Linear(8,1));net.load_state_dict({k.removeprefix('head.'):v for k,v in cp['state'].items()});net.eval().requires_grad_(False)
  def infer(raw,mask):
   x=((raw-cp['mean'])/cp['std']).astype(np.float32);x[~mask.astype(bool)]=0;out=[]
   with torch.inference_mode():
    for i in range(0,len(x),256):
     mm=torch.tensor(mask[i:i+256],dtype=torch.float32);out.append(((torch.sigmoid(net(torch.from_numpy(x[i:i+256])).squeeze(-1))*mm).sum(1)/mm.sum(1)).numpy())
   return np.concatenate(out)
  val=infer(vraw,vmask)
  with np.load(path.with_name('valid.npz'),allow_pickle=False)as old:gap=float(abs(val-old['prediction']).max());assert gap<2e-7,(seed,gap)
  proof.append(dict(seed=seed,model_SHA=asset['SHA'],VALID_readout_maxgap_probability=gap));pred.append(infer(z['x'+str(dim)][2:],z['mask'][2:]))
 out=B/'outputs/TEST_components';path=out/(a.family+'_word_features.npz')
 with path.open('xb')as f:np.savez_compressed(f,**{k:v[2:]for k,v in z.items()},four_seed_predictions=np.asarray(pred),mean_prediction=np.mean(pred,axis=0))
 report=dict(family=a.family,rows=2189,raw_VALID_controls_exact=True,readout_qualification=proof,output_SHA=sha(path),training_updates=0,source_SHA={str(p):sha(p)for p in [af,J/'v10_new_information_20261022.py',rulesfile,folder/'evidence.jsonl']});(out/(a.family+'_word_features_qualification.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report))

if __name__=='__main__':main()
