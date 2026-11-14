"""Compare frozen export inference to TRAIN full-chain tensor pipeline, no labels."""
import os,sys,time,json,hashlib,importlib.util,traceback
from pathlib import Path
from types import SimpleNamespace
if not hasattr(Path,'is_relative_to'):
 def _path_is_relative_to(self,other):
  try:self.relative_to(other);return True
  except ValueError:return False
 Path.is_relative_to=_path_is_relative_to
H=Path(__file__).resolve().parent;ROOT=H.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015';OUT=H/'scoring_port_gate_v2';OUT.mkdir(exist_ok=False)
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','MPLCONFIGDIR','TORCH_HOME']:
 p=OUT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',RAYON_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='',POOLED_TEXT_RAM='0');sys.dont_write_bytecode=True
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
with (OUT/'launch.json').open('x') as f:json.dump({'PID':os.getpid(),'training':False,'labels_read':False},f)
try:
 import numpy as np,torch
 torch.set_num_threads(2)
 p=H/'real_audio64_v1/protocol.json'
 while not p.exists():
  launch=read(H/'real_audio64_v1/launch.json')
  if not Path('/proc',str(launch['PID'])).exists():raise RuntimeError('Audio selector process ended without protocol; inspect before retry')
  time.sleep(10)
 selected=read(p)['ids'];assert len(selected)==64
 inf=module('_qwen_valid_frozen',H/'valid_candidate_v1.py')
 swap=module('_port_swap',OLD/'run_qwen_frame_swap_stage1.py');rf=swap.load_original_module();os.chdir(str(OLD));ctx=rf.prepare(OUT,Path('/home/yangjing/workplace/clip2/clip2_next'))
 assert ctx['identity']==read(ROOT/'full_scale1521_v1/protocol.json')['public_input_identity'] and set(selected)<=set(ctx['train'])
 with np.load(ROOT/'qwen_first_shard_readonly_copy.npz',allow_pickle=False) as z:ids=z['signals'].tolist();lens=z['lengths'].astype(np.int64);frames=z['frames'].copy()
 ctx['frames']=SimpleNamespace(index={s:i for i,s in enumerate(ids)},lengths=lens,offsets=np.r_[0,np.cumsum(lens[:-1])],frames=frames,text=ctx['frames'].text)
 with np.load(OLD/'qwen_parent_word_cache_sharded_v1/qwen_parent_word_cache.npz',allow_pickle=False) as z:common=tuple(z[k].copy() for k in ['evidence','support','words'])
 ds=module('_port_ds',OLD/'train_qwen_downstream.py');sys.path.insert(0,str(ds.LISTENER));f=module('_port_listener',ds.LISTENER/'run_full_listener.py')
 b,a,dmap,width,br=f.training_bank(ctx,OUT);raw_adapted=a.copy();b,a=f.complete_listener_bank(ctx,OUT,b,a,br,dmap);di=np.array([ctx['index'][dmap[s]] for s in ctx['frozen'].ids],np.int64);ix=np.array([ctx['index'][s] for s in selected])
 report={}
 for arm in ['raw_qwen','rms_qwen']:
  point=inf.FrozenQwenCandidate(arm);parent,control=swap.make_model(rf,ctx,torch.device('cpu'));del control
  cp=ROOT/'full_scale1521_v1'/arm/'seed44_boundary_joint_final.pt';parent.load_state_dict(torch.load(cp,map_location='cpu',weights_only=True)['weights']);parent.eval().requires_grad_(False)
  with np.load(ROOT/'full_scale1521_v1'/arm/'parent_logits_and_predictions.npz',allow_pickle=False) as z:cached=(z['logits'].copy(),common[0],common[1],common[2],z['prediction'].copy())
  full=f.tensors(ctx,cached,b,a,di,ix,torch.device('cpu'),'adapted_listener');norm=point.extra_norm;extra=torch.from_numpy(np.clip((full[1][:,:,:24].numpy()-np.array(norm['mean']))/np.array(norm['std']),-8,8).astype(np.float32))
  with torch.inference_mode():expected=point.model(*full,extra)[0].numpy()
  got=[];gaps=np.zeros(7)
  with torch.inference_mode():
   for j in range(0,64,16):
    x=swap.batch(ctx,selected[j:j+16],torch.device('cpu'),'boundary_joint')
    got.extend(point.score(parent,x,raw_adapted[ix[j:j+16],:x[4].shape[1]]).tolist())
    refs=[v[j:j+16,:x[4].shape[1]] if v.ndim>=2 else v[j:j+16] for v in full]+[extra[j:j+16,:x[4].shape[1]]]
    for k,(actual,ref) in enumerate(zip(point.last_inputs,refs)):gaps[k]=max(gaps[k],float(abs(actual.double()-ref.double()).max()))
  maxgap=float(abs(np.array(got)-expected).max());assert maxgap<2e-6 and gaps.max()<2e-5
  report[arm]={'prediction_maxgap':maxgap,'input_field_maxgaps':gaps.tolist(),'candidate_checkpoint_sha256':sha(point.checkpoint)}
  del point,parent,full,extra
 result={'status':'frozen_export_adapter_matches_training_pipeline','n':64,'arms':report,'labels_read':False,'optimizer_steps':0,'VALID_predictions_made':False,'adapter_sha256':sha(H/'valid_candidate_v1.py')}
 with (OUT/'report.json').open('x') as f:json.dump(result,f,indent=2)
 (OUT/'exitcode').write_text('0');print(json.dumps(result),flush=True)
except BaseException:
 traceback.print_exc();(OUT/'exitcode').write_text('1');raise
