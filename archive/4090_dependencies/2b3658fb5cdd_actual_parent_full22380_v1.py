"""TRAIN-only native posterior gradient engineering witness; not a fitted candidate."""
import os,sys,pathlib,json,time,hashlib,traceback,copy,re,ast,textwrap,subprocess
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','NUMBA_CACHE_DIR','TRITON_CACHE_DIR']:
 p=ROOT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
R=ROOT/'actual_parent_full22380_v1';R.mkdir(exist_ok=False);start=time.monotonic();updates=0
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def save(n,o):
 with (R/n).open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def progress(stage,**kw):
 o=dict(stage=stage,PID=os.getpid(),seconds=time.monotonic()-start,optimizer_updates=updates,**kw)
 (R/'progress.next.json').write_text(json.dumps(o,indent=2));os.replace(R/'progress.next.json',R/'progress.json')
 print(json.dumps(o),flush=True)
def run():
 global updates
 import numpy as np,torch,librosa,peft
 from torch import nn
 from torch.nn import functional as F
 from qwen_asr import Qwen3ASRModel
 from peft.tuners.lora.layer import LoraLayer
 from safetensors.torch import load_file
 assert torch.__version__=='2.9.1+cu128'
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
 torch.use_deterministic_algorithms(True);torch.manual_seed(4401601)
 D=torch.device('cuda:0')
 N=ROOT.parent;RUN=N.parent
 P=RUN/'training_pair_direct_v4/inputs'
 B=RUN/'training_pair_direct_v4/side_research_v1/noise_followup_20261016_v1/new_baseline16_26298_transfer_v1'
 M=pathlib.Path('/root/clip2_workspace/models/Qwen3-ASR-1.7B')
 A=pathlib.Path('/root/autodl-tmp/clip2_runs/qwen3_listener_no_x_20261015/full_epoch1_v1/checkpoint_step3322')
 DATA=pathlib.Path('/root/autodl-tmp/clip2_data/cadenza_data/clip2')
 SRC=N/'full_posterior_word_residual_goal16_345312_v1/produce.py'
 protocol=read(ROOT/'ENGINEERING_PREREGISTRATION_V1.json')
 for p,v in protocol['existing_AutoDL_sources'].items():assert sha(p)==v['sha256']
 meta=read(P/'metadata.json');fitids=read(RUN/'partition.json')['train_ids']
 assert len(fitids)==16180 and set(fitids)<=set(meta['train_ids']) and len(meta['train_ids'])==16204
 with np.load(P/'static.npz',allow_pickle=False) as z:
  y=z['y'].copy();words=z['words'][:16204].copy();wt=z['word_targets'].copy();wm=z['word_target_mask'].copy();pi=z['prompt_index'][:16204].copy()
 assert y.shape==(16204,) and wt.shape==wm.shape==(16204,12)
 fit=set(fitids)
 selected=[i for i,s in enumerate(meta['train_ids']) if s in fit and 0<float(y[i])<1 and bool((wm[i]&words[i]).any())][:4]
 assert len(selected)==4
 rows=[dict(signal=meta['train_ids'][i],text=meta['prompts'][int(pi[i])],words=int(words[i].sum()),part='train',audio=str(DATA/'train/signals'/(meta['train_ids'][i]+'.flac'))) for i in selected]
 old=N/'full_cache_replay_repair_v1';manifest=read(old/'manifest.json');gold=[]
 with np.load(N/'full_posterior_word_residual_goal16_345312_v1/packed_v1/native_full_posterior_word32_all22380.npz',allow_pickle=False) as z:
  cached32=z['word_features'][selected].copy()
  assert z['signals'][selected].tolist()==[r['signal'] for r in rows]
 for index,row in zip(selected,rows):
  entry=next(e for e in manifest['chunks'] if e['start']<=index<e['stop']);p=old/entry['file'];assert sha(p)==entry['sha256']
  with np.load(p,allow_pickle=False) as z:
   j=index-entry['start'];lo,hi=map(int,z['token_offsets'][j:j+2])
   assert str(z['signals'][j])==row['signal']
   gold.append(dict(ids=z['token_ids'][lo:hi].copy(),owner=z['token_to_word'][lo:hi].copy(),lp=z['token_logp'][lo:hi,:,1].copy(),f13=z['word_features'][j,:,:,1].copy(),audio_sha256=str(z['audio_sha256'][j])))
 save('TRAIN_SCOPE_AND_SOURCES.json',dict(selected_indices=selected,signals=[r['signal'] for r in rows],selection='First4canonicalstrict24TRAINpartialscoreandknownwordmask; noDEVselection',TRAIN_only=True,DEV_truth_read=False,VALID_truth_read=False,source_sha256=sha(SRC),preregistration_sha256=sha(ROOT/'ENGINEERING_PREREGISTRATION_V1.json'),parent_prior='Actual frozen RMS parents rebuilt from existing AutoDL weights plus storedx9native; two seeds',formal_training_started=False))
 stable=0
 while stable<2:
  v=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
  free,util=map(int,v.stdout.strip().split(','));stable=stable+1 if free>=20480 and util<=5 else 0
  progress('waiting_idle_GPU',free_MiB=free,utilization=util)
  if stable<2:time.sleep(10)
 _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(12*1024**3/total)
 progress('loading_existing_frozen_model')
 wrapper=Qwen3ASRModel.from_pretrained(str(M),dtype=torch.bfloat16,device_map=None)
 native=wrapper.model;processor=wrapper.processor
 prefix=processor.apply_chat_template([[{'role':'system','content':''},{'role':'user','content':[{'type':'audio','audio':None}]}]],add_generation_prompt=True,tokenize=False)[0]+'language English<asr_text>'
 source=SRC.read_text();tree=ast.parse(source);body=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
 helpers={n.name:textwrap.dedent(ast.get_source_segment(source,n)) for n in body.body if isinstance(n,ast.FunctionDef)}
 env=dict(np=np,torch=torch,librosa=librosa,re=re,pathlib=pathlib,processor=processor,prefix=prefix,native=native,sha=sha)
 exec(compile(helpers['grid'],'<existing_AutoDL_grid>','exec'),env)
 exec(compile(helpers['prepare'],'<existing_AutoDL_prepare>','exec'),env)
 pm=peft.PeftModel.from_pretrained(native,str(A),is_trainable=False,autocast_adapter_dtype=True);native=pm.get_base_model();env['native']=native
 original_adapter=load_file(str(A/'adapter_model.safetensors'),device='cpu')
 current={n.replace('.default.','.'):p for n,p in pm.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
 assert len(current)==208 and set(current)==set(original_adapter)
 assert all(p.dtype==torch.float32 and torch.equal(p.detach().cpu(),original_adapter[n]) for n,p in current.items())
 for m in native.modules():
  if isinstance(m,LoraLayer):m.enable_adapters(True)
 pm.to(D).eval().requires_grad_(False)
 frozen=dict(pm.named_parameters())
 def frozen_hash():
  h=hashlib.sha256()
  for n,p in sorted(frozen.items()):
   h.update(n.encode()+b'\0');h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 frozen_before=frozen_hash()
 prepared=[env['prepare'](r) for r in rows]
 for (x,m,h),g in zip(prepared,gold):
  assert h==g['audio_sha256'] and np.array_equal(m['ids'],g['ids']) and np.array_equal(m['owner'],g['owner'])
 def logits(x,m,null):
  hook=None
  if null:
   def remove_audio(module,args,out):
    result=copy.copy(out);result.last_hidden_state=torch.zeros_like(out.last_hidden_state);return result
   hook=native.thinker.audio_tower.register_forward_hook(remove_audio)
  try:out=native.thinker(**{k:v.to(D) for k,v in x.items()},use_cache=False,return_dict=True)
  finally:
   if hook is not None:hook.remove()
  pos=torch.arange(m['prefix_length']-1,x['input_ids'].shape[1]-1,device=D)
  v=out.logits[0,pos].float();assert v.shape==(len(m['ids']),151936)
  return v
 def raw_features(r,n,m):
  lr=r.double().log_softmax(-1);ln=n.double().log_softmax(-1);pr=lr.exp();pn=ln.exp();lm=torch.logaddexp(lr,ln)-__import__('math').log(2.)
  tr=lr.topk(2,dim=-1).values;tn=ln.topk(2,dim=-1).values
  stats=torch.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,0]-tr[:,1],tn[:,0]-tn[:,1],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*(pr-pn).abs().sum(-1)],-1)
  target=torch.as_tensor(m['ids'],device=D)[:,None]
  rp=r.log_softmax(-1).gather(1,target).squeeze(1);np_=n.log_softmax(-1).gather(1,target).squeeze(1)
  owner=torch.as_tensor(m['owner'],device=D);a=[];b=[]
  for wi in range(m['word_count']):
   mask=owner==wi;rs=rp[mask];ns=np_[mask];ds=rs-ns
   f13=torch.stack([v for q in [rs,ns,ds] for v in [q.sum(),q.mean(),q.min(),q[0]]]+[rp.new_tensor(float(mask.sum()))])
   z=stats[mask];f32=torch.cat([z.sum(0),z.mean(0),z.min(0).values,z[0]]).float()
   a.append(f13);b.append(f32)
  a=F.pad(torch.stack(a),(0,0,0,12-m['word_count']));b=F.pad(torch.stack(b),(0,0,0,12-m['word_count']))
  return torch.cat([a,b],-1),torch.stack([rp,np_],-1)
 # Rebuild the ACTUAL frozen RMS parent from existing AutoDL weights.
 import types
 for name,code in read(P/'module_sources.json').items():
  module=types.ModuleType(name);sys.modules[name]=module;exec(compile(code,'<existing_AutoDL_module:'+name+'>','exec'),module.__dict__)
 initial=torch.load(P/'initial_model.pt',map_location='cpu',weights_only=False).eval().requires_grad_(False)
 assert tuple(initial.q_proj.weight.shape)==(192,2048)
 heads={};exec(compile((RUN/'training_pair_direct_v4/strict24_head.py').read_text(),'<existing_AutoDL_head>','exec'),heads)
 with np.load(P/'static.npz',allow_pickle=False) as z:
  static={k:z[k].copy() for k in ['pooled','side','values','support','severity','tokens','token_lengths','word_text','mapped','prompt_index']}
 parent_models={}
 for seed in [44,45]:
  folder=RUN/('training_pair_direct_v4/side_research_v1/human_adapted_encoder_seed%d_v1/adapted_encoder_16180'%seed)
  stage=copy.deepcopy(initial);stage.load_state_dict(torch.load(folder/'stage1_final.pt',map_location='cpu',weights_only=False)['weights'])
  score=heads['FineResidual'](heads['CTCReadout'](heads['TFReadout'](),seed),seed,'bounded_pooled')
  score.load_state_dict(torch.load(folder/'beta_final.pt',map_location='cpu',weights_only=False)['weights'])
  parent_models[seed]=dict(stage=stage.eval().requires_grad_(False),score=score.eval().requires_grad_(False),norm=read(folder/'normalization.json'))
 def parent_forward(i,frames,seed):
  item=parent_models[seed];pi=int(static['prompt_index'][i]);tn=int(static['token_lengths'][pi])
  def t(k):return torch.as_tensor(static[k][i:i+1])
  w=torch.as_tensor(words[i:i+1]);support=t('support')
  audio=frames.float()[None];audio=audio*torch.rsqrt(audio.square().mean(-1,keepdim=True)+1e-6)
  args=(t('pooled'),t('side'),t('values'),support,w,audio,torch.ones(audio.shape[:2],dtype=torch.bool),torch.as_tensor(static['tokens'][pi:pi+1,:tn]),torch.ones((1,tn),dtype=torch.bool),torch.as_tensor(static['word_text'][pi:pi+1]),torch.as_tensor(static['mapped'][pi:pi+1]),t('severity'))
  with torch.no_grad():_,z=item['stage'](*args,mode='boundary_joint')
  norm=item['norm'];ev=np.clip((static['values'][i:i+1]-np.asarray(norm['evidence_mean']))/np.asarray(norm['evidence_std']),-6,6).astype(np.float32);ev[~static['support'][i:i+1]]=0
  with np.load(P/'static.npz',allow_pickle=False) as zz:listener=zz['listener_adapted'][i:i+1].copy()
  listener=np.clip((listener-np.asarray(norm['listener_mean']))/np.asarray(norm['listener_std']),-6,6).astype(np.float32)
  extra=np.clip((ev[:,:,:24]-np.asarray(norm['extra_mean']))/np.asarray(norm['extra_std']),-8,8).astype(np.float32)
  with torch.no_grad():q,logit=item['score'](z,torch.as_tensor(ev),torch.as_tensor(listener),support,w,t('severity'),torch.as_tensor(extra))
  return q.numpy()[0],(logit.sigmoid()*w).numpy()[0]
 sidepath=B.parent/'x9_sealed_all22380.npz';assert sha(sidepath)=='25586bd4f7b348cdf62d33a2935f0169748384db16cead5b44e8431292b0e323'
 with np.load(sidepath,allow_pickle=False) as z:side={k:z[k].copy() for k in z.files}
 with np.load(B/'parent_x9_sealed_all22380.npz',allow_pickle=False) as z:target={k:z[k].copy() for k in z.files}
 ns={};exec(compile((B/'portable_core.py').read_text(),'<existing_AutoDL_portable_core>','exec'),ns)
 norm=read(B/'normalization.json')
 with np.load(P/'static.npz',allow_pickle=False) as z:words=z['words'].copy();listener_adapted=z['listener_adapted'].copy()
 assert words.shape==(20263,12)
 ids=side['signals'].tolist();assert ids[:20263]==meta['ids'] and ids==target['signals'].tolist()
 raw13=[];cached_audio=[]
 for e in manifest['chunks']:
  qpath=old/e['file'];assert sha(qpath)==e['sha256']
  with np.load(qpath,allow_pickle=False) as z:
   assert z['signals'].tolist()==ids[e['start']:e['stop']]
   raw13.append(z['word_features'][:,:,:,1].copy());cached_audio.extend(z['audio_sha256'].tolist())
 raw13=np.concatenate(raw13)
 with np.load(N/'full_posterior_word_residual_goal16_345312_v1/packed_v1/native_full_posterior_word32_all22380.npz',allow_pickle=False) as z:
  raw32=z['word_features'].copy();assert z['signals'].tolist()==ids
 raw45=np.concatenate([raw13,raw32],-1);xx=np.clip((raw45-np.asarray(norm['mean']))/np.asarray(norm['std']),-6,6).astype(np.float32);xx[~side['words']]=0.
 with np.load(B/'portable_VALID_inputs.npz',allow_pickle=False) as vv:assert np.array_equal(xx[20263:],vv['x'][39:])
 def frames_direct(x):
  length=int(x['feature_attention_mask'][0].sum());mel=x['input_features'][0,:,:length].to(D,dtype=torch.bfloat16)
  with torch.inference_mode():out=native.thinker.audio_tower(mel,feature_lens=torch.tensor([length],device=D,dtype=torch.long)).last_hidden_state
  assert out.ndim==2 and out.shape[1]==2048 and torch.isfinite(out).all()
  return out.float().cpu().to(torch.float16)
 direct_gates=[]
 for x,m,h in prepared:
  captured=[]
  hook=native.thinker.audio_tower.register_forward_hook(lambda mod,args,out:captured.append(out.last_hidden_state.detach().cpu().to(torch.float16).clone()))
  with torch.no_grad():dummy=logits(x,m,False)
  hook.remove();direct=frames_direct(x);assert len(captured)==1 and torch.equal(direct,captured[0])
  direct_gates.append(dict(audio_sha256=h,frames=list(direct.shape),direct_vs_native_exact=True))
 save('DIRECT_AUDIO_TOWER_GATE_V1.json',dict(status='same4TRAIN_direct_audio_tower_vs_native_forward_bitwise_equal',reports=direct_gates,optimizer_updates=0))
 def parent_batch(indices,frames,seed):
  item=parent_models[seed];pis=static['prompt_index'][indices];tns=static['token_lengths'][pis];tmax=int(tns.max())
  width=int(words[indices].sum(1).max());fmax=max(len(v) for v in frames)
  audio=torch.zeros(len(indices),fmax,2048);am=torch.zeros(len(indices),fmax,dtype=torch.bool)
  for j,f in enumerate(frames):audio[j,:len(f)]=f.float();am[j,:len(f)]=True
  audio=audio*torch.rsqrt(audio.square().mean(-1,keepdim=True)+1e-6)*am[...,None]
  w=torch.as_tensor(words[indices,:width]);support=torch.as_tensor(static['support'][indices,:width]);sev=torch.as_tensor(static['severity'][indices])
  args=(torch.as_tensor(static['pooled'][indices]),torch.as_tensor(static['side'][indices]),torch.as_tensor(static['values'][indices,:width]),support,w,audio,am,torch.as_tensor(static['tokens'][pis,:tmax]),torch.as_tensor(np.arange(tmax)[None,:]<tns[:,None]),torch.as_tensor(static['word_text'][pis,:width]),torch.as_tensor(static['mapped'][pis,:width]),sev)
  with torch.no_grad():_,logit=item['stage'](*args,mode='boundary_joint')
  n=item['norm'];ev=np.clip((static['values'][indices,:width]-np.asarray(n['evidence_mean']))/np.asarray(n['evidence_std']),-6,6).astype(np.float32);ev[~static['support'][indices,:width]]=0
  lis=np.clip((listener_adapted[indices,:width]-np.asarray(n['listener_mean']))/np.asarray(n['listener_std']),-6,6).astype(np.float32)
  ex=np.clip((ev[:,:,:24]-np.asarray(n['extra_mean']))/np.asarray(n['extra_std']),-8,8).astype(np.float32)
  with torch.no_grad():q,z=item['score'](logit,torch.as_tensor(ev),torch.as_tensor(lis),support,w,sev,torch.as_tensor(ex));wp=(z.sigmoid()*w).numpy()
  return np.pad(wp,((0,0),(0,12-width)))
 parent={s:np.zeros((22380,12),np.float32) for s in [44,45]}
 shards=[];tower=native.thinker.audio_tower
 for lo in range(0,20263,64):
  hi=min(lo+64,20263);indices=np.arange(lo,hi);frames=[]
  for i in indices:
   row=dict(signal=ids[i],text=meta['prompts'][int(static['prompt_index'][i])],words=int(words[i].sum()),audio=str(DATA/'train/signals'/(ids[i]+'.flac')))
   x,m,h=env['prepare'](row);assert h==cached_audio[i]
   frames.append(frames_direct(x))
  for seed in [44,45]:
   hp=parent_batch(indices,frames,seed);p0=(hp+side['seed%d_word_probability'%seed][lo:hi])*.5;p0[~words[lo:hi]]=0.;parent[seed][lo:hi]=p0
  if hi%256==0 or hi==20263:
   name='parent_chunk_%05d_%05d.npz'%(lo,hi)
   # Selected progress shards are diagnostics only; no fitting or DEV truth.
   with (R/name).open('xb') as f:np.savez_compressed(f,signals=np.asarray(ids[lo:hi]),parent44=parent[44][lo:hi],parent45=parent[45][lo:hi])
   shards.append(dict(file=name,sha256=sha(R/name),source_row_start=lo,source_row_stop=hi))
   free,total=torch.cuda.mem_get_info();assert free>=4*1024**3
   progress('reconstructing_frozen_actual_parent',completed_rows=hi,total_audio_rows=20263,free_MiB=free/2**20)
  del frames
 with np.load(B/'portable_VALID_inputs.npz',allow_pickle=False) as z:
  assert z['signals'][39:].tolist()==ids[20263:]
  assert np.array_equal(xx[20263:],z['x'][39:])
  for seed in [44,45]:parent[seed][20263:]=z['parent_x9%d'%seed][39:]
 output={};reports=[]
 allwords=side['words']
 for seed in [44,45]:
  model=ns['NativeWordScore'](seed,1.8);model.load_state_dict(torch.load(B/('parent_x9_seed%d_final.pt'%seed),map_location='cpu',weights_only=False)['weights']);model.eval()
  qs=[];wps=[]
  with torch.no_grad():
   for lo in range(0,22380,128):
    q,w,z=model(torch.as_tensor(xx[lo:lo+128]),torch.as_tensor(parent[seed][lo:lo+128]),torch.as_tensor(allwords[lo:lo+128]));qs.append(q.numpy());wps.append(w.numpy())
  q=np.concatenate(qs);wp=np.concatenate(wps)
  qgap=float(abs(q-target['seed%d_prediction'%seed]).max());wgap=float(abs(wp-target['seed%d_word_probability'%seed]).max())
  assert qgap<=2e-6 and wgap<=2e-6,(seed,qgap,wgap)
  output['seed%d_prediction'%seed]=q;output['seed%d_word_probability'%seed]=wp;output['parent%d'%seed]=parent[seed]
  reports.append(dict(seed=seed,rows=22380,q_maxgap=qgap,word_maxgap=wgap))
 prediction=(output['seed44_prediction'].astype(np.float64)+output['seed45_prediction'].astype(np.float64))*.5
 with (R/'frozen_parent_and_45features_all22380.npz').open('xb') as f:np.savez_compressed(f,signals=np.asarray(ids),words=allwords,x45=xx,parent44=parent[44],parent45=parent[45])
 with (R/'replayed_baseline_all22380.npz').open('xb') as f:np.savez_compressed(f,signals=np.asarray(ids),words=allwords,prediction=prediction,**{k:v for k,v in output.items() if not k.startswith('parent')})
 assert frozen_hash()==frozen_before
 save('ALL22380_ACTUAL_PARENT_REPLAY_V1.json',dict(status='all22380_frozen_RMS_x9parents_and_latest45head_control_replayed',rows=22380,Audio1_rows_fresh_encoder=20263,VALID_parent_rows_reused_from_existing_persistent_public_inputs=2117,VALID_feature_rows_bitwise_same=True,reports=reports,cache_sha256=sha(R/'frozen_parent_and_45features_all22380.npz'),prediction_sha256=sha(R/'replayed_baseline_all22380.npz'),source_prediction_sha256=sha(B/'parent_x9_sealed_all22380.npz'),control_prediction_maxgap=float(abs(prediction-target['prediction']).max()),optimizer_updates=0,DEV_truth_read=False,VALID_truth_read=False,all_frozen_model_parameters_unchanged=True,source_only_existing_AutoDL_files=True,school_files_imported=False,precision='CPU2.9.1readout vs frozen2.4.1 source, <=2e-6 tolerance; not byte-exact CSV claim',seconds=time.monotonic()-start))
 pm.to('cpu');torch.cuda.empty_cache();(R/'exitcode').write_text('0');progress('all22380_parent_reconstruction_complete')
try:run()
except BaseException as e:
 traceback.print_exc();save('FAILURE.json',dict(error=type(e).__name__,message=str(e),optimizer_updates=updates,automatic_retry=False));(R/'exitcode').write_text('1');raise
