
"""AutoDL-local frozen native Qwen evidence using its existing ASR training runtime."""
import os,pathlib,json,time,hashlib,traceback,copy,re,inspect,subprocess
ROOT=pathlib.Path(__file__).resolve().parent
for key in ['HF_HOME','HF_HUB_CACHE','XDG_CACHE_HOME','TMPDIR','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','NUMBA_CACHE_DIR','TRITON_CACHE_DIR']:
 p=ROOT/'runtime'/key;p.mkdir(parents=True,exist_ok=True);os.environ[key]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
R=ROOT/'extraction_v1';R.mkdir(exist_ok=False);start=time.monotonic()
def read(p):return json.loads(pathlib.Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for b in iter(lambda:f.read(2**20),b''):h.update(b)
 return h.hexdigest()
def save(name,v):
 with (R/name).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def status(v):
 v.update(PID=os.getpid(),seconds=time.monotonic()-start,labels_read=False,optimizer_updates=0)
 p=R/'progress.next.json';p.write_text(json.dumps(v,indent=2,allow_nan=False));os.replace(p,R/'progress.json')
def run():
 import numpy as np,torch,librosa,transformers,tokenizers,peft
 from qwen_asr import Qwen3ASRModel
 from peft.tuners.lora.layer import LoraLayer
 from safetensors.torch import load_file
 assert torch.__version__=='2.9.1+cu128' and transformers.__version__=='4.57.6' and peft.__version__=='0.18.1' and tokenizers.__version__=='0.22.2'
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.use_deterministic_algorithms(True)
 D=torch.device('cuda:0');M=pathlib.Path('/root/clip2_workspace/models/Qwen3-ASR-1.7B')
 A=pathlib.Path('/root/autodl-tmp/clip2_runs/qwen3_listener_no_x_20261015/full_epoch1_v1/checkpoint_step3322')
 P=pathlib.Path('/root/autodl-tmp/clip2_runs/qwen_rms_strict24_exclusion_20261015_v1/training_pair_direct_v4/inputs');DATA=pathlib.Path('/root/autodl-tmp/clip2_data/cadenza_data/clip2')
 assert sha(A/'adapter_model.safetensors')=='a4748cb37ed726a7673f651102836bc2e37ba0fa86aded15947cdb8d4219aac8'
 assert sha(P/'metadata.json')=='f2dbd62c9020f8ed8308aaa68f94501df285e079f3c535457870f7c3a51fbc07'
 assert sha(P/'static.npz')=='63b8a57f2a9b5c7220ac50a987ed5495ca9f86990c2c204ae7c9317f8c5f018b'
 meta=read(P/'metadata.json')
 assert len(meta['ids'])==20263 and len(meta['train_ids'])==16204 and meta['ids'][16204:]==meta['dev_ids']
 assert digest(meta['dev_ids'])=='6c486a943337877860c67ac871194c84481944106d54e057c15ab556e96149a3'
 with np.load(P/'static.npz',allow_pickle=False) as z:pi=z['prompt_index'].copy();words=z['words'].copy()
 valid=read(DATA/'metadata/valid_metadata.json')
 assert len(valid)==2117 and all(set(v)=={'signal','original_prompt','prompt','n_words','hearing_loss'} for v in valid)
 assert digest([v['signal'] for v in valid])=='ea2cd972d4ab47835907a11a113de264187e2c487f8f22c0907c8b0a3d0e81d9'
 rows=[dict(signal=s,text=meta['prompts'][int(pi[i])],words=int(words[i].sum()),part='train' if i<16204 else 'DEV',audio=str(DATA/'train/signals'/(s+'.flac'))) for i,s in enumerate(meta['ids'])]
 rows += [dict(signal=v['signal'],text=v['prompt'],words=int(v['n_words']),part='VALID',audio=str(DATA/'valid/signals'/(v['signal']+'.flac'))) for v in valid]
 assert len(rows)==22380 and all(len(v['text'].split())==v['words'] and 0<v['words']<=12 for v in rows)
 save('ordered_public_inputs.json',rows)
 source_training=pathlib.Path('/root/clip2_workspace/preparation/qwen3_asr_20261014')
 sources={str(p):sha(p) for p in [source_training/'train_qwen_no_x_fit.py',source_training/'train_listener_lora_pilot.py',source_training/'build_qwen_no_x_manifests.py',A.parent/'config.json',pathlib.Path(inspect.getsourcefile(Qwen3ASRModel.from_pretrained))]}
 protocol=dict(rows=22380,public_input_universe=22380,TRAIN_input_universe=16204,original_DEV_universe=4059,VALID_input_universe=2117,order_sha256=digest([v['signal'] for v in rows]),DEV_order_sha256=digest(meta['dev_ids']),VALID_order_sha256=digest([v['signal'] for v in valid]),baseline_EvalAI_RMSE=16.345312181365085,baseline_PCC=.8565189070461371,
 variants=['audio_only_A1T0','full_A1T1'],conditions=['real','null'],features=['real_sum','real_mean','real_min','real_first','null_sum','null_mean','null_min','null_first','difference_sum','difference_mean','difference_min','difference_first','subword_count'],
 existing_AutoDL_training_sources=sources,processor='Actual existing Qwen3ASRModel.from_pretrained entry point used by existing ASR training; no tokenizer regex override',
 new_code_sha256=sha(__file__),adapter_sha256=sha(A/'adapter_model.safetensors'),runtime=dict(torch=torch.__version__,transformers=transformers.__version__,peft=peft.__version__,tokenizers=tokenizers.__version__),
 no_school_internal_files_imported=True,labels_read=False,optimizer_updates=0,chunk_size=256,resource=dict(GPU=0,cap_GiB=6,start_free_GiB=10,reserve_GiB=4,stable_idle_seconds=10,release_after_every_chunk=True),
 selection='Frozen checkpoint and fixed inputs; no DEV/VALID labels, feature selection or checkpoint selection.',
 limits='Native likelihood is a feature, not official correctness probability. Null audio activations are OOD. Human-ASR TRAIN ancestors include2strict24 rows; TRAIN features not OOF. Original4059 reused. Fresh same5090 checks do not erase earlier crossdevice failures.')
 protocol.update(variants=['full_A1T1'],features=read(ROOT/'fixed_recipe.json')['added_feature_names'],word_aggregates=['sum','mean','min','first'],word_features=32,fixed_recipe_sha256=sha(ROOT/'fixed_recipe.json'),historical_witness_source_sha256=sha(ROOT.parent/'posterior_distribution_witness_v1/run_witness.py'))
 save('protocol.json',protocol)
 completed=0;forwards=0
 def acquire():
  stable=0
  while True:
   r=subprocess.run(['nvidia-smi','--id=0','--query-gpu=memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=10);assert r.returncode==0
   free,util=map(int,r.stdout.strip().split(','));stable=stable+1 if free>=10240 and util<=5 else 0
   status(dict(stage='waiting_idle_GPU',completed_rows=completed,total_rows=22380,GPU_free_MiB=free,GPU_utilization=util))
   if stable>=2:break
   time.sleep(10)
  _,total=torch.cuda.mem_get_info();torch.cuda.set_per_process_memory_fraction(6*1024**3/total)
 acquire();status(dict(stage='loading_existing_native_model',completed_rows=0,total_rows=22380))
 wrapper=Qwen3ASRModel.from_pretrained(str(M),dtype=torch.bfloat16,device_map=None)
 native=wrapper.model;processor=wrapper.processor
 prefix=processor.apply_chat_template([[{'role':'system','content':''},{'role':'user','content':[{'type':'audio','audio':None}]}]],add_generation_prompt=True,tokenize=False)[0]+'language English<asr_text>'
 def grid(text,n_audio):
  expanded=processor.replace_multimodal_special_tokens([prefix+text],iter([int(n_audio)]))[0]
  ep=processor.replace_multimodal_special_tokens([prefix],iter([int(n_audio)]))[0]
  encoded=processor.tokenizer(expanded,add_special_tokens=False,return_offsets_mapping=True)
  pre=processor.tokenizer(ep,add_special_tokens=False)['input_ids']
  assert expanded==ep+text and encoded['input_ids'][:len(pre)]==pre
  spans=[m.span() for m in re.finditer(r'\S+',text)];owner=[]
  for left,right in encoded['offset_mapping'][len(pre):]:
   left-=len(ep);right-=len(ep);assert 0<=left<right<=len(text)
   nonblank=[k for k in range(left,right) if not text[k].isspace()]
   if nonblank:
    matches=[j for j,(a,b) in enumerate(spans) if a<=nonblank[0] and nonblank[-1]<b];assert len(matches)==1
    owner.append(matches[0])
   else:
    following=[j for j,(a,b) in enumerate(spans) if a>=right];assert following;owner.append(following[0])
  ids=np.asarray(encoded['input_ids'][len(pre):],np.int64);own=np.asarray(owner,np.int64)
  assert len(ids)==len(own) and set(own)==set(range(len(spans))) and (np.diff(own)>=0).all()
  return dict(ids=ids,owner=own,prefix_length=len(pre),full_ids=np.asarray(encoded['input_ids']),word_count=len(spans))
 # Full public lexical grid, no audio or labels, two placeholder lengths.
 prompts=sorted(set(v['text'] for v in rows));cases=0
 for text in prompts:
  for length in [17,37]:grid(text,length);cases+=1
 save('public_word_grid_check.json',dict(unique_prompts=len(prompts),cases=cases,all_words_covered=True,tokenizer_backend_sha256=hashlib.sha256(processor.tokenizer.backend_tokenizer.to_str().encode()).hexdigest(),prefix=prefix))
 def prepare(row):
  p=pathlib.Path(row['audio']);stat=p.stat();h=sha(p);wave,_=librosa.load(p,sr=16000,mono=True);now=p.stat();assert (stat.st_size,stat.st_mtime_ns)==(now.st_size,now.st_mtime_ns)
  x=processor(text=[prefix+row['text']],audio=[wave],return_tensors='pt',padding=True,truncation=False).to('cpu').to(torch.bfloat16)
  m=grid(row['text'],int((x['input_ids']==native.config.thinker_config.audio_token_id).sum()))
  assert m['word_count']==row['words'] and np.array_equal(x['input_ids'][0].numpy(),m['full_ids'])
  return x,m,h
 def base_hash():
  h=hashlib.sha256()
  for name,param in sorted([(n.replace('.base_layer.','.'),p) for n,p in native.named_parameters() if '.lora_' not in n],key=lambda t:t[0]):
   h.update(name.encode()+b'\0');h.update(param.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
  return h.hexdigest()
 expected_base='a9b782d7315ca7099cc595babf6400f492b2625818c6be5354fc18215e7123f3'
 assert base_hash()==expected_base
 native.to(D).eval().requires_grad_(False);assert native.thinker.model.embed_tokens.weight is native.thinker.lm_head.weight

 native.to('cpu');native.thinker.rope_deltas=None;torch.cuda.empty_cache()
 pm=peft.PeftModel.from_pretrained(native,str(A),is_trainable=False,autocast_adapter_dtype=True);native=pm.get_base_model()
 source=load_file(str(A/'adapter_model.safetensors'),device='cpu')
 def adapters_exact():
  current={n.replace('.default.','.'):p for n,p in pm.named_parameters() if '.lora_A.' in n or '.lora_B.' in n}
  return set(current)==set(source) and len(current)==208 and sum(p.numel() for p in current.values())==2392064 and all(p.dtype==torch.float32 and torch.equal(p.detach().cpu(),source[n]) for n,p in current.items())
 assert adapters_exact()
 for name,module in native.named_modules():
  if isinstance(module,LoraLayer):module.enable_adapters(True)
 acquire();pm.to(D).eval().requires_grad_(False)
 def distribution_logits(x,m,null):
  batch={k:v.to(D) for k,v in x.items()};hook=None
  if null:
   def remove_audio(module,args,out):
    result=copy.copy(out);result.last_hidden_state=torch.zeros_like(out.last_hidden_state);return result
   hook=native.thinker.audio_tower.register_forward_hook(remove_audio)
  try:
   with torch.inference_mode():out=native.thinker(**batch,use_cache=False,return_dict=True)
  finally:
   if hook is not None:hook.remove()
  pos=torch.arange(m['prefix_length']-1,batch['input_ids'].shape[1]-1,device=D)
  logits=out.logits[0,pos].float();assert logits.shape==(len(m['ids']),151936) and torch.isfinite(logits).all()
  target=torch.as_tensor(m['ids'],device=D);lp=logits.log_softmax(-1).gather(1,target[:,None]).squeeze(1).cpu().numpy()
  return logits,lp
 def gpu_stats(r,n):
  lr=r.double().log_softmax(-1);ln=n.double().log_softmax(-1);pr=lr.exp();pn=ln.exp();lm=torch.logaddexp(lr,ln)-__import__('math').log(2)
  tr=lr.topk(2,dim=-1).values;tn=ln.topk(2,dim=-1).values
  return torch.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,0]-tr[:,1],tn[:,0]-tn[:,1],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*(pr-pn).abs().sum(-1)],-1).cpu().numpy()
 def cpu_stats(r,n):
  r=r.astype(np.float64);n=n.astype(np.float64)
  def normalized(a):
   v=a-a.max(-1,keepdims=True);return v-np.log(np.exp(v).sum(-1,keepdims=True))
  lr=normalized(r);ln=normalized(n);pr=np.exp(lr);pn=np.exp(ln);lm=np.logaddexp(lr,ln)-np.log(2.)
  tr=np.partition(r,-2,axis=-1)[:,-2:];tn=np.partition(n,-2,axis=-1)[:,-2:]
  return np.stack([-(pr*lr).sum(-1),-(pn*ln).sum(-1),tr[:,1]-tr[:,0],tn[:,1]-tn[:,0],(pr*(lr-ln)).sum(-1),(pn*(ln-lr)).sum(-1),.5*((pr*(lr-lm)).sum(-1)+(pn*(ln-lm)).sum(-1)),.5*np.abs(pr-pn).sum(-1)],-1)

 W=ROOT.parent/'posterior_distribution_witness_v1/extraction_v1'
 old=ROOT.parent/'full_cache_replay_repair_v1'
 original=read(old/'manifest.json');assert original['rows']==22380 and len(original['chunks'])==88
 assert read(W/'INDEPENDENT_CPU_WITNESS_REPLAY.json')['word_reaggregation_maxgap']==0
 with np.load(W/'native_distribution_witness16.npz',allow_pickle=False) as z: witness={k:z[k].copy() for k in z.files}
 rowmap={v['signal']:v for v in rows};checks=[];forwards=0
 for j,signal in enumerate(witness['signals'].tolist()):
  x,m,h=prepare(rowmap[signal]);r,rp=distribution_logits(x,m,False);n,np_=distribution_logits(x,m,True);forwards+=2
  lo,hi=map(int,witness['token_offsets'][j:j+2]);v=gpu_stats(r,n);cpu=cpu_stats(r.cpu().numpy(),n.cpu().numpy())
  gap=float(abs(v-cpu).max());assert gap<=1e-9 and np.array_equal(np.stack([rp,np_],-1),witness['reference_logp'][lo:hi])
  assert np.array_equal(m['owner'],witness['token_to_word'][lo:hi]) and float(abs(v-witness['token_features'][lo:hi]).max())<=1e-9
  if j==0:
   again,ap=distribution_logits(x,m,False);forwards+=1;assert torch.equal(r,again) and np.array_equal(rp,ap);del again
  checks.append(gap);del r,n,v,cpu
 peak=torch.cuda.max_memory_allocated()/2**20;assert peak<=6144
 pm.to('cpu');native.thinker.rope_deltas=None;torch.cuda.empty_cache()
 assert adapters_exact() and base_hash()==expected_base
 save('GPU_preflight_report.json',dict(status='fresh16_full_vocab_CPU64_and_existing_witness_reference_exact',CPU64_maxgap=max(checks),rows=16,first_full_logits_repeat_exact=True,all208_FP32_adapter_tensors_exact=True,public_base_unchanged=True,resident_peak_MiB=peak,historical_witness_preflight_sha256=sha(W/'GPU_preflight_report.json'),historical_witness_independent_sha256=sha(W/'INDEPENDENT_CPU_WITNESS_REPLAY.json'),labels_read=False,optimizer_updates=0))
 chunks=[]
 for record in original['chunks']:
  lo,hi=record['start'],record['stop'];part=rows[lo:hi];src=old/record['file'];assert sha(src)==record['sha256']
  with np.load(src,allow_pickle=False) as z: reference={k:z[k].copy() for k in z.files}
  assert reference['signals'].tolist()==[v['signal'] for v in part]
  pack=[prepare(row) for row in part]
  acquire();pm.to(D).eval().requires_grad_(False);torch.cuda.reset_peak_memory_stats()
  lengths=[len(m['ids']) for x,m,h in pack];offset=np.r_[0,np.cumsum(lengths)]
  assert np.array_equal(offset,reference['token_offsets'])
  features=np.zeros((len(part),12,32),np.float32);tokens=[];logps=[]
  for j,(row,(x,m,h)) in enumerate(zip(part,pack)):
   free,total=torch.cuda.mem_get_info()
   if free<4*1024**3+max(0,6*1024**3-torch.cuda.memory_allocated()):
    pm.to('cpu');native.thinker.rope_deltas=None;torch.cuda.empty_cache();acquire();pm.to(D).eval()
   a,b=map(int,offset[j:j+2])
   assert h==reference['audio_sha256'][j] and np.array_equal(m['ids'],reference['token_ids'][a:b]) and np.array_equal(m['owner'],reference['token_to_word'][a:b])
   assert np.array_equal(reference['wordmask'][j],np.arange(12)<m['word_count'])
   r,rp=distribution_logits(x,m,False);n,np_=distribution_logits(x,m,True);forwards+=2
   lp=np.stack([rp,np_],-1);assert np.array_equal(lp,reference['token_logp'][a:b,:,1]),'Fresh fullnative reference support differs from winning source'
   v=gpu_stats(r,n);assert np.isfinite(v).all() and v[:,4:7].min()>=-1e-10 and v[:,6].max()<=np.log(2)+1e-10 and v[:,7].min()>=-1e-10 and v[:,7].max()<=1+1e-10
   for wi in range(m['word_count']):
    z=v[m['owner']==wi];features[j,wi]=np.concatenate([z.sum(0),z.mean(0),z.min(0),z[0]]).astype(np.float32)
   tokens.append(v);logps.append(lp);del r,n,v
   if j%16==0:status(dict(stage='extracting_full_posterior',completed_rows=lo,chunk_row=j,total_rows=22380,native_forwards=forwards))
  peak=torch.cuda.max_memory_allocated()/2**20;assert peak<=6144
  pm.to('cpu');native.thinker.rope_deltas=None;torch.cuda.empty_cache();assert np.isfinite(features).all()
  name=record['file']
  with (R/name).open('xb') as f:np.savez_compressed(f,signals=reference['signals'],wordmask=reference['wordmask'],word_features=features,token_features=np.concatenate(tokens),token_ids=reference['token_ids'],token_to_word=reference['token_to_word'],token_offsets=offset,reference_logp=np.concatenate(logps),audio_sha256=reference['audio_sha256'])
  chunks.append(dict(file=name,start=lo,stop=hi,sha256=sha(R/name),peak_MiB=peak,winning_source_sha256=record['sha256']));completed=hi
  status(dict(stage='chunk_sealed_GPU_released',completed_rows=completed,total_rows=22380,native_forwards=forwards));print('FULL_POSTERIOR_CHUNK',completed,22380,flush=True)
 assert completed==22380 and adapters_exact() and base_hash()==expected_base
 save('manifest.json',dict(chunks=chunks,rows=completed,protocol_sha256=sha(R/'protocol.json'),source_sha256=sha(__file__),winning_cache_manifest_sha256=sha(old/'manifest.json')))
 save('report.json',dict(status='all22380_full151936_native_posterior_evidence_produced',rows=completed,production_forwards=44760,native_forwards=forwards,all_reference_logps_exact_to_winning_source=True,all208_FP32_adapters_unchanged=True,public_base_unchanged=True,manifest_sha256=sha(R/'manifest.json'),labels_read=False,optimizer_updates=0,seconds=time.monotonic()-start,accuracy_claim=False))
 (R/'exitcode').write_text('0');print('FULL_POSTERIOR_PRODUCTION_COMPLETE',flush=True)
try:run()
except BaseException as e:
 traceback.print_exc();save('failure.json',dict(error=type(e).__name__,message=str(e),training=False,automatic_retry=False,seconds=time.monotonic()-start));(R/'exitcode').write_text('1');raise
