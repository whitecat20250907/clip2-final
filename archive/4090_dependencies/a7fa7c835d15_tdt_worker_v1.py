import os,sys,pathlib,json,hashlib,time
sys.dont_write_bytecode=True
R=pathlib.Path(__file__).resolve().parent;p=json.loads((R/'TDT_PLAN_V1.json').read_text())
os.environ.update(CUDA_VISIBLE_DEVICES=p['gpu_uuid'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1',TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
for k in ['HF_HOME','HF_HUB_CACHE','TORCH_HOME','CUDA_CACHE_PATH','TORCHINDUCTOR_CACHE_DIR','XDG_CACHE_HOME','TMPDIR']:
 d=R/'runtime'/k;d.mkdir(parents=True,exist_ok=True);os.environ[k]=str(d)
import numpy as np,torch
torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
from native_tdt_core_v1 import NativeTDT
def read(f):return json.loads(pathlib.Path(f).read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def put(n,v,compact=False):
 raw=json.dumps(v,indent=None if compact else 2,sort_keys=True,ensure_ascii=False,allow_nan=False,separators=(',',':')if compact else None).encode()
 with(R/n).open('xb')as f:f.write(raw)
def progress(**v):
 f=R/'progress.next.json';f.write_text(json.dumps(v));os.replace(f,R/'progress.json')
start=time.monotonic()
for n,h in read(R/'TDT_BINDING_V1.json')['files'].items():assert sha(R/n)==h
assert sha(p['roles'])==p['roles_sha']
with np.load(p['roles'],allow_pickle=False)as z:train=z['strict_TRAIN_ids'].tolist();fit=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist();dev=z['formal_DEV_ids'].tolist()
assert len(train)==16180 and digest(train)==p['TRAIN_ID_SHA']and len(dev)==4059 and digest(dev)==p['DEV_ID_SHA']and set(fit+held)==set(train)and not set(train)&set(dev)
C=pathlib.Path(p['cache']['root']);assert sha(C/'index.json')==p['cache']['index_sha']and sha(p['cache']['source'])==p['cache']['source_sha'];index=read(C/'index.json');assert index['n']==20297 and index['dim']==1024 and index['dtype']=='float16'and index['model']==p['cache']['model']and(C/'features.f16').stat().st_size==index['total_tokens']*1024*2
ci={s:i for i,s in enumerate(index['signal_ids'])};assert set(train)<=set(ci);mm=np.memmap(C/'features.f16',mode='r',dtype=np.float16,shape=(index['total_tokens'],1024))
def frames(s):
 i=ci[s];off=index['offsets'][i];n=index['lengths'][i];return np.asarray(mm[off:off+n],dtype=np.float32).copy()
put('TDT_SCOPE_BEFORE_NATIVE_INFERENCE_V1.json',dict(TRAIN_rows=16180,TRAIN_ID_SHA=digest(train),qualification_public_FIT_rows=32,formal_DEV_not_selected=True,read_count_labels=False,read_human_transcriptions=False,raw_input='FP16cached1024->F32->encoder_projector640ONCE;noRMS/LN/interpolation',cache_index_sha=p['cache']['index_sha']))
assert torch.cuda.is_available();free,total=torch.cuda.mem_get_info();assert free>=3*1024**3;torch.cuda.set_per_process_memory_fraction(2*1024**3/total);engine=NativeTDT(p);put('TDT_NATIVE_WEIGHT_SCOPE_V1.json',dict(weight_keys=engine.weight_keys,weight_tensor_sha=engine.weight_sha,parameters=engine.parameters,snapshot=p['snapshot'],config_sha=sha(pathlib.Path(p['snapshot'])/'config.json'),generation_config_sha=sha(pathlib.Path(p['snapshot'])/'generation_config.json'),tokenizer_sha=p['tokenizer_sha'],modeling_sha=p['modeling_sha'],generation_sha=p['generation_sha'],new_ASR_updates=0,encoder_weights_loaded=False))
qual=fit[:32];arrays=[frames(s)for s in qual];proof=[]
with torch.inference_mode():
 records,traces=engine.decode_many(arrays,trace_first=4);cap=engine.cfg.max_symbols_per_step*max(map(len,arrays))-1
 for j,s in enumerate(qual[:4]):
  ref,logits=engine.prefix_reference(arrays[j],cap);actual=records[j];assert all(ref[k]==actual[k]for k in ['raw_tokens','durations','frames']),(s,'native_cache_vs_full_prefix_path')
  assert len(logits)==len(traces[j]);gap=max(float(np.max(abs(a-b)))for a,b in zip(logits,traces[j]));assert gap<1e-4,(s,gap)
  proof.append(dict(signal=s,joint_logit_maxgap=gap,token_duration_frame_path_exact=True,steps=len(logits),native_text_matches_reference=engine.text(ref['raw_tokens'])==actual['raw_text']))
 probe=next(v for v in engine.tok.encode('hello',add_special_tokens=False).ids if engine.tok.id_to_token(v)and engine.tok.id_to_token(v).strip('▁'))
 test=[engine.blank,probe,probe,engine.eos,2,0];assert engine.text(test)==engine.ref_tok.decode(test,skip_special_tokens=True);assert engine.text([probe,probe])!=engine.text([probe])
put('TDT_NATIVE_CACHE_QUALIFICATION_V1.json',dict(actual_public_FIT_cases=proof,source_classes_AST_reused=True,batch_size=32,independent_full_nonblank_prefix_LSTM=True,no_CTC_repeat_collapse=True,EOS_special_registration_parity=True,explicit_duplicate_token_test=True,no_count_or_human_labels_read=True))
put('TDT_QUALIFIED32_RECORDS_V1.json',dict(zip(qual,records)),compact=True);bank=dict(zip(qual,records));todo=[s for s in train if s not in bank];progress(stage='native_TDT_qualified_start_frozen_TRAIN_decode',qualified32=True,processed=len(bank),total=16180,seconds=time.monotonic()-start)
with torch.inference_mode():
 for pos in range(0,len(todo),32):
  ids=todo[pos:pos+32];recs,_=engine.decode_many([frames(s)for s in ids])
  for s,v in zip(ids,recs):bank[s]=v
  if(pos//32+1)%8==0:progress(stage='frozen_native_TDT_TRAIN_decode',processed=len(bank),total=16180,seconds=time.monotonic()-start)
assert set(bank)==set(train)and not set(bank)&set(dev)
put('TDT_NATIVE_STRICT_TRAIN_RECORDS_V1.json',{s:bank[s]for s in train},compact=True)
put('TDT_TRAIN_RECORDS_SEALED_V1.json',dict(UTC=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),TRAIN_rows=16180,TRAIN_ID_SHA=digest(train),records_sha=sha(R/'TDT_NATIVE_STRICT_TRAIN_RECORDS_V1.json'),source_binding_sha=sha(R/'TDT_BINDING_V1.json'),decoder_tensor_sha=engine.weight_sha,qualification_sha=sha(R/'TDT_NATIVE_CACHE_QUALIFICATION_V1.json'),formal_DEV_selected=False,count_or_human_labels_read=False,new_ASR_updates=0,cap_rows=sum(v['summary']['cap_hit']for v in bank.values()),EOS_rows=sum(v['summary']['EOS']for v in bank.values()),seconds=time.monotonic()-start))
put('TDT_WORKER_TERMINAL_V1.json',dict(actual_exit_code=0));print('TDT_STRICT_TRAIN_COMPLETE',len(bank),time.monotonic()-start,flush=True)
