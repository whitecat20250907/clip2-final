"""Frozen existing two-fold ASR on Audio1 only. No references/targets loaded."""
import sys,os,pathlib,json,hashlib,time,traceback,gc
sys.dont_write_bytecode=True
ROOT=pathlib.Path(__file__).resolve().parent
for k in ['TMPDIR','XDG_CACHE_HOME','TORCH_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR','HF_HOME','TRANSFORMERS_CACHE','TRITON_CACHE_DIR']:
 p=ROOT/'runtime'/k;p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
os.environ.update(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def put(n,o):
 with (ROOT/n).open('x') as f:json.dump(o,f,indent=2,allow_nan=False)
def progress(stage,**kw):
 d=dict(stage=stage,PID=os.getpid(),**kw);p=ROOT/'production_progress.next.json';p.write_text(json.dumps(d));os.replace(p,ROOT/'production_progress.json');print(json.dumps(d),flush=True)
start=time.monotonic()
try:
 import torch
 from peft import PeftModel
 from qwen_asr import Qwen3ASRModel
 torch.set_num_threads(2);torch.manual_seed(20261017);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 binding=read(ROOT/'PRODUCTION_SOURCE_BINDING_V1.json');assert sha(__file__)==binding['producer_sha256'] and sha(ROOT/'PREREGISTRATION_V1.json')==binding['plan_sha256']
 gate=read(ROOT/'PUBLIC_INPUT_AND_LINEAGE_GATE_V1.json');assert gate['status']=='6176public_generation_and16204TRAIN_OOF_boundary_verified'
 assert sha(ROOT/'generation_audio_records.json')==gate['generation_records_sha256'] and sha(ROOT/'teacher_sources.json')==gate['teacher_sources_sha256']
 records=read(ROOT/'generation_audio_records.json');teachers=read(ROOT/'teacher_sources.json')
 assert len(records)==6176 and all(set(r)=={'signal','audio','fold','role'} for r in records)
 free,total=torch.cuda.mem_get_info();assert free>=12*1024**3;torch.cuda.set_per_process_memory_fraction(.45)
 allrows=[];artifacts={}
 for fold in [0,1]:
  selected=[r for r in records if r['fold']==fold]
  source=teachers[str(fold)];cp=pathlib.Path(source['adapter_path']);assert sha(cp/'adapter_model.safetensors')==source['adapter_sha256']
  out=ROOT/('generated_fold%d.jsonl'%fold);assert not out.exists()
  progress('loading_fixed_ASR_teacher',fold=fold,total=len(selected),seconds=time.monotonic()-start)
  wrapper=Qwen3ASRModel.from_pretrained('/root/clip2_workspace/models/Qwen3-ASR-1.7B',dtype=torch.bfloat16,device_map=None,max_inference_batch_size=8,max_new_tokens=128)
  wrapper.model=PeftModel.from_pretrained(wrapper.model,str(cp)).merge_and_unload();wrapper.model.to('cuda').eval().requires_grad_(False);wrapper.model.generation_config.do_sample=False
  with out.open('x') as f:
   for lo in range(0,len(selected),8):
    free,_=torch.cuda.mem_get_info();assert free>=4*1024**3,'GPU reserve below4GiB; no automatic production retry'
    group=selected[lo:lo+8];audio=[r['audio'] for r in group]
    with torch.inference_mode():results=wrapper.transcribe(audio=audio,language='English')
    assert len(results)==len(group)
    for row,item in zip(group,results):
     result=dict(signal=row['signal'],text=str(item.text),fold=fold);f.write(json.dumps(result,ensure_ascii=False)+'\n');allrows.append(result)
    f.flush()
    if lo==0 or (lo+len(group))%256<8:progress('generating_Audio1_only',fold=fold,completed=lo+len(group),total=len(selected),seconds=time.monotonic()-start,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30)
  artifacts[out.name]=sha(out);assert sha(cp/'adapter_model.safetensors')==source['adapter_sha256']
  del wrapper;gc.collect();torch.cuda.empty_cache()
 byid={r['signal']:r for r in allrows};assert len(byid)==len(allrows)==6176
 ordered=[byid[r['signal']] for r in records]
 with (ROOT/'generated_DEV_VALID6176.jsonl').open('x') as f:
  for r in ordered:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 artifacts['generated_DEV_VALID6176.jsonl']=sha(ROOT/'generated_DEV_VALID6176.jsonl')
 put('ASR_PRODUCTION_COMPLETE_V1.json',dict(status='6176fixedteacher_machine_transcriptions_complete',rows=6176,fold_rows={str(f):sum(r['fold']==f for r in records) for f in [0,1]},artifacts=artifacts,input_gate_sha256=sha(ROOT/'PUBLIC_INPUT_AND_LINEAGE_GATE_V1.json'),producer_sha256=sha(__file__),ASR_training_updates=0,source_adapter_files_unchanged=True,ASR_training=False,reference_lyrics_loaded=False,DEV_truth_read=False,human_responses_read=False,seconds=time.monotonic()-start,peak_gpu_GiB=torch.cuda.max_memory_allocated()/2**30))
 progress('ASR_production_complete',rows=6176,seconds=time.monotonic()-start)
except BaseException as e:
 traceback.print_exc();put('ASR_PRODUCTION_FAILURE_V1.json',dict(error=type(e).__name__,message=str(e),PID=os.getpid(),automatic_retry=False));raise
