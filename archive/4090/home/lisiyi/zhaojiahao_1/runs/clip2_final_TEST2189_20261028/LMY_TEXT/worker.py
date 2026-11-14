"""Frozen RoBERTa cache identity, original first-text mapping, fixed new-key encoding."""
import argparse,ast,hashlib,json,pathlib,re,sys,time,unicodedata

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=pathlib.Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
 sys.path.insert(0,cfg['tf_overlay'])
 import numpy as np,torch,transformers
 import transformers.utils.import_utils as av
 av._torchvision_available=False;av._torchaudio_available=False
 from transformers import AutoTokenizer,AutoModel
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(2*1024**3/torch.cuda.get_device_properties(0).total_memory)
 def sha(p):
  h=hashlib.sha256()
  with pathlib.Path(p).open('rb')as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  return h.hexdigest()
 def put(n,v):
  p=out/(n+'.next');p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));p.replace(out/n)
 def require(ok,msg):
  if not ok:raise ValueError(msg)
 def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()
 source=pathlib.Path(cfg['mapping']['path']);assert sha(source)==cfg['mapping']['SHA'];names={'words','source_sequence','unique_lcs','word_vectors','cache_encoded_texts'};nodes=[n for n in ast.parse(source.read_text()).body if isinstance(n,ast.FunctionDef)and n.name in names];assert{n.name for n in nodes}==names;env=dict(np=np,re=re,unicodedata=unicodedata,require=require,digest=digest);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(source),'exec'),env)
 rows=json.loads(pathlib.Path(cfg['metadata']).read_text());train=json.loads(pathlib.Path(cfg['train_metadata']).read_text());valid=json.loads(pathlib.Path(cfg['valid_metadata']).read_text());controls=[r for r in valid if r['signal']in cfg['controls']];assert len(rows)==2189 and len(controls)==2;ordered=controls+rows
 wanted={r['prompt']for r in ordered}|{r['prompt']for r in train[:32]};assert not (({r['prompt']for r in controls}-{r['prompt']for r in train})&{r['prompt']for r in rows});encoding,audit=env['cache_encoded_texts'](train+rows+controls,wanted);put('text_key_audit.json',audit)
 with np.load(cfg['cache'],allow_pickle=True)as z:emb,offsets,lengths=(z[k].copy()for k in ['emb','offsets','lengths']);keys=[str(p)for p in z['prompts']];model_id=str(z['model'].item())
 assert model_id in ['roberta-base','FacebookAI/roberta-base']and lengths.max()<=64
 tok=AutoTokenizer.from_pretrained(cfg['model'],local_files_only=True,use_fast=True);assert tok.is_fast and digest(tok.backend_tokenizer.to_str())=='071aba27009492a64caf65c00d04040fd781796f38d4b7b6630c988f3f2bc347'
 index={p:i for i,p in enumerate(keys)};missing=sorted(wanted-set(index));tokens={};vectors={};masks={}
 def accept(p,value):
  raw=encoding[p];encoded=tok(raw,truncation=True,max_length=64,add_special_tokens=True,return_offsets_mapping=True);assert value.shape==(len(encoded['input_ids']),768)and np.isfinite(value).all();value=value.astype(np.float32);v,k,_=env['word_vectors'](raw,p,encoded['offset_mapping'],value);tokens[p]=value[:64];vectors[p]=v;masks[p]=k
 for p in sorted(wanted&set(index)):
  j=index[p];accept(p,emb[int(offsets[j]):int(offsets[j])+int(lengths[j])])
 proof=[]
 if missing:
  encoder=AutoModel.from_pretrained(cfg['model'],local_files_only=True,torch_dtype=torch.float16).to('cuda').eval().requires_grad_(False)
  def encode(pp):
   batch=tok([encoding[p]for p in pp],return_tensors='pt',padding=True,truncation=True,max_length=64,add_special_tokens=True)
   with torch.inference_mode():h=encoder(**batch.to('cuda')).last_hidden_state
   return[h[i][batch['attention_mask'][i].bool()].float().cpu().numpy().astype(np.float16)for i in range(len(pp))]
  probe=sorted(wanted&set(index))[:8];assert len(probe)==8
  for p,got in zip(probe,encode(probe)):
   j=index[p];old=emb[int(offsets[j]):int(offsets[j])+int(lengths[j])];assert old.shape==got.shape;proof.append(dict(prompt=p,maxgap=float(abs(old.astype(np.float32)-got.astype(np.float32)).max())))
  put('TRAIN_text_coordinate_control.json',dict(cases=proof,limit=.02));assert max(x['maxgap']for x in proof)<=.02
  for lo in range(0,len(missing),32):
   pp=missing[lo:lo+32]
   for p,v in zip(pp,encode(pp)):accept(p,v)
   put('progress.json',dict(done=lo+len(pp),total=len(missing),seconds=time.monotonic()-start));print('TEXT',lo+len(pp),len(missing),flush=True)
  del encoder;torch.cuda.empty_cache()
 tl=np.array([len(tokens[r['prompt']])for r in ordered]);wl=np.array([r['n_words']for r in ordered]);assert all(len(vectors[r['prompt']])==r['n_words']for r in ordered)
 p=out/'EVAL_ROBERTA_TEXT.npz';np.savez_compressed(p,signals=np.asarray([r['signal']for r in ordered]),prompts=np.asarray([r['prompt']for r in ordered]),token_lengths=tl,tokens=np.concatenate([tokens[r['prompt']]for r in ordered]),word_lengths=wl,word_vectors=np.concatenate([vectors[r['prompt']]for r in ordered]),mapped=np.concatenate([masks[r['prompt']]for r in ordered]))
 put('complete.json',dict(rows=2189,VALID_control_rows=2,missing_keys=len(missing),fixed_tolerance=.02,TRAIN_text_coordinate_controls=proof,training_updates=0,output_SHA=sha(p),seconds=time.monotonic()-start))

if __name__=='__main__':main()
