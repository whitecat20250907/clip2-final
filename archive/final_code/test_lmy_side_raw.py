"""Original public LMY side inputs except the separately gated LM4 decoder."""
import argparse,ast,fcntl,hashlib,json,math,os,re,subprocess,time,types,unicodedata
from pathlib import Path

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--base');a=ap.parse_args();root=Path(a.base);assert str(root).startswith('/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028/')
 root.mkdir(exist_ok=True);lock=(root/'writer.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);out=root/'tasks/raw_0';out.mkdir(parents=True,exist_ok=True);(out/'rows').mkdir(exist_ok=True);start=time.monotonic()
 import numpy as np,soundfile as sf,soxr
 def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
 cfg=json.loads((root/'protocol.json').read_text());assert cfg['training_updates']==0
 for p,h in cfg['files'].items():assert sha(p)==h,p
 def put(n,x):
  p=out/(n+'.next');p.write_text(json.dumps(x,indent=2,allow_nan=False));p.replace(out/n)
 def module(key,names,constants=(),extra=None):
  p=Path(cfg['sources'][key]);env=dict(np=np,math=math,json=json,Path=Path,sf=sf,soxr=soxr,re=re,subprocess=subprocess,unicodedata=unicodedata,SIDE_FIELDS=('prob.mean','prob.min','prob.std','margin.mean','margin.min','margin.std'))
  if extra:env.update(extra)
  tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name in names or isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id in constants for t in n.targets)]
  assert {n.name for n in nodes if isinstance(n,ast.FunctionDef)}==set(names)
  exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env);return types.SimpleNamespace(**env)
 parse=module('parser',['_finite_float','extract_channel_features','load_confidence_jsonl_with_text','_normalize_text_for_wer','_word_error_rate','build_better_ear_wer_table','build_extra_wer_table'])
 phone=module('phone',['normalized_words','normalize_phone','phonemize_word','phone_class','strict_phone_class','alignment_feature_sets','phrase_phones','espeak_version'],['STOP','FRICATIVE','SONORANT','STRICT_STOP','STRICT_FRICATIVE_AFFRICATE','STRICT_SONORANT','NORMALIZE_DROP'])
 paired=module('paired',['read_stereo','band_power','correlation','shift_pair','channel_features'],['SAMPLE_RATE','N_FFT','HOP','BANDS'])
 assert phone.espeak_version()==cfg['espeak_version']
 rows=json.loads((root/'selection.json').read_text());assert len(rows)==2191
 asr={};texts={}
 for part in ['valid','eval']:
  for model in ['v3','v2','medium']:
   vv,tt=parse.load_confidence_jsonl_with_text(Path(cfg[part][model]));asr[part,model]=vv;texts[part,model]=tt
 gold={}
 for name in ['paired','strict','candidate']:
  gold[name]={r['signal']:r for r in map(json.loads,Path(cfg['valid'][name]).read_text().splitlines())}
 # Reuse only the original word-to-IPA dictionary; additions use the exact espeak binary/version.
 original=json.loads(Path(cfg['word_cache']).read_text());word_cache={k:tuple(v)for k,v in original['words'].items()}
 missing=set()
 for r in rows:
  part='valid'if r['role']=='VALID_CONTROL'else'eval';tt=texts[part,'v3'][r['signal']]
  for text in [r['prompt'],tt['left'],tt['right']]:missing.update(set(phone.normalized_words(text))-set(word_cache))
 for i,w in enumerate(sorted(missing)):
  word_cache[w]=phone.phonemize_word(w)
  if i%100==0:put('progress.json',dict(stage='exact_espeak_additions',done=i,total=len(missing)))
 put('new_word_IPA.json',{w:word_cache[w]for w in sorted(missing)})
 allv=[];proof=[]
 for i,r in enumerate(rows):
  sid=r['signal'];part='valid'if i<2 else'eval';saved=out/'rows'/(sid+'.npz')
  if saved.exists():
   with np.load(saved,allow_pickle=False)as z:vals={k:z[k].copy()for k in z.files}
  else:
   ref=phone.phrase_phones(r['prompt'],word_cache);tt=texts[part,'v3'][sid];left=phone.phrase_phones(tt['left'],word_cache);right=phone.phrase_phones(tt['right'],word_cache);assert len(ref)>0
   counts=np.zeros(3,np.float64)
   for token in ref:
    c=phone.strict_phone_class(token)
    if c is not None:counts[c]+=1
   lr,lb,_=phone.alignment_feature_sets(ref,left,classifier=phone.strict_phone_class);rr,rb,_=phone.alignment_feature_sets(ref,right,classifier=phone.strict_phone_class)
   strict=np.asarray((np.log1p(len(ref)),*(counts/len(ref)),lr,rr,*lb,*rb),np.float32)
   x=paired.read_stereo(Path(r['audio']));y=paired.read_stereo(Path(r['audio2']));acoustic=np.asarray([v for ear in [0,1]for v in paired.channel_features(x[:,ear],y[:,ear])],np.float32)
   candidate=json.loads((Path(cfg[part]['k5'])/(sid+'.json')).read_text())['binaural_summary']['candidate_max_overall']
   wer=parse.build_better_ear_wer_table(texts[part,'v3'],[r],clip_max=1.)[sid];extra=[]
   for model,mode in [('medium','better_ear'),('v2','better_ear'),('v3','gap'),('v2','gap')]:extra.extend(np.asarray(parse.build_extra_wer_table(texts[part,model],[r],mode=mode,clip_max=1.,source_name=model+'_'+mode)[sid]).reshape(-1).tolist())
   vals=dict(confidence_v3=asr[part,'v3'][sid],confidence_v2=asr[part,'v2'][sid],wer=np.asarray([wer],np.float32).reshape(1),extra_wer=np.asarray(extra,np.float32),candidate=np.asarray([candidate],np.float32),paired_acoustic=acoustic,strict=strict)
   assert all(np.isfinite(v).all()for v in vals.values());tmp=saved.with_suffix('.next.npz');np.savez_compressed(tmp,**vals);tmp.replace(saved)
  if i<2:
   gaps={name:float(abs(vals[key]-np.asarray(gold[name][sid][field],np.float32)).max())for name,key,field in [('paired','paired_acoustic','features'),('strict','strict','features'),('candidate','candidate','candidate_max_overall')]}
   proof.append(dict(signal=sid,gaps=gaps));put('VALID_control_observations.json',dict(cases=proof));assert max(gaps.values())<=2e-5,proof
   if i==1:put('VALID_control_passed.json',dict(cases=proof))
  allv.append(vals)
  if i%16==0 or i==len(rows)-1:put('progress.json',dict(stage='public_side_raw',done=max(0,i-1),total=2189,seconds=time.monotonic()-start));print('DONE',max(0,i-1),flush=True)
 p=out/'EVAL_LMY_PUBLIC_SIDE_RAW.npz';np.savez_compressed(p,signals=np.asarray([r['signal']for r in rows]),**{k:np.stack([r[k]for r in allv])for k in allv[0]});put('complete.json',dict(rows=2189,VALID_control_rows=2,output_SHA=sha(p),training_updates=0,LM4_separate_required=True,seconds=time.monotonic()-start,VALID_controls=proof))

if __name__=='__main__':main()
