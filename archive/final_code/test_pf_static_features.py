"""Frozen Y/Z feature export with original VALID feature controls. No fitting."""
import collections,hashlib,json,pathlib
from functools import lru_cache
import numpy as np
import nltk
B=pathlib.Path(__file__).parent;V=B.parents[2];O=B/'outputs';C=O/'TEST_components'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def lines(paths,key):
 out={}
 for p in paths:
  for line in p.read_text().splitlines():
   r=json.loads(line);out[r[key]]=r
 return out
nltk.data.path.insert(0,str(V/'third_round_20261023/assets/nltk_data'))
from nltk.corpus import wordnet as wn
wn.ensure_loaded()
lex=read(V/'final15h_20261022/Y1/beep_dictionary.json')
lexical=read(V/'final15h_20261022/Y3/lexical_table.json')
prior=read(V/'third_round_20261023/Z13_repair/external_word_prior.json')
fill=read(V/'third_round_20261023/Z8_measurement.json')['neutral_margin']
@lru_cache(None)
def pron(w):return {str(x) for x in lex.get(w,[])}
@lru_cache(None)
def lemma(w):return wn.morphy(w,wn.VERB) or wn.morphy(w,wn.NOUN) or w
def z8(row,words):
 cnt=collections.Counter(row['mapping']);x=[]
 for word,origin in zip(words,row['mapping']):
  choices=[]
  if row['pieces'][origin]==1 and cnt[origin]==1:
   for w,lp in row['alternatives'][origin]:
    if w==word or lemma(w)==lemma(word) or (pron(w) and pron(word) and pron(w)&pron(word)):continue
    choices.append((w,lp))
  margin=row['surface_target_logp'][origin]-choices[0][1] if choices else fill
  x.append([margin,row['sentence_PLL']])
 return np.asarray(x)
def z11(words,sev):
 x=np.zeros((len(words),6))
 for j,(w,pos) in enumerate(nltk.pos_tag(words,lang='eng')):
  cat=None
  if w.endswith("'s") and len(w)>3:cat=1
  elif w.endswith(('s','es')) and pos in ['NNS','VBZ'] and any(wn.morphy(w,p) not in [None,w] for p in [wn.NOUN,wn.VERB]):cat=0
  elif w.endswith('ed') and pos in ['VBD','VBN'] and wn.morphy(w,wn.VERB) not in [None,w]:cat=2
  if cat is not None and sev in [1,2]:x[j,cat*2+sev-1]=1
 return x
def features(rows,width,links,evidence):
 output={k:np.zeros((len(rows),width,d)) for k,d in [('Z2',2),('Z8',2),('Z11',6),('Z9',6),('Z13',1)]}
 output.update({k:np.zeros((len(rows),d),np.float32) for k,d in [('Y2',18),('Y2-TF',12),('Y3',1),('Y4',18)]})
 for i,r in enumerate(rows):
  sid=r['signal_id'];words=r['prompt'].split();n=len(words);sev=r['severity'];lk=links[sid]
  for k in ['Y2','Y2-TF','Y4']:
   q=np.asarray(evidence[k][sid]['x']).mean(0);output[k][i]=np.r_[q,q*(sev==1),q*(sev==2)]
  output['Y3'][i]=np.mean([lexical['effects'].get(w,dict(effect=0.))['effect'] for w in words])
  output['Z2'][i,:n]=evidence['Z2'][lk['text_key']]['x']
  output['Z8'][i,:n]=z8(evidence['Z8'][lk['text_key']],words)
  output['Z11'][i,:n]=z11(words,sev)
  output['Z9'][i,:n]=evidence['Z9'][lk['audio_key']]['x']
  output['Z13'][i,:n,0]=[prior['word_effects'].get(w,0.) for w in words]
 return output
def main():
 z=np.load(V/'fourth_round_20261023/V11_replay/joint_fit_inputs_conditional.npz',allow_pickle=False)
 meta={r['signal_id']:r for r in read(V/'new_angles_20261021/music_witness_stage1/valid_metadata_features_only.json')}
 valid=[meta[str(s)] for s in z['valid_signal_ids']];links=read(V/'second_round_20261023/model_row_links.json')
 evidence={
 'Y2':lines((V/'final15h_20261022/Y2/results').glob('*/evidence.jsonl'),'signal_id'),
 'Y2-TF':lines((V/'final15h_20261022/Y2-TF/results').glob('*/evidence.jsonl'),'signal_id'),
 'Y4':lines((V/'final15h_20261022/Y4/samewave_repair/results').glob('*/full/*/evidence.jsonl'),'signal_id'),
 'Z2':lines((V/'second_round_20261023/model_results').glob('text_*/evidence.jsonl'),'text_key'),
 'Z8':lines((V/'third_round_20261023/model_results').glob('text_*/evidence.jsonl'),'text_key'),
 'Z9':lines((V/'third_round_20261023/model_results').glob('scene_*/evidence.jsonl'),'audio_key')}
 f=features(valid,11,links,evidence);proof={}
 for k,x in f.items():
  gap=float(np.max(abs(x-z['valid_'+k])));proof[k]=dict(VALID_rows=2117,max_abs_gap=gap);assert gap<1e-6,(k,gap)
 print('VALID cached-feature qualification',json.dumps(proof),flush=True)
 paths={'Y2':'Y2_BLIND','Y2-TF':'Y2_TF','Y4':'Y4','Z2':'Z2','Z8':'Z8','Z9':'Z9'}
 new={k:lines([O/('TEST_'+p)/'collected/evidence.jsonl'],'signal_id' if k.startswith('Y') else 'text_key' if k in ['Z2','Z8'] else 'audio_key')for k,p in paths.items()}
 # Test exact consumed Z8 features, including native equal-logit top-k ties.
 ckeys=[r['text_key'] for r in read(O/'TEST_Z8/VALID_expected.json')];textsel={r['text_key']:r for r in read(O/'TEST_Z8/text_selection.json')}
 tieproof=[]
 for key in ckeys:
  ws=textsel[key]['prompt'].split();gap=float(abs(z8(new['Z8'][key],ws)-z8(evidence['Z8'][key],ws)).max());assert gap==0.;tieproof.append(dict(text_key=key,consumed_Z8_feature_max_abs_gap=gap))
 erows=[dict(signal_id=r['signal'],prompt=r['prompt'],severity=r['severity']) for r in read(O/'TEST_LMY_NATIVE/selection.json')[2:]]
 assert len(erows)==2189 and len({r['signal_id']for r in erows})==2189
 elinks=read(O/'TEST_Z2/EVAL_row_links.json');ef=features(erows,12,elinks,new)
 for k,x in ef.items():assert np.isfinite(x).all(),k
 target=C/'PF_static_features.npz'
 with target.open('xb') as fp:np.savez_compressed(fp,signal_ids=np.asarray([r['signal_id']for r in erows]),**ef)
 report=dict(rows=2189,VALID_feature_qualification=proof,Z8_native_tie_consumed_feature_qualification=tieproof,training_updates=0,output_SHA=sha(target),input_SHA={str(O/('TEST_'+p)/'collected/evidence.jsonl'):sha(O/('TEST_'+p)/'collected/evidence.jsonl')for p in paths.values()},frozen_assets_SHA={str(p):sha(p)for p in [V/'final15h_20261022/Y1/beep_dictionary.json',V/'final15h_20261022/Y3/lexical_table.json',V/'third_round_20261023/Z13_repair/external_word_prior.json',V/'third_round_20261023/Z8_measurement.json']})
 (C/'PF_static_features_qualification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
if __name__=='__main__':main()
