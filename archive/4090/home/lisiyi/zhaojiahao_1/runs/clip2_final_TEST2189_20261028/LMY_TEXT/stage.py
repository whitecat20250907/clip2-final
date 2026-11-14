ROOT='/home/lisiyi/zhaojiahao_1/runs/clip2_final_TEST2189_20261028/LMY_TEXT'

import pathlib,json,hashlib,subprocess,os
r=pathlib.Path(ROOT);exps=pathlib.Path('/home/lisiyi/clip2_lmy/CLIP2_lmy/my_experiments')
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
cfg=dict(tf_overlay=str(r.parent/'LMY_LISTENER/tf_overlay'),mapping=dict(path=str(exps/'d6e_pooled_word_interaction_v1/word_inputs.py'),SHA=sha(exps/'d6e_pooled_word_interaction_v1/word_inputs.py')),metadata='/home/yangjing/workplace/clip2/clip2_eval/cadenza_data/clip2/metadata/eval_metadata.json',train_metadata='/home/lisiyi/icassp2data/clip2/metadata/train_metadata.json',valid_metadata='/home/lisiyi/ICASSP2_2026/data/clip2/metadata/valid_metadata.json',controls=['4c1305298289cc31081a508c','15816b64c1e78664a75a360a'],model='/home/lisiyi/models/roberta-base',cache='/home/yangjing/workplace/clip2/clip2_next/output/text_emb_roberta_wid_all.npz')
(r/'protocol.json').write_text(json.dumps(cfg,indent=2));files=list(r.glob('*.py'))+list(r.glob('*.json'))+[pathlib.Path(cfg[k])for k in ['metadata','train_metadata','valid_metadata','cache']]+[pathlib.Path(cfg['mapping']['path'])]+[p for p in pathlib.Path(cfg['model']).glob('*')if p.is_file()and p.suffix in ['.json','.txt','.safetensors']]
binding=dict(training_updates=0,wrapper_SHA=sha(r/'wrapper.py'),files=[dict(path=str(p),SHA=sha(p))for p in files],roles=['text_0'],minimum_free_MiB=3500,maximum_admission_utilization=10,admission_wait_seconds=900);(r/'binding.json').write_text(json.dumps(binding,indent=2));h=sha(r/'binding.json');py='/tmp/clip2_lisiyi_zhaojiahao_clean183_20261014/v10_private_runtime_20261019_0945/runtime/bin/python';cmd=[py,'-u','-B',str(r/'wrapper.py'),'--base',str(r),'--role','text_0','--gpu','0','--binding-sha',h]
with(r/'run.log').open('wb')as f:p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
result=dict(PID=p.pid,command=cmd,binding_SHA=h);(r/'stage.json').write_text(json.dumps(result,indent=2));print(json.dumps(result))
