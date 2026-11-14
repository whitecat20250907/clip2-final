"""Verify every frozen native measurement before exporting one raw score bank."""
import argparse,hashlib,importlib.util,json,pathlib
import numpy as np

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--judge',choices=['MF','Phi4MM'],required=True);a=ap.parse_args();r=pathlib.Path(__file__).parent
 source=r/'test_inference.py';assert hashlib.sha256(source.read_bytes()).hexdigest()=='157f642e8ba20a6854778f76212fe509ccc1367198ef0267c4a3d38e998fe11c'
 sp=importlib.util.spec_from_file_location('frozen_point',source);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
 args=argparse.Namespace(config=str(r/'TEST_POINT_CONFIG.json'),config_sha='0c0053bf5fabba34690d6e4ca7458c1ca1f2961d18c05dfa88185f54e740b545',judge=a.judge)
 c,U,core,p,binding,model,tasks,identity=m.point_prepare(args);assert len(tasks)==26268
 scores=np.empty((2189,6,2));mass=np.empty_like(scores);manifest=[]
 prompts=[q['id']for q in p['prompts']];assert len(prompts)==len(set(prompts))==6
 for t in tasks:
  i=t['row_index'];pr=t['prompt'];ear=t['ear'];q=r/'measurements'/a.judge/(f"{i:04d}_{pr['id']}_{ear}.json");raw=q.read_bytes();v=json.loads(raw)
  assert v['complete']and v['judge']==a.judge and v['identity_SHA']==identity and v['split']=='eval'
  assert v['signal_id']==t['row']['signal_id']and v['audio_source_SHA']==t['audio_SHA']and v['ear']==ear and v['prompt_SHA']==hashlib.sha256(pr['text'].encode()).hexdigest()
  assert v['diagnostic']==U.expectation(v['measurement']);d=v['diagnostic'];j=prompts.index(pr['id']);scores[i,j,ear]=d['oriented_score'];mass[i,j,ear]=d['raw_total_mass'];manifest.append(dict(path=str(q),SHA=hashlib.sha256(raw).hexdigest()))
 assert np.isfinite(scores).all()and np.isfinite(mass).all();out=r/'point_exports'/a.judge;out.mkdir(parents=True,exist_ok=True)
 with(out/'raw_fields.npz').open('xb')as f:np.savez_compressed(f,signal_ids=np.asarray(c['EVAL_signal_ids']),split=np.asarray(['eval']*2189),prompt_ids=np.asarray(prompts),score_ears=scores,raw_prompt_fields=scores.mean(2),raw6mean=scores.mean((1,2)),raw_mass_ears=mass)
 (out/'measurement_manifest.json').write_text(json.dumps(manifest,indent=2));complete=dict(judge=a.judge,rows=2189,measurements=26268,identity_SHA=identity,training_updates=0,files={p.name:hashlib.sha256(p.read_bytes()).hexdigest()for p in out.iterdir()if p.is_file()});(out/'complete.json').write_text(json.dumps(complete,indent=2));print(json.dumps(complete))

if __name__=='__main__':main()
