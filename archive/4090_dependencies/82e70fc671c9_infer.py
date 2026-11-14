"""Frozen cached-feature inference only; no audio extractor, labels, or training."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

SIDE = ('confidence_v3','confidence_v2','wer','extra_wer','candidate','paired')
DIMS = (12,12,1,4,1,24)
FORBIDDEN = {'target','targets','correctness','response','original_response','hits','phoneme_hits','phoneme_correctness','labels'}


def require(ok,message):
    if not ok: raise ValueError(message)


def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''): h.update(block)
    return h.hexdigest()


def digest(v):
    return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def normalized_side(raw,norm):
    require(set(norm['side']) <= set(SIDE),'Unknown saved normalizer key')
    n=len(raw['pooled']); parts=[]
    for key,width in zip(SIDE,DIMS):
        x=np.asarray(raw[key],np.float32)
        require(x.shape==(n,width) and np.isfinite(x).all(),'Invalid side '+key)
        if key in norm['side']:
            spec=norm['side'][key]; mean,std=np.asarray(spec['mean'],np.float32),np.asarray(spec['std'],np.float32)
            require(mean.shape==std.shape==(width,) and np.isfinite(mean).all() and np.isfinite(std).all()
                    and (std>0).all(),'Invalid saved normalization')
            x=(x-mean)/std
        parts.append(x)
    return np.concatenate(parts,-1)


def validate_input(raw,feature_contract):
    required={'signals','pooled','word_evidence','evidence_mask','word_mask','feature_identity_sha256',*SIDE}
    require(set(raw)==required and not FORBIDDEN.intersection(raw),'Unexpected/missing inference columns; labels prohibited')
    require(np.asarray(raw['feature_identity_sha256']).shape==()
            and str(raw['feature_identity_sha256'].item())==digest(feature_contract),'Feature recipe identity differs')
    ids=np.asarray(raw['signals']); require(ids.ndim==1 and len(ids)>0 and ids.dtype.kind in 'US','Invalid signal IDs')
    require(len(set(ids.tolist()))==len(ids) and all(ids.tolist()),'Empty or duplicate signal ID')
    n=len(ids); pooled=np.asarray(raw['pooled']); values=np.asarray(raw['word_evidence'])
    evidence,words=np.asarray(raw['evidence_mask']),np.asarray(raw['word_mask'])
    require(pooled.shape==(n,1280) and np.isfinite(pooled).all(),'Invalid pooled encoder features')
    require(values.ndim==3 and values.shape[0]==n and values.shape[-1]==261,'Invalid word evidence shape')
    require(evidence.dtype==words.dtype==np.bool_ and evidence.shape==words.shape==values.shape[:2], 'Invalid masks')
    require(words.any(1).all() and not (evidence & ~words).any(),'Empty sentence or evidence outside words')
    # Word positions must be a prefix, matching the cached sentence denominator.
    require(not ((~words[:,:-1]) & words[:,1:]).any(),'Non-prefix word mask')
    require(np.isfinite(values[evidence]).all(),'Nonfinite supported word evidence')
    # Unsupported/padding evidence may be NaN; mask it before the exported model.
    return ids,pooled.astype(np.float32),np.where(evidence[...,None],values,0).astype(np.float32),evidence,words


class Bundle:
    def __init__(self,root):
        self.root=Path(root).resolve(); manifest=read(self.root/'bundle_manifest.json')
        require(manifest['schema']=='pooled_cached_candidate_v1','Wrong bundle kind')
        for name,h in manifest['files'].items():
            path=self.root/name
            require(Path(name).name==name and path.resolve().parent==self.root and sha(path)==h,'Bundle file changed: '+name)
        self.recipe=read(self.root/'recipe.json'); self.norm=read(self.root/'normalization.json')
        require(self.norm['fit_ids_sha256']==self.recipe['id_hashes']['train_ids']
                and self.norm['fit_n']==16204,'Frozen TRAIN normalization scope changed')
        self.feature_contract=read(self.root/'feature_contract.json')
        spec=importlib.util.spec_from_file_location('_pooled_bundle_models',self.root/'model_defs.py')
        mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); self.mod=mod
        self.models={}
        require(self.recipe['default_seed']==42 and self.recipe['available_seeds']==[42,43,44,45]
                and self.recipe['prediction_averaging_used'] is False,'Seed policy changed')
        for seed in self.recipe['available_seeds']:
            ck=torch.load(self.root/f'seed{seed}.pt',map_location='cpu',weights_only=True)
            require(ck['seed']==seed and ck['recipe_sha256']==digest(self.recipe),'Weight/recipe identity mismatch')
            baseline=mod.ReferenceModel(1280,54); baseline.load_state_dict(ck['baseline'])
            residual=mod.WordCorrection(); residual.load_state_dict(ck['residual'])
            self.models[seed]=(baseline.eval().requires_grad_(False),residual.eval().requires_grad_(False))

    @torch.no_grad()
    def predict(self,raw,seed=42,batch=64):
        require(seed in self.models and isinstance(batch,int) and batch>0,'Invalid seed or batch size')
        ids,pooled,values,evidence,words=validate_input(raw,self.feature_contract)
        side=normalized_side(raw,self.norm); require(np.isfinite(side).all(),'Nonfinite normalized side')
        baseline,residual=self.models[seed]; parts=[]; bases=[]
        for start in range(0,len(ids),batch):
            sl=slice(start,start+batch); x=torch.from_numpy(pooled[sl]); s=torch.from_numpy(side[sl])
            # Direct logits avoid lossy probability-to-logit conversion near 0/1.
            logits=baseline.head[:-1](torch.cat((baseline.projection(x),s),-1)).squeeze(-1)
            v,k,w=map(torch.from_numpy,(values[sl],evidence[sl],words[sl]))
            v,k,w=self.mod.sentence_pool(v,k,w)
            pred,_,_=residual(v,k,w,logits)
            require(bool(torch.isfinite(pred).all()) and bool(((pred>=0)&(pred<=1)).all()),'Invalid frozen predictions')
            parts.append(pred.numpy()); bases.append(logits.sigmoid().numpy())
        return ids,np.concatenate(parts),np.concatenate(bases)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bundle',type=Path,default=Path(__file__).resolve().parent)
    p.add_argument('--features',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44,45],default=42); args=p.parse_args()
    require(args.out.resolve()!=args.features.resolve() and not args.out.exists(),'Preserve input/existing outputs')
    torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
    bundle=Bundle(args.bundle)
    with np.load(args.features,allow_pickle=False) as z: raw={n:z[n].copy() for n in z.files}
    ids,pred,baseline=bundle.predict(raw,args.seed)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    # Exclusive create prevents overwriting a concurrently created result.
    with args.out.open('xb') as f:
        np.savez(f,signals=ids,predictions=pred,baseline=baseline,seed=np.asarray(args.seed),
                 recipe_sha256=np.asarray(digest(bundle.recipe)))
    print('POOLED_CACHED_INFERENCE_COMPLETE '+json.dumps(dict(rows=len(ids),seed=args.seed,out=str(args.out),
           labels_used=False,raw_audio_extraction_supported=False,submission=False)),flush=True)


if __name__=='__main__': main()
