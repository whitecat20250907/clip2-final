"""Original LMY native48/local81 and dual-phone35 inference on EVAL caches."""
import argparse,ast,hashlib,json,math,re,time
from pathlib import Path
from functools import lru_cache

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--base');ap.add_argument('--role');a=ap.parse_args();root=Path(a.base);cfg=json.loads((root/'protocol.json').read_text());out=root/'tasks'/a.role;out.mkdir(parents=True,exist_ok=True);start=time.monotonic()
    def save(name,v):
        p=out/(name+'.next');p.write_text(json.dumps(v,indent=2,allow_nan=False));p.replace(out/name)
    def progress(stage,**kw):save('progress.json',dict(stage=stage,seconds=time.monotonic()-start,**kw))
    import numpy as np,torch
    from torch import nn
    from torch.nn import functional as F
    from scipy.special import logsumexp
    from safetensors import safe_open
    from tokenizers import Tokenizer
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.cuda.set_per_process_memory_fraction(1.5*1024**3/torch.cuda.get_device_properties(0).total_memory)
    env=dict(np=np,torch=torch,nn=nn,F=F,logsumexp=logsumexp,re=re,math=math,lru_cache=lru_cache)
    for item in cfg['definitions']:
        p=Path(item['path']);raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==item['SHA'];nodes=[n for n in ast.parse(raw.decode()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))and n.name in item['names']];assert {n.name for n in nodes}==set(item['names']);exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),env)
    records=json.loads((root/'selection.json').read_text());assert len(records)==2191 and [r['role']for r in records[:2]]==['VALID_CONTROL']*2
    ss=[r['signal']for r in records];assert len(set(ss))==2191
    dictionary={};phone_dictionary={};phones={p:i+1 for i,p in enumerate('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG OW OY P R S SH T TH UH UW V W Y Z ZH'.split())}
    for line in Path(cfg['cmu']).read_text().splitlines():
        fields=line.split()
        if len(fields)<2 or fields[0].startswith(';;;'):continue
        word=re.sub(r'\(\d+\)$','',fields[0]).lower();pron=tuple(re.sub(r'\d','',v)for v in fields[1:])
        if pron and pron not in dictionary.get(word,[]):dictionary.setdefault(word,[]).append(pron)
        if word not in phone_dictionary and all(p in phones for p in pron):phone_dictionary[word]=[phones[p]for p in pron]
    env['dictionary']=dictionary;contractions=cfg['contractions'];public={};publicphone={}
    for row in records:
        words=row['prompt'].split();assert len(words)==row['n_words']
        public[row['signal']]=dict(words=words,prompt=row['prompt'],original_prompt=row['original_prompt'],severity=row['severity'])
        publicphone[row['signal']]=dict(n_words=len(words),severity=row['severity'],reference=None if any(w not in phone_dictionary for w in words)else[v for w in words for v in phone_dictionary[w]])
    banks={}
    for family,roots in cfg['caches'].items():
        for role,path in roots.items():
            p=Path(path);ix=json.loads((p/'index.json').read_text());assert ix['dim']==1024
            banks[family,role]=(ix,np.memmap(p/'features.f16',mode='r',dtype=np.float16,shape=(ix['total_tokens'],1024)),{s:i for i,s in enumerate(ix['signal_ids'])})
    def frame(s,family):
        ix,mm,lookup=banks[family,'VALID'if s in ss[:2]else'EVAL'];j=lookup[s];o=ix['offsets'][j];return np.asarray(mm[o:o+ix['lengths'][j]],dtype=np.float32).copy()
    with np.load(cfg['expected_native'],allow_pickle=False)as z:
        ii=[z['signals'].tolist().index(s)for s in ss[:2]];expected={k:z[k][ii].copy()for k in ['native48','local81','legal','texts']}
    tok=Tokenizer.from_file(str(Path(cfg['granite_model'])/'tokenizer.json'))
    with safe_open(str(Path(cfg['granite_model'])/'model.safetensors'),framework='pt',device='cpu')as f:W=f.get_tensor('encoder.out.weight').float().cuda();bias=f.get_tensor('encoder.out.bias').float().cuda()
    hs=nn.ModuleDict({n:env['Head'](seed)for n,seed in [('ref',80044),('response',81044)]}).cuda().eval().requires_grad_(False)
    cp=torch.load(cfg['adapt_checkpoint'],map_location='cpu',weights_only=False);assert cp['steps']==768;hs.load_state_dict(cp['weights']);del cp
    native=np.empty((len(ss),48));hard=np.zeros((len(ss),12,30));phone=np.zeros((len(ss),12,18));soft=np.zeros((len(ss),12,33));legal=np.zeros((len(ss),12),bool);texts=[];proof=[];mass=0.
    with torch.inference_mode():
        for i,s in enumerate(ss):
            if i==2:save('VALID_native_control_passed.json',dict(cases=proof))
            x=torch.from_numpy(frame(s,'Granite')).cuda();base=torch.cat([F.linear(v,W,bias)for v in torch.split(x,256)],0);tt=[];ww=public[s]['words'];nw=len(ww);legal[i,:nw]=True;tids,mapping=env['token_word_map'](ww,tok)
            for h,name in enumerate(['frozen','ref','response']):
                logits=base if h==0 else base+(x@hs[name].A.T)@hs[name].B.T
                if h==0:
                    lp=logits.double().log_softmax(-1).cpu().numpy();feat,txt,_=env['extract_feature'](s,lp,lp.argmax(1),tok,public);native[i]=feat;tt.append(txt)
                else:
                    raw=logits.argmax(1).cpu().tolist();tt.append(tok.decode([v for j,v in enumerate(raw)if v!=0 and(j==0 or v!=raw[j-1])],skip_special_tokens=True))
                lz=torch.logsumexp(logits.double(),1);tlp=(logits[:,tids].double()-lz[:,None]).cpu().numpy();blp=(logits[:,0].double()-lz).cpu().numpy();mp=torch.exp(logits.max(1).values.double()-lz).cpu().numpy()
                ff,logZ,gap=env['soft_word_stats'](tlp,blp,mp,tids,mapping,nw);soft[i,:nw,h*11:(h+1)*11]=ff;mass=max(mass,gap)
                if i<2 and logZ is not None:
                    dense=logits.double().log_softmax(-1).cpu();value=float(F.ctc_loss(dense[:,None,:],torch.tensor(tids),torch.tensor([len(x)]),torch.tensor([len(tids)]),blank=0,reduction='sum'));assert abs(value+logZ)<1e-8
            _,hard[i,:nw]=env['local_features'](ww,tt,public[s]['severity'],contractions)
            for h,text in enumerate(tt):
                _,extra,_=env['phonetic_alignment'](ww,env['expanded_words'](text,contractions));phone[i,:nw,h*6:(h+1)*6]=extra
            texts.append(tt)
            if i<2:
                local=np.concatenate([hard[i],phone[i],soft[i]],-1);assert np.array_equal(np.isfinite(native[i]),np.isfinite(expected['native48'][i]))
                ng=float(np.nanmax(abs(native[i]-expected['native48'][i])));lg=float(np.max(abs(local-expected['local81'][i])));assert ng<1e-3 and lg<1e-3
                assert np.array_equal(local[:,:48],expected['local81'][i,:,:48])and tt==expected['texts'][i].tolist()and np.array_equal(legal[i],expected['legal'][i]);proof.append(dict(signal=s,native48_maxgap=ng,local81_maxgap=lg,hard_phone48_equal=True))
            if i%16==0:progress('native48_local81',done=max(0,i-1),total=2189)
    local=np.concatenate([hard,phone,soft],2);assert np.isfinite(local).all()
    np.savez_compressed(out/'EVAL_NATIVE48_WORD81.npz',signals=np.asarray(ss[2:]),native48=native[2:],local81=local[2:],legal=legal[2:],texts=np.asarray(texts[2:]))
    del hs,W,bias,base,logits,x;torch.cuda.empty_cache()
    for family in ['Parakeet','Granite']:
        for seed in [44,45]:
            spec=cfg['basis'][family][str(seed)];heads=env['make_heads']().eval().requires_grad_(False);cp=torch.load(spec['checkpoint'],map_location='cpu',weights_only=False);heads.load_state_dict(cp['heads']);del cp;features=np.empty((len(ss),35))
            with np.load(spec['expected'],allow_pickle=False)as z:
                ii=[z['signals'].tolist().index(s)for s in ss[:2]];target=z['features'][ii].copy()
            with torch.inference_mode():
                for pos in range(0,len(ss),32):
                    group=ss[pos:pos+32];lps=env['batch_lp'](heads,group,{s:frame(s,family)for s in group})
                    for j,(s,lp)in enumerate(zip(group,lps)):features[pos+j]=env['dual35'](lp[:,0].numpy(),lp[:,1].numpy(),publicphone[s])
                    if pos==0:
                        assert np.array_equal(np.isfinite(features[:2]),np.isfinite(target));gap=float(np.nanmax(abs(features[:2]-target)));assert gap<1e-3;save(f'VALID_{family}_{seed}_control_passed.json',dict(maxgap=gap))
                    if pos%128==0:progress('dual_phone35',family=family,seed=seed,done=min(pos+32,len(ss))-2,total=2189)
            np.savez_compressed(out/f'EVAL_{family.upper()}35_seed{seed}.npz',signals=np.asarray(ss[2:]),features=features[2:]);del heads;torch.cuda.empty_cache()
    def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
    save('complete.json',dict(rows=2189,training_updates=0,max_posterior_mass_gap=mass,seconds=time.monotonic()-start,outputs={p.name:sha(p)for p in out.glob('EVAL_*.npz')}))

if __name__=='__main__':main()
