from prefix_common_v1 import *
from score_math_v1 import *
import gc,time
E=ROOT.parent.parent
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def plan():return read(ROOT/'PREREGISTRATION_V1.json')
def validate():
 p=plan();b=read(ROOT/'SOURCE_BINDING_V1.json')
 for n,h in b['files'].items():assert sha(ROOT/n)==h,n
 for f,h in b['upstream'].items():assert sha(f)==h,f
 return p
def input_data():
 p=validate();b=read(ROOT/'PREPARED_INPUTS_BINDING_V1.json')
 for n,h in b['files'].items():assert sha(ROOT/n)==h,n
 with np.load(p['roles'],allow_pickle=False) as z:fit=z['FIT_ids'].tolist();held=z['historical_held_ids'].tolist()
 with np.load(ROOT/'FIT_SENTENCE_TARGETS_V1.npz',allow_pickle=False) as z:
  assert z['signals'].tolist()==fit;y=z['y'].copy()
 return p,fit,held,read(ROOT/'PUBLIC_RECORDS_V1.json'),read(ROOT/'AUXILIARY_FIT_PHONE_TARGETS_V1.json'),y
def cache_data(fit,held):
 p=plan();C=pathlib.Path(p['cache_root'])
 assert sha(C/'SOURCE_BINDING_V1.json')==p['cache_binding_sha'] and sha(C/'PREFIX14_QUALIFICATION_V1.json')==p['cache_qualification_sha']
 assert read(C/'CACHE_PROCESS_TERMINAL_V1.json')['actual_exit_code']==0
 seal=read(C/'PUBLIC_PREFIX_CACHE_COMPLETE_V1.json');assert seal['real_rows']==20239 and seal['all_public_parameters_unchanged'] and seal['public_checkpoint_sha']==p['checkpoint_sha']
 wanted=set(fit+held);D={}
 for r in seal['real_receipts']:
  path=C/r['file'];assert sha(path)==r['sha256']
  if not wanted.intersection(r['signals']):continue
  with np.load(path,allow_pickle=False) as z:
   ss=z['signals'].tolist();off=z['offsets'];co=z['cu_offsets'];hh=z['h14'];cc=z['cu_values'];nf=z['mel_frames']
   for j,sid in enumerate(ss):
    if sid not in wanted:continue
    assert sid not in D
    D[sid]=dict(h=hh[off[j]:off[j+1]].copy(),cu=cc[co[j]:co[j+1]].copy(),mel=int(nf[j]))
 assert set(D)==wanted
 needed={d['mel'] for d in D.values()};N={}
 for r in seal['null_receipts']:
  path=C/r['file'];assert sha(path)==r['sha256']
  if not needed.intersection(map(int,r['signals'])):continue
  with np.load(path,allow_pickle=False) as z:
   ss=z['signals'].tolist();off=z['offsets'];co=z['cu_offsets'];hh=z['h14'];cc=z['cu_values']
   for j,s in enumerate(ss):
    nf=int(s)
    if nf in needed:N[nf]=dict(h=hh[off[j]:off[j+1]].copy(),cu=cc[co[j]:co[j+1]].copy())
 assert set(N)==needed
 return D,N,sha(C/'PUBLIC_PREFIX_CACHE_COMPLETE_V1.json')
def public_hash(tower):
 h=hashlib.sha256()
 for n,p in sorted(tower.named_parameters()):
  if n.endswith('.A') or n.endswith('.B'):continue
  h.update(n.replace('.base.','.').encode()+b'\0');h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def tensor_hash(st):
 h=hashlib.sha256()
 for n,v in sorted(st.items()):h.update(n.encode()+b'\0');h.update(v.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
 return h.hexdigest()
def delta_state(mods):return {n+'.'+k:p.detach().cpu().clone() for n,m in mods.items() for k,p in [('A',m.A),('B',m.B)]}
def set_delta(mods,st):
 with torch.no_grad():
  for n,m in mods.items():
   for k,p in [('A',m.A),('B',m.B)]:p.copy_(st[n+'.'+k].to(p.device))
def initialize(mods,seed,arm):
 assert arm in plan()['arms']
 torch.manual_seed(60000+seed)
 with torch.no_grad():
  for m in mods.values():m.A.normal_(std=.02);m.B.zero_()
 for m in mods.values():m.A.requires_grad_(True);m.B.requires_grad_(True)
 hs=nn.ModuleDict()
 for name,offset in [('ref',80000),('response',81000)]:
  torch.manual_seed(offset+seed)
  h=nn.Sequential(nn.LayerNorm(1024),nn.Linear(1024,64),nn.GELU(),nn.Linear(64,40))
  with torch.no_grad():h[-1].bias.zero_();h[-1].bias[0]=np.log(39)
  hs[name]=h
 return hs
def make_tail():
 tower,_,_,_=load_tower();original=public_hash(tower);mods=insert_lora(tower);tower.config._attn_implementation='sdpa'
 for m in list(tower.layers[14:])+[tower.ln_post,tower.proj1,tower.proj2]:m.to('cuda:0')
 assert public_hash(tower)==original
 return tower,mods,original
def frame_forward(tower,item):
 h=torch.from_numpy(item['h']).to('cuda:0');cu=torch.from_numpy(item['cu']).to('cuda:0')
 with mathctx():fr=tail(tower,h,cu)
 return fr

def lp_forward(tower,heads,item,detach_second=False):
 fr=frame_forward(tower,item)
 return ctc_logp(heads['ref'],fr),ctc_logp(heads['response'],fr.detach() if detach_second else fr)

def features(lp,null_lp,record):
 assert lp.ndim==2 and lp.shape[1]==40 and lp.shape==null_lp.shape and np.isfinite(lp).all() and np.isfinite(null_lp).all()
 prob=np.exp(lp);decoded=greedy(lp);ref=record['reference'];n=record['n_words'];sev=record['severity']
 out=np.full(18,np.nan,np.float64)
 out[7]=float(-(prob*lp).sum(1).mean()/np.log(40));out[8]=float(prob[:,0].mean());out[9]=float(prob.max(1).mean())
 out[11]=np.log1p(len(lp)//2);out[13]=n;out[14]=1.;out[15]=sev==1;out[16]=sev==2;out[17]=1/n
 if ref:
  nr=len(ref);_,s,d,i=edits(ref,decoded);out[3]=np.clip(1-(s+d+i)/nr,0,1);out[4:7]=[s/nr,d/nr,i/nr]
  out[10]=np.log1p(len(decoded))-np.log1p(nr);out[12]=np.log1p(nr)
  if minimum(ref)<=len(lp):
   a=float(ctc(torch.from_numpy(lp),ref))/nr;b=float(ctc(torch.from_numpy(null_lp),ref))/nr
   out[:3]=[a,b,b-a];out[14]=0.
 assert np.isfinite(out[[7,8,9,11,13,14,15,16,17]]).all()
 return out
def disagreement(lpa,lpb):
 assert lpa.shape==lpb.shape and lpa.ndim==2 and lpa.shape[1]==40
 pa,pb=np.exp(lpa),np.exp(lpb);m=np.logaddexp(lpa,lpb)-np.log(2.)
 js=float(.5*((pa*(lpa-m)).sum(1)+(pb*(lpb-m)).sum(1)).mean()/np.log(2.))
 blank=float(abs(pa[:,0]-pb[:,0]).mean())
 ga,gb=greedy(lpa),greedy(lpb);_,s,d,i=edits(ga,gb)
 return np.array([js,blank,(s+d+i)/max(1,len(ga),len(gb))],np.float64)

def dual_features(lp,nlp,record):
 return np.r_[features(lp[:,0,:],nlp[:,0,:],record),features(lp[:,1,:],nlp[:,1,:],record),disagreement(lp[:,0,:],lp[:,1,:])]

def features_all(tower,heads,D,N,public,fit,held,r):
 null_bank={}
 with torch.no_grad():
  for nf,item in sorted(N.items()):null_bank[nf]=np.stack([v.numpy().copy() for v in lp_forward(tower,heads,item)],axis=1)
  fit_x=[];held_x=[];held_lp=[];off=[0]
  for role,ids,dest in [('FIT',fit,fit_x),('held_public',held,held_x)]:
   for j,sid in enumerate(ids):
    lp=np.stack([v.numpy().copy() for v in lp_forward(tower,heads,D[sid])],axis=1);dest.append(dual_features(lp,null_bank[D[sid]['mel']],public[sid]))
    if role=='held_public':held_lp.append(lp);off.append(off[-1]+len(lp))
    if (j+1)%1024==0:print('DUAL_PUBLIC_FEATURES',role,j+1,len(ids),flush=True)
 null_keys=sorted(null_bank);noff=np.r_[0,np.cumsum([len(null_bank[k]) for k in null_keys])]
 save(r/'HELD_LOGPROBS_AND_NULL_V1.npz',signals=np.asarray(held),real_offsets=np.asarray(off,np.int64),real_lp=np.concatenate(held_lp),
 held_mel=np.asarray([D[s]['mel'] for s in held],np.int64),null_mel=np.asarray(null_keys,np.int64),null_offsets=noff.astype(np.int64),null_lp=np.concatenate([null_bank[k] for k in null_keys]))
 return np.asarray(fit_x,np.float64),np.asarray(held_x,np.float64)
