import re,json
import numpy as np
import torch
from numba import njit

def byte_inverse():
 vals=list(range(ord('!'),ord('~')+1))+list(range(161,173))+list(range(174,256));chars=vals.copy();n=0
 for b in range(256):
  if b not in vals:vals.append(b);chars.append(256+n);n+=1
 return{chr(c):b for b,c in zip(vals,chars)}

def tokenizer_pieces(tokenizer_json):
 data=json.loads(open(tokenizer_json,encoding='utf-8').read());vocab=data['model']['vocab'];inv=byte_inverse();special={int(t['id'])for t in data['added_tokens']if t['special']}|{0}
 assert data['model']['type']=='BPE'and data['decoder']['type']=='ByteLevel'
 out=[None]*len(vocab)
 for text,i in vocab.items():
  out[i]=b''if i in special else bytes(inv[c]for c in text)
 assert all(v is not None for v in out)
 return out

def fold_piece(piece):
 return bytes(c+32 if 65<=c<=90 else c for c in piece)

def wordchar(c):return 97<=c<=122 or 48<=c<=57 or c==39 or c>=128

def piece_description(piece):
 piece=fold_piece(piece);runs=[];cur=bytearray();hasdelim=False
 for c in piece:
  if wordchar(c):cur.append(c)
  else:runs.append(bytes(cur));cur.clear();hasdelim=True
 runs.append(bytes(cur))
 return (piece,False,b'',b'',())if not hasdelim else(piece,True,runs[0],runs[-1],tuple(q for q in runs[1:-1]if q))

def state_after_prefix(word,text,eligible):
 L=len(word)
 if not text:return 1
 if not word.startswith(text):return 0
 if len(text)==L:return L+2 if eligible else L+1
 return len(text)+1

def map_piece(desc,word,eligible):
 raw,hasdelim,first,last,inside=desc;L=len(word);Q=L+4;A=Q-1
 if not raw:return np.arange(Q,dtype=np.int32)
 if not hasdelim:
  out=np.zeros(Q,np.int32);out[A]=A;out[1]=state_after_prefix(word,raw,eligible)
  for k in range(1,L):
   if word[k:].startswith(raw):
    n=k+len(raw);out[k+1]=(L+2 if eligible else L+1)if n==L else n+1
  return out
 out=np.full(Q,state_after_prefix(word,last,eligible),np.int32);out[A]=A
 if eligible and word in inside:out[:]=A;return out
 if not first:out[L+2]=A
 elif eligible:
  if first==word:out[1]=A
  for k in range(1,L):
   if first==word[k:]:out[k+1]=A
 return out

def compile_word(pieces,query,descriptions=None):
 assert re.fullmatch("[a-z0-9']+",query)
 word=query.encode('ascii');Q=len(word)+4;desc=descriptions if descriptions is not None else[piece_description(p)for p in pieces]
 identity=np.arange(Q,dtype=np.int32);maps=[np.stack([identity,identity])];classes=np.empty(len(pieces),np.int32);groups={(tuple(identity),tuple(identity)):0};merged=0;cache={}
 prefixes={word[:k]:k for k in range(len(word)+1)};suffixes={word[k:]:k for k in range(len(word))}
 for i,p in enumerate(desc):
  raw,hasdelim,first,last,inside=p
  if not raw:key=('empty',)
  elif not hasdelim:key=('plain',raw if raw in word else None)
  else:key=('delim',not bool(first),suffixes.get(first,-1),word in inside,prefixes.get(last,-1))
  if key not in cache:
   g=np.stack([map_piece(p,word,False),map_piece(p,word,True)])
   merge=all(np.array_equal(g[new][g[old]],g[old])for old in range(2)for new in range(2))
   if merge:
    gkey=(tuple(g[0]),tuple(g[1]))
    if gkey not in groups:groups[gkey]=len(maps);maps.append(g)
    cache[key]=(g,groups[gkey])
   else:cache[key]=(g,None)
  g,group=cache[key]
  if group is None:group=len(maps);maps.append(g)
  else:merged+=1
  classes[i]=group
 assert classes[0]==0
 trans=np.stack(maps,axis=2).astype(np.int32);final=np.zeros(Q,np.float64);final[len(word)+2:]=1.
 return dict(query=query,classes=classes,trans=trans,final=final,start=1,merged_tokens=merged,C=len(maps),Q=Q)


@njit(cache=False)
def compress_prob(full,classes,C):
 out=np.zeros((len(full),C),np.float64)
 for t in range(len(full)):
  for v in range(full.shape[1]):out[t,classes[v]]+=full[t,v]
 return out

@njit(cache=False)
def forward_event(p,trans,window,final,start):
 T,C=p.shape;Q=len(final);a=np.zeros((Q,C),np.float64);a[start,0]=1.
 for t in range(T):
  total=a.sum(axis=1);n=a.copy();n[:,0]=total;g=trans[int(window[t])]
  for c in range(1,C):
   for k in range(Q):n[g[k,c],c]+=total[k]-a[k,c]
  a=n*p[t]
 return(a.sum(axis=1)*final).sum()

@njit(cache=False)
def event_with_grad(p,trans,window,final,start):
 T,C=p.shape;Q=len(final);alpha=np.zeros((T+1,Q,C),np.float64);alpha[0,start,0]=1.
 for t in range(T):
  a=alpha[t];total=a.sum(axis=1);n=a.copy();n[:,0]=total;g=trans[int(window[t])]
  for c in range(1,C):
   for k in range(Q):n[g[k,c],c]+=total[k]-a[k,c]
  alpha[t+1]=n*p[t]
 value=(alpha[-1].sum(axis=1)*final).sum();beta=np.zeros((Q,C),np.float64)+final.reshape(Q,1);grad=np.zeros_like(p)
 for t in range(T-1,-1,-1):
  a=alpha[t];total=a.sum(axis=1);g=trans[int(window[t])];baseline=np.zeros(Q,np.float64)
  for k in range(Q):
   baseline[k]=p[t,0]*beta[k,0]
   grad[t,0]+=total[k]*beta[k,0]
   for c in range(1,C):
    b=beta[g[k,c],c];baseline[k]+=p[t,c]*b;grad[t,c]+=a[k,c]*beta[k,c]+(total[k]-a[k,c])*b
  prev=np.zeros((Q,C),np.float64)+baseline.reshape(Q,1)
  for k in range(Q):
   for c in range(1,C):prev[k,c]+=p[t,c]*(beta[k,c]-beta[g[k,c],c])
  beta=prev
 return value,grad

class _WordEvent(torch.autograd.Function):
 @staticmethod
 def forward(ctx,prob,model,window):
  full=prob.detach().cpu().numpy();p=compress_prob(full,model['classes'],model['C']);value,grad=event_with_grad(p,model['trans'],window,model['final'],model['start'])
  assert np.isfinite(value)and -1e-10<=value<=1+1e-10
  ctx.save_for_backward(torch.as_tensor(grad[:,model['classes']],dtype=prob.dtype,device=prob.device))
  return prob.new_tensor(value)
 @staticmethod
 def backward(ctx,grad_output):
  grad,=ctx.saved_tensors
  return grad*grad_output,None,None

def word_event_probability(full_prob,model,window=None):
 if window is None:window=np.ones(len(full_prob),np.bool_)
 return _WordEvent.apply(full_prob.double(),model,np.asarray(window,np.bool_))

def brute_word_hit(path,pieces,query,window):
 text=bytearray();times=[];last=-1
 for t,v in enumerate(path):
  if v!=0 and v!=last:
   q=fold_piece(pieces[int(v)]);text.extend(q);times.extend([t]*len(q))
  last=v
 word=bytearray();wt=[]
 for c,t in zip(bytes(text)+b' ',times+[len(path)]):
  if wordchar(c):word.append(c);wt.append(t)
  else:
   if bytes(word)==query.encode()and window[wt[-1]]:return True
   word.clear();wt.clear()
 return False
