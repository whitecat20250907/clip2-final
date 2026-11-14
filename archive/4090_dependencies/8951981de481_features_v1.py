"""Public reference/machine-transcript features; no human-response reader."""
import re,unicodedata
import numpy as np
from alignment_reference_v1 import align_words
def normalize_machine(text):
 if not isinstance(text,str):raise TypeError('Machine transcript must be str')
 text=unicodedata.normalize('NFKC',text).casefold().translate(str.maketrans({'’':"'",'‘':"'",'“':'"','”':'"'}))
 text=re.sub(r'[-–—/]',' ',text);text=re.sub(r"[^\w\s']",'',text)
 return ' '.join(text.split())
def char_distance(a,b):
 row=list(range(len(b)+1))
 for i,aa in enumerate(a,1):
  nextrow=[i]
  for j,bb in enumerate(b,1):nextrow.append(min(row[j]+1,nextrow[-1]+1,row[j-1]+(aa!=bb)))
  row=nextrow
 return row[-1]
def raw_features(reference,machine):
 ref=reference.casefold().split();hyp=normalize_machine(machine)
 if not ref:raise ValueError('Empty reference')
 alignment=align_words(reference,hyp);hw=hyp.split();n=len(ref)
 out=np.zeros((n,9),np.float32)
 for i,(hit,hi) in enumerate(zip(alignment['labels'],alignment['hyp_indices'])):
  deletion=hi is None;substitution=(not hit) and not deletion
  sim=0. if deletion else 1.-char_distance(ref[i],hw[hi])/max(len(ref[i]),len(hw[hi]),1)
  out[i,:6]=[hit,substitution,deletion,not alignment['label_mask'][i],not alignment['mapping_mask'][i],sim]
 out[:,6]=alignment['hits']/n;out[:,7]=min(3.,len(hw)/n)/3.;out[:,8]=min(2.,alignment['distance']/n)/2.
 assert np.isfinite(out).all() and (out>=0).all() and (out<=1).all() and np.all(out[:,:3].sum(1)==1)
 return out
def controlled_features(arm,features,words):
 if arm=='wrong54_h16':
  result=features.copy()
  for i,n in enumerate(words.sum(1)):
   n=int(n);result[i,:n,:6]=features[i,(np.arange(n)+1)%n,:6]
  return result
 if arm=='global54_h16':
  result=features.copy();result[...,:6]=0.;return result
 return features
