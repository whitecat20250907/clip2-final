"""Frozen raw/RMS Qwen candidate inference, using reviewed original normalize/batch functions."""
import json,hashlib,importlib.util,sys
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;A=ROOT.parent;OLD=A/'d6e_qwen_encoder_direct_swap_20261015'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def module(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
def require(ok,message):
 if not ok:raise ValueError(message)
def normalize(raw, mean, std, mask=None):
    raw = np.asarray(raw, np.float32)
    mean, std = np.asarray(mean, np.float64), np.asarray(std, np.float64)
    if raw.shape[-1:] != mean.shape or std.shape != mean.shape:
        raise ValueError('Raw evidence coordinate dimension differs')
    if not np.isfinite(raw).all() or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 1e-4):
        raise ValueError('Invalid evidence or frozen normalization')
    out = np.clip((raw - mean) / std, -6, 6).astype(np.float32)
    if mask is not None:
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.shape != raw.shape[:-1]:
            raise ValueError('Normalization support mask differs')
        out[~mask] = 0
    return out

def validate_batch(logits, evidence, listener, support, words, severity):
    if logits.ndim != 2 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError('Expected nonempty batch of word logits')
    shape = logits.shape
    if evidence.shape != shape + (261,) or listener.shape != shape + (3,):
        raise ValueError('Word evidence shapes differ')
    if support.shape != shape or words.shape != shape or support.dtype != bool or words.dtype != bool:
        raise ValueError('Boolean word/support masks required')
    lengths = words.sum(1)
    if np.any(lengths == 0) or not np.array_equal(words, np.arange(shape[1])[None] < lengths[:, None]):
        raise ValueError('Every scoring word must remain in a contiguous prefix')
    if np.any(support & ~words):
        raise ValueError('Support outside scoring words')
    if severity.shape != (shape[0],) or severity.dtype.kind not in 'iu' or np.any((severity < 0) | (severity > 2)):
        raise ValueError('Public severity must be integer0/1/2')
    if not all(np.isfinite(x).all() for x in (logits, evidence, listener)):
        raise ValueError('Nonfinite public word input')

def valid_batch(ids, index, pooled, side, evidence, audio, text, word_text, masks, rows, device):
    ix = [index[s] for s in ids]
    lengths = [len(evidence[s][0]) for s in ids]
    width = max(lengths); n = len(ids)
    v = np.zeros((n, width, 261), np.float32); k = np.zeros((n, width), bool)
    w = np.arange(width)[None, :] < np.asarray(lengths)[:, None]
    frame_n = [len(audio(s)) for s in ids]
    frames = np.zeros((n, max(frame_n), 2048), np.float32)
    am = np.arange(frames.shape[1])[None, :] < np.asarray(frame_n)[:, None]
    tt = [text[rows[s]['prompt']] for s in ids]
    tokens = np.zeros((n, max(map(len, tt)), 768), np.float32)
    tm = np.zeros(tokens.shape[:2], bool)
    words = np.zeros((n, width, 768), np.float32); mapped = np.zeros((n, width), bool)
    severity = []
    for j, s in enumerate(ids):
        size = lengths[j]; p = rows[s]['prompt']
        require(word_text[p].shape == (size, 768), 'Word-text order/count changed')
        v[j, :size], k[j, :size] = evidence[s]
        a = audio(s); require(a.shape == (frame_n[j], 2048) and np.isfinite(a).all(), 'Bad valid frames')
        frames[j, :len(a)] = a; tokens[j, :len(tt[j])] = tt[j]; tm[j, :len(tt[j])] = True
        words[j, :size] = word_text[p]; mapped[j, :size] = masks[p]
        severity.append(('No Loss', 'Mild', 'Moderate').index(rows[s]['hearing_loss']))
    return tuple(torch.as_tensor(x, device=device) for x in
                 (pooled[ix], side[ix], v, k, w, frames, am, tokens, tm, words, mapped,
                  np.asarray(severity, np.int64)))
class FrozenQwenCandidate:
 def __init__(self,arm,device='cpu'):
  assert arm in ['raw_qwen','rms_qwen']
  self.arm=arm;self.device=torch.device(device);self.protocol=read(ROOT/'full_readout_v1/protocol.json')
  deps=self.protocol['dependency_sha256']
  paths=[A/'d6e_boundary_listener_normalized_context_v1/tf_readout.py',A/'d6e_expert_bottleneck_diagnosis_v1/ctc_competition_readout.py',A/'d6e_shared24_smallNN_beta_valid_export_side_v1/fine_residual.py']
  for p in paths:assert sha(p)==deps[str(p)]
  Head=module('_valid_'+arm+'_head',paths[0]).TFReadout;CTC=module('_valid_'+arm+'_ctc',paths[1]).CTCReadout;Fine=module('_valid_'+arm+'_fine',paths[2]).FineResidual
  self.model=Fine(CTC(Head(),104440),124443,'bounded_pooled')
  self.checkpoint=ROOT/'full_readout_v1'/arm/'beta'/(arm+'_beta_final.pt')
  done=read(ROOT/'full_readout_v1'/arm/'completion.json');assert sha(self.checkpoint)==done['beta_checkpoint_sha256']
  c=torch.load(self.checkpoint,map_location='cpu',weights_only=True)
  assert c['steps']==6084 and c['protocol_sha256']==hashlib.sha256(json.dumps(self.protocol,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
  self.model.load_state_dict(c['weights'],strict=True);self.model.to(self.device).eval().requires_grad_(False)
  cr=read(OLD/'qwen_parent_word_cache_sharded_v1/report.json');assert cr['cache_sha256']=='79e27772183e2beb69439f2abbff824e58a4e0ff91ea20d6b28bb53c56417d96'
  self.context_norm={'mean':cr['TRAIN_evidence_mean'],'std':cr['TRAIN_evidence_std']}
  self.listener_norm=read(ROOT/'full_readout_v1/full_TRAIN_listener_normalization.json')
  self.extra_norm=read(ROOT/'full_readout_v1'/arm/'TRAIN_extra_normalization.json')
 @torch.inference_mode()
 def score(self,parent,x,raw_listener):
  xx=list(x)
  assert xx[5].ndim==3 and xx[5].shape[-1]==2048
  if self.arm=='rms_qwen':xx[5]=xx[5]*torch.rsqrt(xx[5].square().mean(-1,keepdim=True)+1e-6)*xx[6][...,None]
  _,logits=parent(*xx,mode='boundary_joint')
  raw=[logits.cpu().numpy(),xx[2].cpu().numpy(),np.asarray(raw_listener,np.float32),xx[3].cpu().numpy(),xx[4].cpu().numpy(),xx[-1].cpu().numpy()]
  validate_batch(*raw)
  l,e,listener,support,words,severity=raw
  ev=normalize(e,self.context_norm['mean'],self.context_norm['std'],support)
  li=normalize(listener,self.listener_norm['mean'],self.listener_norm['std'])
  extra=np.clip((ev[...,:24]-np.array(self.extra_norm['mean']))/np.array(self.extra_norm['std']),-8,8).astype(np.float32)
  self.last_inputs=tuple(torch.as_tensor(v,device=self.device) for v in (l.astype(np.float32),ev,li,support,words,severity.astype(np.int64),extra))
  q,z=self.model(*self.last_inputs);wp=z.sigmoid()
  assert float(abs((wp*self.last_inputs[4]).sum(1)/self.last_inputs[4].sum(1)-q).max())<2e-6
  self.last_word_probabilities=wp.cpu().numpy()
  return q.cpu().numpy()
