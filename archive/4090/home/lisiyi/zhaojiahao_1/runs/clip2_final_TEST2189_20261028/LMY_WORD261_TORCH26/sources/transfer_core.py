"""Ordered-path posterior -> paired phoneme evidence -> small word correction.

No human response or score label is an inference input. This module does not
establish localization accuracy or intelligibility efficacy.
"""
import re
import unicodedata
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

FEATURE_DIM = 261


def normalized_words(text):
    text = unicodedata.normalize('NFKC', text).lower().replace('’', "'").replace('‘', "'")
    if re.search(r'[^a-z\s\.,!?;:\-—–\'"()…]', text):
        raise ValueError('Unspecified sung-lyric normalization')
    return re.findall(r"[a-z]+(?:'[a-z]+)*'?", text)


def mandatory_equal_mapping(source, scoring):
    """Only word pairs present in every optimal LCS alignment are accepted.

    Repeated-word ambiguity and optional matches are masked, not resolved by
    an arbitrary tie break. Returned indices belong to the *scoring* words.
    """
    n, m = len(source), len(scoring)
    f = np.zeros((n+1, m+1), np.int32)
    b = np.zeros_like(f)
    for i in range(n):
        for j in range(m):
            f[i+1, j+1] = max(f[i, j+1], f[i+1, j], f[i,j]+int(source[i] == scoring[j]))
    for i in range(n-1, -1, -1):
        for j in range(m-1, -1, -1):
            b[i,j] = max(b[i+1,j], b[i,j+1], b[i+1,j+1]+int(source[i] == scoring[j]))
    optimum = int(f[n,m]); result = [-1]*n
    for i in range(n):
        js = [j for j in range(m) if source[i] == scoring[j] and f[i,j]+1+b[i+1,j+1] == optimum]
        optional_i = any(f[i,j]+b[i+1,j] == optimum for j in range(m+1))
        if len(js) != 1 or optional_i:
            continue
        j = js[0]
        other_is = [k for k in range(n) if source[k] == scoring[j] and f[k,j]+1+b[k+1,j+1] == optimum]
        optional_j = any(f[k,j]+b[k,j+1] == optimum for k in range(n+1))
        if len(other_is) == 1 and not optional_j:
            result[i] = j
    return result


def scoring_mapping(sung_words, scoring_words, contractions):
    """Contraction expansion checks identity but never fabricates phone splits."""
    expanded, owner, sizes = [], [], []
    for i, word in enumerate(sung_words):
        tokens = contractions.get(word, word).split()
        expanded.extend(tokens); owner.extend([i]*len(tokens)); sizes.append(len(tokens))
    mapped = mandatory_equal_mapping(expanded, scoring_words)
    result = [-1]*len(sung_words)
    for i, j in enumerate(mapped):
        if j >= 0 and sizes[owner[i]] == 1:
            result[owner[i]] = j
    return result


@torch.no_grad()
def path_posterior(emissions, temperature=.2):
    """Exact state occupancy under the frozen matcher's ordered path graph.

    Input [T,N] cosine emissions. Returns [T,N] unnormalized phone-state
    occupancy (background mass is separate), log partition and diagnostics.
    Uses float64 to keep forward/backward subtraction stable. Every phone is
    visited at least once; repeated positions need no CTC blank.
    """
    if emissions.ndim != 2 or not torch.isfinite(emissions).all() or temperature <= 0:
        raise ValueError('Invalid emissions/temperature')
    t, n = emissions.shape
    if not 0 < n <= t:
        raise ValueError('Each phone needs at least one frame')
    e = emissions.double()/temperature
    states = 2*n+1
    x = e.new_zeros((t, states)); x[:,1::2] = e
    a = e.new_full((t, states), -torch.inf)
    a[0,0] = 0.; a[0,1] = e[0,0]
    odd = torch.arange(states, device=e.device) % 2 == 1
    target_skip = odd & (torch.arange(states, device=e.device) >= 3)
    source_skip = odd & (torch.arange(states, device=e.device) < states-2)
    for i in range(1,t):
        prev = a[i-1]
        one = F.pad(prev[:-1], (1,0), value=-torch.inf)
        two = F.pad(prev[:-2], (2,0), value=-torch.inf).masked_fill(~target_skip, -torch.inf)
        a[i] = torch.logsumexp(torch.stack((prev,one,two)),0)+x[i]
    z = torch.logsumexp(a[-1,-2:],0)
    beta = e.new_full((t,states), -torch.inf); beta[-1,-2:] = 0.
    for i in range(t-2,-1,-1):
        nxt = beta[i+1]+x[i+1]
        one = F.pad(nxt[1:], (0,1), value=-torch.inf)
        two = F.pad(nxt[2:], (0,2), value=-torch.inf).masked_fill(~source_skip, -torch.inf)
        beta[i] = torch.logsumexp(torch.stack((nxt,one,two)),0)
    all_states = (a+beta-z).exp()
    p = all_states[:,1::2]
    gap = float((all_states.sum(1)-1).abs().max())
    minimum = float(p.sum(0).min())
    if not torch.isfinite(p).all() or gap > 1e-7 or minimum < 1-1e-7:
        raise ValueError('Invalid ordered-path posterior')
    return p.float(), z, dict(state_mass_max_gap=gap, minimum_phone_frame_mass=minimum)


def transfer_weights(posterior, reference_seconds, heard_frames, heard_seconds, frame_step=.02):
    """Transport by physical bin overlap, never stretch different durations.

    Both streams are genuine 50Hz valid encoder frames. Short final bins use
    actual audio duration. Uncovered tail is reported, then surviving weights
    are normalized per phone. No extrapolation into nonexistent audio.
    """
    t, n = posterior.shape
    if not reference_seconds > 0 or not heard_seconds > 0 or heard_frames < 1:
        raise ValueError('Invalid stream durations')
    if abs(t - int(np.ceil(reference_seconds/frame_step))) > 0 or heard_frames != int(np.ceil(heard_seconds/frame_step)):
        raise ValueError('Valid-frame/duration mismatch')
    device, dtype = posterior.device, posterior.dtype
    rs = torch.arange(t, device=device, dtype=dtype)*frame_step
    re = torch.minimum(rs+frame_step, rs.new_tensor(reference_seconds))
    hs = torch.arange(heard_frames, device=device, dtype=dtype)*frame_step
    he = torch.minimum(hs+frame_step, hs.new_tensor(heard_seconds))
    overlap = (torch.minimum(re[:,None],he[None])-torch.maximum(rs[:,None],hs[None])).clamp_min(0)
    ref = posterior*(re-rs)[:,None]
    hear = overlap.T @ posterior
    mass, received = ref.sum(0), hear.sum(0)
    valid = (mass > 1e-12) & (received > 1e-12)
    reference = ref/mass.clamp_min(1e-12)
    heard = hear/received.clamp_min(1e-12)
    return reference, heard, valid, (received/mass.clamp_min(1e-12)).clamp(0,1)


@torch.no_grad()
def phone_evidence(text, reference, heard, posterior, reference_seconds, heard_seconds):
    """Keep all phone positions until the verified scoring-word aggregation."""
    wr, wh, valid, coverage = transfer_weights(posterior, reference_seconds, len(heard), heard_seconds)
    r, h = wr.T @ reference, wh.T @ heard
    rc, hc = (r*text).sum(-1), (h*text).sum(-1)
    entropy = -(wr*wr.clamp_min(1e-12).log()).sum(0)/np.log(max(2,len(reference)))
    duration = (posterior.sum(0)*.02/reference_seconds).clamp(0,1)
    scalars = torch.stack((hc,rc,duration,entropy,coverage),-1)
    value = torch.cat((text,r,h,h-r,scalars),-1)
    if value.shape[1] != FEATURE_DIM or not torch.isfinite(value).all():
        raise ValueError('Nonfinite/invalid phone evidence')
    return value, valid


def word_evidence(phone_values, phone_to_word, phone_valid, word_count):
    if word_count < 1 or len(phone_values) != len(phone_to_word):
        raise ValueError('Invalid word mapping')
    belongs = torch.as_tensor(phone_to_word, device=phone_values.device, dtype=torch.long)
    keep = phone_valid & (belongs >= 0) & (belongs < word_count)
    result = phone_values.new_zeros((word_count,FEATURE_DIM)); count = phone_values.new_zeros(word_count)
    result.index_add_(0,belongs[keep],phone_values[keep])
    count.index_add_(0,belongs[keep],torch.ones_like(count[belongs[keep]]))
    return result/count.clamp_min(1)[:,None], count > 0


class WordCorrection(nn.Module):
    """Shared small word network; zero-start bounded logit corrections.

    Unsupported scoring words retain the baseline probability and still count
    in the sentence denominator. Context never grants an unsupported word an
    invented correction. The D6e trunk stays frozen.
    """
    def __init__(self, width=32):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(FEATURE_DIM),nn.Linear(FEATURE_DIM,width),nn.GELU())
        self.temporal = nn.Conv1d(width,width,3,padding=1,groups=width)
        self.mix = nn.Linear(width,width)
        self.output = nn.Linear(width,1)
        nn.init.zeros_(self.output.weight); nn.init.zeros_(self.output.bias)

    def forward(self, values, evidence_mask, word_mask, base_logit):
        if values.ndim != 3 or evidence_mask.shape != word_mask.shape or values.shape[:2] != word_mask.shape:
            raise ValueError('Invalid word batch')
        if not bool(word_mask.any(1).all()) or bool((evidence_mask & ~word_mask).any()):
            raise ValueError('Empty sentence/invalid evidence mask')
        safe = values.masked_fill(~evidence_mask[...,None],0.)
        x = self.project(safe).masked_fill(~evidence_mask[...,None],0.)
        local = self.temporal(x.transpose(1,2)).transpose(1,2)
        x = (x+self.mix(F.gelu(local))).masked_fill(~evidence_mask[...,None],0.)
        delta = .5*torch.tanh(self.output(x).squeeze(-1))*evidence_mask
        words = torch.sigmoid(base_logit[:,None]+delta)
        return (words*word_mask).sum(1)/word_mask.sum(1), words, delta
