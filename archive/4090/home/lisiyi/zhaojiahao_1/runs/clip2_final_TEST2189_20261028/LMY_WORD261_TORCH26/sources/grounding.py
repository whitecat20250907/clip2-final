"""Learned ordered acoustic/phone matching, independent of score regression."""
from functools import lru_cache
import torch
from torch import nn
from torch.nn import functional as F

PHONES = tuple(('AA AE AH AO AW AY B CH D DH EH ER EY F G HH IH IY JH K L M N NG '
                'OW OY P R S SH T TH UH UW V W Y Z ZH').split())
VOCAB = {name: i + 1 for i, name in enumerate(PHONES)}


def safe_lse(x, dim):
    """All-unreachable states remain -inf with zero gradient, not NaN."""
    reachable = torch.isfinite(x).any(dim=dim)
    safe = torch.where(reachable.unsqueeze(dim), x, torch.zeros_like(x))
    return torch.where(reachable, torch.logsumexp(safe, dim), x.new_full((), -torch.inf))


def log_partition(emissions, frame_lengths, phone_lengths):
    """Sum legal paths through BG0, phone1, BG1, ... phoneN, BGN.

    Every step consumes one frame. Stay, advance 1, or phone->next phone
    skipping a background. Every phone visited >=1 time; no phone skip.
    Start at BG0/phone1 and terminate at last phone/last background.
    Repeated phone *positions* are separate states and need no CTC blank.
    """
    if emissions.ndim != 3 or not emissions.is_floating_point():
        raise ValueError('Expected floating [batch,time,phone] compatibility')
    b, t, n = emissions.shape
    fl = torch.as_tensor(frame_lengths, device=emissions.device, dtype=torch.long)
    pl = torch.as_tensor(phone_lengths, device=emissions.device, dtype=torch.long)
    if fl.shape != (b,) or pl.shape != (b,) or not bool(((fl >= pl) & (pl > 0) & (pl <= n) & (fl <= t)).all()):
        raise ValueError('Every phone requires one frame; invalid lengths')
    if not bool(torch.isfinite(emissions).all()):
        raise ValueError('Nonfinite emissions')
    s = 2 * n + 1
    states = torch.arange(s, device=emissions.device)
    valid = states[None] <= 2 * pl[:, None]
    skip = ((states >= 3) & (states % 2 == 1))[None]
    # Float32/float64 only: long paths must not accumulate in fp16.
    if emissions.dtype not in (torch.float32, torch.float64):
        emissions = emissions.float()
    alpha = emissions.new_full((b, s), -torch.inf)
    alpha[:, 0] = 0
    alpha[:, 1] = emissions[:, 0, 0]
    alpha = alpha.masked_fill(~valid, -torch.inf)
    for i in range(1, t):
        one = F.pad(alpha[:, :-1], (1, 0), value=-torch.inf)
        two = F.pad(alpha[:, :-2], (2, 0), value=-torch.inf).masked_fill(~skip, -torch.inf)
        predecessors = safe_lse(torch.stack((alpha, one, two), dim=0), 0)
        emission = torch.stack((emissions.new_zeros((b, n)), emissions[:, i]), -1).flatten(1)
        emission = F.pad(emission, (0, 1))
        updated = (predecessors + emission).masked_fill(~valid, -torch.inf)
        alpha = torch.where((i < fl)[:, None], updated, alpha)
    last = torch.stack((alpha.gather(1, (2 * pl - 1)[:, None]).squeeze(1),
                        alpha.gather(1, (2 * pl)[:, None]).squeeze(1)), dim=1)
    return safe_lse(last, 1)


@lru_cache(maxsize=8192)
def zero_partition(t, n):
    # Same graph in scalar Python, no tensor/kernel launches for path count.
    import math
    a = [0., 0.] + [-math.inf] * (2 * n - 1)
    def lse(values):
        m = max(values)
        return m if not math.isfinite(m) else m + math.log(sum(math.exp(v - m) for v in values))
    for _ in range(1, t):
        a = [lse([a[j]] + ([a[j-1]] if j else []) +
                 ([a[j-2]] if j >= 3 and j % 2 else [])) for j in range(2 * n + 1)]
    return lse(a[-2:])


def matching_score(emissions, frame_lengths, phone_lengths, temperature=.2):
    if not temperature > 0:
        raise ValueError('Positive temperature required')
    z = log_partition(emissions / temperature, frame_lengths, phone_lengths)
    fl = torch.as_tensor(frame_lengths).detach().cpu().tolist()
    pl = torch.as_tensor(phone_lengths).detach().cpu().tolist()
    neutral = z.new_tensor([zero_partition(t, n) for t, n in zip(fl, pl)])
    return (z - neutral) / z.new_tensor(pl)


class Matcher(nn.Module):
    def __init__(self, width=64):
        super().__init__()
        self.audio_norm = nn.LayerNorm(1280)
        self.audio_projection = nn.Linear(1280, width)
        self.audio_temporal = nn.Conv1d(width, width, 3, padding=1, groups=width)
        self.audio_mix = nn.Linear(width, width)
        self.phone_embedding = nn.Embedding(len(VOCAB) + 1, width, padding_idx=0)
        self.boundary_embedding = nn.Embedding(2, width)
        self.phone_temporal = nn.Conv1d(width, width, 5, padding=2, groups=width)
        self.phone_mix = nn.Linear(width, width)

    def encode_audio(self, frames, lengths):
        mask = torch.arange(frames.shape[1], device=frames.device)[None] < lengths[:, None]
        x = F.gelu(self.audio_projection(self.audio_norm(frames.float()))) * mask[..., None]
        x = x + self.audio_mix(F.gelu(self.audio_temporal(x.transpose(1, 2)).transpose(1, 2)))
        return F.normalize(x, dim=-1) * mask[..., None]

    def encode_phones(self, ids, boundaries, lengths):
        mask = torch.arange(ids.shape[1], device=ids.device)[None] < lengths[:, None]
        x = (self.phone_embedding(ids) + self.boundary_embedding(boundaries)) * mask[..., None]
        x = x + self.phone_mix(F.gelu(self.phone_temporal(x.transpose(1, 2)).transpose(1, 2)))
        return F.normalize(x, dim=-1) * mask[..., None]

    def forward(self, frames, frame_lengths, phones, boundaries, phone_lengths):
        audio = self.encode_audio(frames, frame_lengths)
        text = self.encode_phones(phones, boundaries, phone_lengths)
        return matching_score(torch.bmm(audio, text.transpose(1, 2)), frame_lengths, phone_lengths)


def pair_loss(scores):
    """Column 0 is the positive; remaining columns are verified text negatives."""
    if scores.ndim != 2 or scores.shape[1] < 2 or not torch.isfinite(scores).all():
        raise ValueError('Finite [pairs,positive+negatives] scores required')
    return F.cross_entropy(scores, torch.zeros(len(scores), dtype=torch.long, device=scores.device))
