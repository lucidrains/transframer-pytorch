import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange

# helpers

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

# tensor helpers

def l2norm(t):
    return F.normalize(t, dim = -1)

# dct related encoding / decoding functions

def images_to_dct(images):
    raise NotImplementedError

def dct_to_images(images):
    raise NotImplementedError

# attention, what else?
# here we will use one headed key / values (as described in paper, from Noam Shazeer) - along with cosine sim attention

class Attention(nn.Module):
    def __init__(
        self,
        *,
        dim,
        dim_head = 64,
        heads = 8,
        scale = 10,
        causal = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias = False)

    def forward(self, x):
        h, scale, causal, device = self.heads, self.scale, self.causal, x.device

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        k, v = self.to_kv(x).chunk(2, dim = -1)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class Transframer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
