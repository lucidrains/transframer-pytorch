import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

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

# feedforward

def FeedForward(
    dim,
    *,
    mult = 4.
):
    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.LayerNorm(dim),
        nn.Linear(dim, inner_dim, bias = False),
        nn.GELU(),
        nn.LayerNorm(inner_dim),  # from normformer paper
        nn.Linear(inner_dim, dim, bias = False)
    )

# attention, what else?
# here we will use one headed key / values (as described in paper, from Noam Shazeer) - along with cosine sim attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_head = 64,
        heads = 8,
        scale = 10,
        causal = False,
        norm_context = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = scale
        self.causal = causal

        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(dim) if norm_context else nn.Identity()

        self.to_q = nn.Linear(dim, dim_head * heads, bias = False)
        self.to_kv = nn.Linear(dim, dim_head * 2, bias = False)
        self.to_out = nn.Linear(dim_head * heads, dim, bias = False)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        h, scale, causal, device = self.heads, self.scale, self.causal, x.device

        x = self.norm(x)

        context = default(context, x)

        q = self.to_q(x)
        q = rearrange(q, 'b n (h d) -> b h n d', h = h)

        if exists(context):
            context =self.norm_context(context)

        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k = map(l2norm, (q, k))

        sim = einsum('b h i d, b j d -> b h i j', q, k) * self.scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(context_mask, mask_value)

        if causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# main class

class Transframer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_channels,
        max_positions,
        max_values,
        dim_head = 32,
        heads = 8,
        ff_mult = 4.
    ):
        super().__init__()
        self.start_token = nn.Parameter(torch.randn(dim))

        self.channels = nn.Embedding(max_channels, dim)
        self.positions = nn.Embedding(max_positions, dim)
        self.values = nn.Embedding(max_values, dim)

        self.postemb_norm = nn.LayerNorm(dim) # done in Bloom and YaLM for stability

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads),
                Attention(dim, dim_head = dim_head, heads = heads, norm_context = True),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim)

        self.to_channel_logits = nn.Linear(dim, max_channels)
        self.to_position_logits = nn.Linear(dim, max_positions)
        self.to_value_logits = nn.Linear(dim, max_values)

    def forward(
        self,
        x,
        encoded,
        return_loss = False
    ):
        assert x.shape[-1] == 3

        batch = x.shape[0]

        channels, positions, values = x.unbind(dim = -1)

        channel_emb = self.channels(channels)
        position_emb = self.positions(positions)
        value_emb = self.values(values)

        embed = channel_emb + position_emb + value_emb

        start_token = repeat(self.start_token, 'd -> b 1 d', b = batch)
        embed = torch.cat((start_token, embed), dim = 1)

        if return_loss:
            embed = embed[:, :-1]

        embed = self.postemb_norm(embed)

        # layers of attention + cross attention

        for attn, cross_attn, ff in self.layers:
            embed = attn(embed) + embed
            embed = cross_attn(embed, encoded) + embed
            embed = ff(embed) + embed

        # to logits

        embed = self.final_norm(embed)

        channel_logits = self.to_channel_logits(embed)
        position_logits = self.to_position_logits(embed)
        value_logits = self.to_value_logits(embed)

        if not return_loss:
            return channel_logits, position_logits, value_logits

        channel_logits, position_logits, value_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), (channel_logits, position_logits, value_logits))

        channel_loss = F.cross_entropy(channel_logits, channels)
        position_loss = F.cross_entropy(channel_logits, channels)
        value_loss = F.cross_entropy(channel_logits, channels)

        return (channel_loss + position_loss + value_loss) / 3
