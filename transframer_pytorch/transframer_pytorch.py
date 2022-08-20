from functools import partial
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

# unet

class Block(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x):
        x = self.proj(x)
        x = self.norm(x)
        return self.act(x)

class ResnetBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        groups = 8
    ):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

class Unet(nn.Module):
    def __init__(
        self,
        dim,
        *,
        dim_mults = (1, 2, 3, 4),
        dim_out
    ):
        super().__init__()
        self.to_out = nn.Conv2d(dim, dim_out, 1)
        dims = [dim, *map(lambda t: t * dim, dim_mults)]
        dim_pairs = tuple(zip(dims[:-1], dims[1:]))
        mid_dim = dims[-1]

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        self.mid = ResnetBlock(mid_dim, mid_dim)

        for dim_in, dim_out in dim_pairs:
            self.downs.append(nn.ModuleList([
                ResnetBlock(dim_in, dim_in),
                nn.Conv2d(dim_in, dim_out, 3, 2, 1)
            ]))

            self.ups.insert(0, nn.ModuleList([
                ResnetBlock(dim_out * 2, dim_out),
                nn.ConvTranspose2d(dim_out, dim_in, 4, 2, 1)
            ]))

    def forward(self, x):

        hiddens = []

        for block, downsample in self.downs:
            x = block(x)
            x = downsample(x)
            hiddens.append(x)

        x = self.mid(x)

        for block, upsample in self.ups:
            x = torch.cat((x, hiddens.pop()), dim = 1)
            x = block(x)
            x = upsample(x)

        out = self.to_out(x)
        return rearrange(out, 'b c h w -> b (h w) c')

# main class

class Transframer(nn.Module):
    def __init__(
        self,
        *,
        unet: Unet,
        dim,
        depth,
        max_channels,
        max_positions,
        max_values,
        dim_head = 32,
        heads = 8,
        ff_mult = 4.,
        ignore_index = -100
    ):
        super().__init__()
        self.unet = unet

        self.start_token = nn.Parameter(torch.randn(dim))

        self.channels = nn.Embedding(max_channels, dim)
        self.positions = nn.Embedding(max_positions, dim)
        self.values = nn.Embedding(max_values, dim)

        self.postemb_norm = nn.LayerNorm(dim) # done in Bloom and YaLM for stability

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, dim_head = dim_head, heads = heads, causal = True),
                Attention(dim, dim_head = dim_head, heads = heads, norm_context = True),
                FeedForward(dim, mult = ff_mult)
            ]))

        self.final_norm = nn.LayerNorm(dim)

        # give channels and positions separate embedding for final prediction

        self.axial_channels = nn.Embedding(max_channels, dim)
        self.axial_positions = nn.Embedding(max_positions, dim)

        self.axial_attn = Attention(dim, dim_head = dim_head,  heads = heads, causal = True)
        self.axial_ff = FeedForward(dim, mult = ff_mult)

        self.axial_final_norm = nn.LayerNorm(dim)

        # projection to logits

        self.to_channel_logits = nn.Linear(dim, max_channels)
        self.to_position_logits = nn.Linear(dim, max_positions)
        self.to_value_logits = nn.Linear(dim, max_values)

        self.ignore_index = ignore_index

    def forward(
        self,
        x,
        context_frames,
        return_loss = False
    ):
        assert x.shape[-1] == 3

        encoded = self.unet(context_frames)

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

        embed = self.final_norm(embed)

        # now do axial attention from the summed previous embedding of channel + position + value -> next channel -> next position
        # this was successfully done in the residual quantization transformer (RQ-Transformer) https://arxiv.org/abs/2203.01941
        # one layer of attention should be enough, as in the Deepmind paper, they use a pretty weak baseline and it still worked well

        axial_channels_emb = self.axial_channels(channels)
        axial_positions_emb = self.axial_positions(positions)

        embed = torch.stack((embed, axial_channels_emb, axial_positions_emb), dim = -2)

        embed = rearrange(embed, 'b m n d -> (b m) n d')

        embed = self.axial_attn(embed) + embed
        embed = self.axial_ff(embed) + embed

        embed = self.axial_final_norm(embed)

        embed = rearrange(embed, '(b m) n d -> b m n d', b = batch)

        pred_channel_embed, pred_position_embed, pred_value_embed = embed.unbind(dim = -2)

        # to logits

        channel_logits = self.to_channel_logits(pred_channel_embed)
        position_logits = self.to_position_logits(pred_position_embed)
        value_logits = self.to_value_logits(pred_value_embed)

        if not return_loss:
            return channel_logits, position_logits, value_logits

        channel_logits, position_logits, value_logits = map(lambda t: rearrange(t, 'b n c -> b c n'), (channel_logits, position_logits, value_logits))

        ce = partial(F.cross_entropy, ignore_index = self.ignore_index)

        channel_loss = ce(channel_logits, channels)
        position_loss = ce(position_logits, positions)
        value_loss = ce(value_logits, values)

        return (channel_loss + position_loss + value_loss) / 3
