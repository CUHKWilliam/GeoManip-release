import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math, copy
from torch.autograd import Variable

TRAIN = True

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, use_sc=False):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.use_sc = use_sc

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k
        if not self.use_sc:
            query, key = \
                [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                 for l, x in zip(self.linears, (query, key))]

        else:
            query_dir, key_dir = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2) for l, x in zip([self.linears[0], self.linears[0]], (query, key))]
            query_norm = self.linears[1](query)[:, :, :self.h].view(nbatches, -1, self.h).transpose(1, 2)
            key_norm = self.linears[1](key)[:, :, :self.h].view(nbatches, -1, self.h).transpose(1, 2)
            query = query_dir / query_dir.norm(dim=-1).unsqueeze(-1) * 10 * query_norm.unsqueeze(-1)
            key = key_dir / key_dir.norm(dim=-1).unsqueeze(-1) * 10 * key_norm.unsqueeze(-1)
        
        value = value.repeat(self.h, 1, 1).transpose(0, 1).contiguous().unsqueeze(-1)
        
        if not TRAIN:
            query = query.detach().cpu()
            key = key.detach().cpu()
            value = value.detach().cpu()
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        if not TRAIN:
            x = x.cuda()
        # 3) "Concat" using a view and apply a final linear.
        return torch.mean(x, -3)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=10000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])
