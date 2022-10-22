import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy
import warnings
import numpy as np
from einops import rearrange

warnings.simplefilter("ignore")
print(torch.__version__)

class PositionalEmbedding(nn.Module):

    # register buffer in Pytorch ->
    # If you have parameters in your model, which should be saved and restored in the state_dict,
    # but not trained by the optimizer, you should register them as buffers.

    def __init__(self, max_seq_len, embed_model_dim):
        """
        Args:
            seq_len: max length of input sequence
            embed_model_dim: demension of embedding
        """
        super(PositionalEmbedding, self).__init__()

        pe = torch.zeros(max_seq_len, embed_model_dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        # div_term = torch.exp(
        #     torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        # )
        div_term = 1 / (10000 ** ((2 * np.arange(embed_model_dim)) / embed_model_dim))
        pe[:, 0::2] = torch.sin(position * div_term[0::2])
        pe[:, 1::2] = torch.cos(position * div_term[1::2])

        self.register_buffer('pe', pe)
        # 位置信息不需要train

    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            x: output
        """
        # make embeddings relatively larger
        x = x * math.sqrt(self.embed_dim) # Maybe do not need this ?
        # add constant to embedding
        seq_len = x.size(1) # ( batch ,seq_len , embedding dim)
        x = x + self.pe[:seq_len].repeat(x.size(0),1,1)
        return x



class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads

        A tutorial : https://theaisummer.com/einsum-attention/
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        assert self.embed_dim % self.n_heads == 0 , 'embed_dim must be divisible by n_heads !'
        self.single_head_dim = self.embed_dim / self.n_heads
        self.scale_factor = self.single_head_dim ** -0.5
        self.to_qkv = nn.Linear(embed_dim,embed_dim*3,bias=False)
        self.output = nn.Linear(embed_dim,embed_dim)

    def forward(self, x, mask=None):
        assert x.dim() == 3 , 'The input must be of the form (batch, seq_len, dim) !'
        qkv = self.to_qkv(x) # shape with ( b , seq , dim * 3 )
        q, k, v = tuple(rearrange(qkv, 'b s (k d h) -> k b h s d ', k=3, h=self.heads))
        # here b : batch  , s : seqlen , k : 3 , h : head num , d : dim // head
        scaled_dot_prod = torch.einsum('b h i d , b h j d -> b h i j', q, k) * self.scale_factor
        # here i = j = s , which is the seq_len
        if mask is not None:
            assert mask.shape == scaled_dot_prod.shape[2:] , ' MASK should be a square matrix of seq * seq !'
            scaled_dot_prod = scaled_dot_prod.masked_fill(mask, -np.inf)

        attention = torch.softmax(scaled_dot_prod, dim=-1) # ( b , h , s , s) , where is last dim will be scaled
        out = torch.einsum('b h i j , b h j d -> b h i d', attention, v)
        out = rearrange(out,'b h s d -> b s (h d)')
        output = self.output(out)
        return output
