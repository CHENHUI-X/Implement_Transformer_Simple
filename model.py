# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy
import warnings
warnings.simplefilter("ignore")
print(torch.__version__)


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        out = self.embed(x)
        return out


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
        self.embed_dim = embed_model_dim

        pe = torch.zeros(max_seq_len, self.embed_dim)
        # pos -> refers to order in the sentence
        # i -> refers to position along embedding vector dimension
        # if we have batch size of 32 and seq length of 10 and
        # let embedding dimension be 512. Then we will have embedding vector
        # of dimension 32 x 10 x 512. Similarly we will have positional encoding
        # vector of dimension 32 x 10 x 512. Then we add both.
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / self.embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / self.embed_dim)))
        pe = pe.unsqueeze(0)
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
        x = x * math.sqrt(self.embed_dim)
        # add constant to embedding
        seq_len = x.size(1) # ( batch ,seq_len , embedding dim)
        x = x + torch.autograd.Variable(self.pe[:, :seq_len], requires_grad=False)
        # 这一步不需要计算梯度
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim  # 512 dim
        self.n_heads = n_heads  # 8
        self.single_head_dim = int(self.embed_dim / self.n_heads)
        # 512/8 = 64  . each key,query, value will be of 64d

        # key,query and value matrix
        # here we use q,k,v in same dimension (64)
        # every single key (or query or value) matrix shape is 64 x 64,
        # which means in a single transform matrix , first 64 means the input dim is 64 (that's 512/head)
        # second 64 means the Key (or query or value) dimension is 64 (or whatever you want)
        # matrix for all 8 keys # 512x512
        self.query_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim,bias=False)
        self.key_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(
            self.single_head_dim, self.single_head_dim, bias=False)

        self.out = nn.Linear(
            self.n_heads * self.single_head_dim, self.embed_dim)

    def forward(self, key, query, value, mask=None):
        # input k : ( batch_size x sequence_length x key_embedding_dim)
        # input q : ( batch_size x sequence_length x query_embedding_dim)
        # input v : ( batch_size x sequence_length x value_embedding_dim)
        # here embedding dimension is same : 32 x 10 x 512


        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder

        Returns:
           output vector from multihead attention
        """

        batch_size = key.size(0)
        seq_length = key.size(1)
        # 32x10x512
        # print(key.shape,query.shape,value.shape)
        key = key.view(
            batch_size, seq_length, self.n_heads,self.single_head_dim)
        # batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)
        value = value.view(
            batch_size, seq_length, self.n_heads, self.single_head_dim)  # (32x10x8x64)

        # In mutihead attention , the q, k, v are transformed using a matrix respectively
        # Then implement the attention
        # 这里再次经过不同的 linner transform 得到计算attention用 q ,k ,v
        # 但 实际上这里 传进来的 key , query , value 就是input x 复制了 3份
        # 然后分别经过transform 得到 q , k , v
        k = self.key_matrix(key)  # (32x10x8x64) -> (32x10x8x64)
        q = self.query_matrix(query) # same as above
        v = self.value_matrix(value) # ~

        q = q.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        # (32 x 8 x 10 x 64)
        k = k.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1, 2)  # (batch_size, n_heads, seq_len, single_head_dim)

        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1, -2)
        # (batch_size, n_heads, single_head_dim, seq_ken)  # (32 x 8 x 64 x 10)

        product = torch.matmul(q, k_adjusted)   # Q*K^T
        # (32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
        # 10 * 10 ,  first line represents the attention of the first word with other words

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20"))

        # divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim)  # / sqrt(64)

        # applying softmax
        scores = F.softmax(product, dim=-1)

        # mutiply with value matrix
        scores = torch.matmul(scores, v)
        ## (32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64)

        # concatenated output
        concat = scores.transpose(1, 2).contiguous().view(
            batch_size, seq_length,self.single_head_dim * self.n_heads)
        # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)

        output = self.out(concat)  # (32,10,512) -> (32,10,512)

        return output


class EncoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(EncoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator which determines output dimension of linear layer
           n_heads: number of attention heads

        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)

        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim)
        )

        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

    def forward(self, key, query, value, mask=None):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attnetion(used only for the decoder)
        Returns:
           norm2_out: output of transformer block

        """

        attention_out = self.attention(key, query, value, mask)  # 32x10x512
        attention_residual_out = attention_out + value  # 32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out))  # 32x10x512

        feed_fwd_out = self.feed_forward(norm1_out)  # 32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out  # 32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))  # 32x10x512

        return norm2_out


class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention

    Returns:
        out: output of the encoder
    """

    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerEncoder, self).__init__()

        self.embedding_layer = Embedding(vocab_size, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                EncoderBlock(embed_dim, expansion_factor, n_heads) \
                for _ in range(num_layers)]
        )

    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out) # attention + forward
            # note : 实际上这里应该传进去的是 out 的 linner transform
            # 即输入 x 的 q,k,v , 然后再根据original paper 的 mutihead attention 显示的,
            # 再分别对 q,k,v , 分别经过一个linner transform,然后得到一个新的q,k,v去计算attention.
            # 但这里实际上第一个transform不做也行,即直接将x 复制三份 ,并直接将mutihead attention layer
            # 里边 在做attention之间的linner transform结果 作为q,k,v使用

        return out  # 32x10x512


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator which determines output dimension of linear layer
           n_heads: number of attention heads

        """
        # actually  just adds a masked mutihead attention layer to the encoder block
        self.attention = MultiHeadAttention(embed_dim, n_heads=8) # use masked
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.encoder_block = EncoderBlock(embed_dim, expansion_factor, n_heads)

    def forward(self, key, query, x, mask):
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention
        Returns:
           out: output of transformer block

        """
        # # the value come from previous layer ,that's generate from x
        # key and query come from output of encoder,that's the parameter

        mask_attention_out = self.attention(x, x, x, mask)  # 32x10x512
        value = self.dropout(self.norm(mask_attention_out + x))
        out = self.encoder_block(key, query, value, mask)

        return out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """
        self.word_embedding = nn.Embedding(target_vocab_size, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8)
                for _ in range(num_layers)
            ]

        )
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, enc_out, trg_mask):
        """
        Args:
            x: input vector from target
            enc_out : output from encoder layer
            trg_mask: mask for decoder self attention
        Returns:
            out: output vector
        """
        batch_size, seq_length = x.shape[0], x.shape[1]  # 32x10

        x = self.word_embedding(x)  # 32x10x512
        x = self.position_embedding(x)  # 32x10x512
        x = self.dropout(x)

        for layer in self.layers:
            # the value come from previous layer , that's x
            # key and query come from output of encoder , that's enc_out
            x = layer(enc_out, enc_out, x, trg_mask)

        out = F.softmax(self.fc_out(x),dim = -1)

        return out


class Transformer(nn.Module):
    def __init__(
          self,
          embed_dim,
          src_vocab_size,
          target_vocab_size,
          seq_length,
          num_layers=2,
          expansion_factor=4,
          n_heads=8):
        super(Transformer, self).__init__()

        """  
        Args:
           embed_dim:  dimension of embedding 
           src_vocab_size: vocabulary size of source
           target_vocab_size: vocabulary size of target
           seq_length : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention

        """

        self.encoder = TransformerEncoder(
            seq_length, src_vocab_size, embed_dim, num_layers = num_layers,
            expansion_factor = expansion_factor, n_heads = n_heads)
        self.decoder = TransformerDecoder(
            target_vocab_size, embed_dim, seq_length, num_layers = num_layers,
            expansion_factor = expansion_factor, n_heads = n_heads)

    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask

    def forward(self, src, trg):
        """
        Args:
            src: input to encoder , input sequence
            trg: input to decoder , target sequence
        out:
            out: final vector which returns probabilities of each target word
        """
        trg_mask = self.make_trg_mask(trg)
        enc_src = self.encoder(src)
        # the output of encoder , used to generate q , k for attention in decoder
        out = self.decoder(trg, enc_src, trg_mask)
        return out


if __name__ == '__main__':
    # test  : flip the input
    x1 = list(range(10))
    x2 = x1[::-1]

    x = torch.tensor([x1, x2])
    target = torch.tensor([x2, x1])

    src_vocab_size = 10
    target_vocab_size = 10
    num_layers = 2
    seq_length = 10

    model = Transformer(
        embed_dim=512,
        src_vocab_size=src_vocab_size,
        target_vocab_size=target_vocab_size,
        seq_length=seq_length,
        num_layers=num_layers,
        expansion_factor=4, n_heads=8)

    out = model(x, target)
    print(x.shape,target.shape)
    print(out.shape)
    import numpy as np
    print(np.argmax(out.detach().numpy(),-1))
