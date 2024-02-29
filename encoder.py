import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_encoder import pos_sinusoid_embedding
from position_wise_ffn import PoswiseFFN


class EncoderLayer(nn.Module):
    def __init__(self, dim, num_head, ffn_dim, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            num_head: number of attention heads
            ffn_dim: dimension of PosFFN (Positional Feed Forward Network)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """

        assert dim % num_head == 0
        hdim = dim // num_head
        super(EncoderLayer, self).__init__()
        # LayerNorm
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        # Multiple head attention
        self.multi_head_attn = MultiHeadAttention(hdim, hdim, dim, num_head, dropout_attn)
        # Position-wise FNN
        self.poswise_ffn = PoswiseFFN(dim, ffn_dim, p=dropout_posffn)

    def forward(self, enc_in, attn_mask):
        residual = enc_in
        context = self.multi_head_attn(enc_in, enc_in, enc_in, attn_mask)
        out = self.norm1(residual + context)
        residual = out

        out = self.poswise_ffn(out)
        out = self.norm2(residual + out)

        return out


class Encoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn, num_layers, enc_dim, num_heads, ffn_dim, max_len):
        """
            Args:
                dropout_emb: dropout ratio of Position Embeddings.
                dropout_posffn: dropout ratio of PosFFN.
                dropout_attn: dropout ratio of attention module.
                num_layers: number of encoder layers
                enc_dim: input dimension of encoder
                num_heads: number of attention heads
                ffn_dim: dimensionf of Position-wise FFN
                max_len: the maximum length of sequences
        """
        super(Encoder, self).__init__()
        # The maximum length of input sequence
        self.max_len = max_len
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(max_len, enc_dim), freeze=True)
        self.emb_dropout = nn.Dropout(dropout_emb)
        self.layers = nn.ModuleList(
            [EncoderLayer(enc_dim, num_heads, ffn_dim, dropout_posffn, dropout_attn) for _ in range(num_layers)]
        )

    def forward(self, X, X_lens, mask=None):
        batch_size, seq_len, d_model = X.shape
        out = X + self.pos_emb(torch.arange(seq_len, device=X.device))
        out = self.emb_dropout(out)

        for layer in self.layers:
            out = layer(out, mask)

        return out
    