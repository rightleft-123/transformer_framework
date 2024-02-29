import torch
import torch.nn as nn

from multi_head_attention import MultiHeadAttention
from position_encoder import pos_sinusoid_embedding
from position_wise_ffn import PoswiseFFN


class DecoderLayer(nn.Module):
    def __init__(self, dim, num_head, ffn_dim, dropout_posffn, dropout_attn):
        """
        Args:
            dim: input dimension
            num_head: number of attention heads
            ffn_dim: dimention of PosFFN (Positional FeedForward)
            dropout_posffn: dropout ratio of PosFFN
            dropout_attn: dropout ratio of attention module
        """

        super(DecoderLayer, self).__init__()
        assert dim % num_head == 0
        hdim = dim // num_head
        # LayerNorms
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        # Position-wise FFN
        self.poswise_ffn = PoswiseFFN(dim, ffn_dim, p=dropout_posffn)
        # MultiHeadAttention, both self-attention and encoder-decoder cross attention)
        self.dec_attn = MultiHeadAttention(hdim, hdim, dim, num_head, dropout_attn)
        self.enc_dec_attn = MultiHeadAttention(hdim, hdim, dim, num_head, dropout_attn)

    def forward(self, dec_in, enc_out, dec_mask, dec_enc_mask, cache=None, freqs_cis=None):
        # decoder's self-attention
        residual = dec_in
        context = self.dec_attn(dec_in, dec_in, dec_in, dec_mask)
        dec_out = self.norm1(residual + context)

        # encoder-decoder cross attention
        residual = dec_out
        context = self.dec_attn(dec_out, enc_out, enc_out, dec_enc_mask)
        dec_out = self.norm2(residual + context)

        # position-wise FFN
        residual = dec_out
        out = self.poswise_ffn(dec_out)
        dec_out = self.norm3(residual + out)

        return dec_out


class Decoder(nn.Module):
    def __init__(self, dropout_emb, dropout_posffn, dropout_attn, num_layers, dec_dim, num_heads, dff, tgt_len,
                 max_eng_letter_siz):
        """
            Args:
                dropout_emb: dropout ratio of Position Embeddings.
                dropout_posffn: dropout ratio of PosFFN.
                dropout_attn: dropout ratio of attention module.
                num_layers: number of encoder layers
                dec_dim: input dimension of decoder
                num_heads: number of attention heads
                dff: dimensionf of PosFFN
                tgt_len: the target length to be embedded.
                tgt_vocab_size: the target vocabulary size.
        """
        super(Decoder, self).__init__()
        # output embedding
        self.tgt_emb = nn.Embedding(max_eng_letter_siz, dec_dim)
        self.dropout_emb = nn.Dropout(p=dropout_emb)
        # position embedding
        self.pos_emb = nn.Embedding.from_pretrained(pos_sinusoid_embedding(tgt_len, dec_dim), freeze=True)
        # decoder layers
        self.layers = nn.ModuleList(
            [
                DecoderLayer(dec_dim, num_heads, dff, dropout_posffn, dropout_attn) for _ in
                range(num_layers)
            ]
        )

    def forward(self, labels, enc_out, dec_mask, dec_enc_mask, cache=None):
        # output embedding and position embedding
        tgt_emb = self.tgt_emb(labels)
        pos_emb = self.pos_emb(torch.arange(labels.size(1), device=labels.device))
        dec_out = self.dropout_emb(tgt_emb + pos_emb)

        # decoder layers
        for layer in self.layers:
            dec_out = layer(dec_out, enc_out, dec_mask, dec_enc_mask)
        return dec_out
