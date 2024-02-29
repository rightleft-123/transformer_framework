import torch
import torch.nn as nn

from encoder import Encoder
from decoder import Decoder
from transformer import Transformer


if __name__ == '__main__':
    # ----------  Constants  ----------
    batch_size = 16
    max_feature_len = 100
    max_label_len = 50
    mel_fbank_dim = 80  # The dimension of Mel filter bank
    hidden_dim = 512
    eng_letter_size = 26

    # ----------  Data preparing  ----------
    # Generate input sequence
    mel_fbank_feature = torch.randn(batch_size, max_feature_len, mel_fbank_dim)
    # The length of each input sequence in the batch
    feature_lens = torch.randint(1, max_feature_len, (batch_size,))
    # Generate output sequence
    labels = torch.randint(0, eng_letter_size, (batch_size, max_label_len))
    # The length of each output sequence in the batch
    labels_lens = torch.randint(1, max_label_len, (batch_size,))

    # ----------  Model preparing  ----------
    feature_extractor = nn.Linear(mel_fbank_dim, hidden_dim)

    encoder = Encoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, enc_dim=hidden_dim, num_heads=8, ffn_dim=2048, max_len=2048
    )

    decoder = Decoder(
        dropout_emb=0.1, dropout_posffn=0.1, dropout_attn=0.,
        num_layers=6, dec_dim=hidden_dim, num_heads=8, dff=2048, tgt_len=2048, max_eng_letter_siz=eng_letter_size
    )

    transformer = Transformer(
        feature_extractor, encoder, decoder, hidden_dim, eng_letter_size
    )

    # ----------  Forward propagation  ----------
    result = transformer(mel_fbank_feature, feature_lens, labels)
    print(f"result size: {result.shape}")
    print(f"result: {result}")
