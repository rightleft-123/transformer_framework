import torch
import torch.nn as nn

from attention_mask import get_len_mask, get_subsequent_mask, get_enc_dec_mask


class Transformer(nn.Module):
    def __init__(self, frontend: nn.Module, encoder: nn.Module, decoder: nn.Module,
                 dec_out_dim: int, vocab: int) -> None:
        """
            Initialize the model.

            Args:
                frontend: Feature extraction module, such as a neural network for processing input features.
                encoder: Encoder module for encoding input sequences.
                decoder: Decoder module for decoding output sequences.
                dec_out_dim: Dimension of the decoder output.
                vocab: Size of the vocabulary, representing the number of unique tokens in the output.
        """
        super().__init__()
        # feature extractor
        self.frontend = frontend
        self.encoder = encoder
        self.decoder = decoder
        self.linear = nn.Linear(dec_out_dim, vocab)

    def forward(self, X: torch.Tensor, X_lens: torch.Tensor, labels: torch.Tensor):
        X_lens, labels = X_lens.long(), labels.long()
        b = X.size(0)
        device = X.device

        # Frontend
        out = self.frontend(X)
        max_feat_len = out.size(1)
        max_label_len = labels.size(1)

        # Encoder
        enc_mask = get_len_mask(b, max_feat_len, X_lens, device)
        enc_out = self.encoder(out, X_lens, enc_mask)

        # Decoder
        dec_mask = get_subsequent_mask(b, max_label_len, device)
        dec_enc_mask = get_enc_dec_mask(b, max_feat_len, X_lens, max_label_len, device)
        dec_out = self.decoder(labels, enc_out, dec_mask, dec_enc_mask)
        logits = self.linear(dec_out)

        return logits
