import torch
import numpy as np


def pos_sinusoid_embedding(seq_len, d_model):
    """
        Generate positional sinusoidal embeddings.

        Args:
            seq_len: Length of the sequence
            d_model: Dimension of the model

        Returns:
            torch.Tensor: Sinusoidal positional embeddings
    """
    embeddings = torch.zeros((seq_len, d_model))
    for i in range(d_model):
        f = torch.sin if i % 2 == 0 else torch.cos
        embeddings[:, i] = f(torch.arange(0, seq_len) / np.power(1e4, 2 * (i // 2) / d_model))
    return embeddings.float()
