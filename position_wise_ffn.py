import torch
import torch.nn as nn


class PoswiseFFN(nn.Module):
    def __init__(self, model_dim, ffn_dim, p=0.):
        """
        Args:
            model_dim: input dimension for the model
            ffn_dim: dimension of Poswise Feed Forward Network (PoswiseFFN)
            p: dropout probability for the dropout layer
        """
        super(PoswiseFFN, self).__init__()
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.conv1 = nn.Conv1d(model_dim, ffn_dim, 1, 1, 0)
        self.conv2 = nn.Conv1d(ffn_dim, model_dim, 1, 1, 0)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=p)

    def forward(self, X):
        out = self.conv1(X.transpose(1, 2))     # (N, d_model, seq_len) -> (N, d_ff, seq_len)
        out = self.relu(out)
        out = self.conv2(out).transpose(1, 2)   # (N, d_ff, seq_len) -> (N, d_model, seq_len)
        out = self.dropout(out)
        return out
    