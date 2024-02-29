import torch


def get_len_mask(b: int, max_len: int, feat_lens: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
        Generate an attention mask for variable-length sequences.

        Args:
            b: Batch size
            max_len: Maximum sequence length
            feat_lens: Tensor containing the actual lengths of sequences in the batch
            device: Device on which the mask should be created

        Returns:
            torch.Tensor: Attention mask
    """
    attention_mask = torch.ones((b, max_len, max_len), device=device)
    for i in range(b):
        attention_mask[i, :, :feat_lens[i]] = 0
    return attention_mask.to(torch.bool)


def get_subsequent_mask(b: int, max_len: int, device: torch.device) -> torch.Tensor:
    """
        Generate a subsequent mask to prevent attending to future tokens.

        Args:
            b: Batch size
            max_len: Maximum sequence length
            device: Device on which the mask should be created

        Returns:
            torch.Tensor: Subsequent mask
    """
    return torch.triu(torch.ones((b, max_len, max_len), device=device), diagonal=1).to(torch.bool)


def get_enc_dec_mask(b: int, max_feat_len: int, feat_lens: torch.Tensor, max_label_len: int, device: torch.device) -> \
        torch.Tensor:
    """
       Generate an attention mask for encoder-decoder attention.

       Args:
           b: Batch size
           max_feat_len: Maximum feature sequence length (from encoder)
           feat_lens: Tensor containing the actual lengths of feature sequences in the batch
           max_label_len: Maximum label sequence length (from decoder)
           device: Device on which the mask should be created

       Returns:
           torch.Tensor: Attention mask
   """
    attention_mask = torch.zeros((b, max_label_len, max_feat_len), device=device)
    for i in range(b):
        attention_mask[i, :, feat_lens[i]:] = 1
    return attention_mask.to(torch.bool)


'''
m = get_len_mask(2, 4, torch.tensor([2, 4]), "cpu")
m = get_subsequent_mask(2, 4, "cpu")
m = m.int()

print(m)
'''