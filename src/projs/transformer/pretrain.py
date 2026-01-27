from src.core.loss.mask_ce_loss import MaskedCrossEntropyLoss
import torch.nn as nn
import torch

class transformer_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MaskedCrossEntropyLoss(reduction='none')
    
    def forward(self, tgt_hat, tgt, tgt_valid_lens):
        # from network: tgt_hat, tensor of logits(batch_size, context_size, vocab_size)
        # from dataset/label: tgt(batch_size, context_size), tgt_valid_lens(batch_size, )
        # label_mask: (batch_size, context_size)bool
        label_mask = torch.arange(tgt.size(1), dtype=torch.int64, device=tgt_valid_lens.device).unsqueeze(0) < tgt_valid_lens.unsqueeze(1)
        return self.loss(tgt_hat.transpose(1, 2), tgt, label_mask)