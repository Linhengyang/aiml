from src.core.loss.mask_ce_loss import MaskedCrossEntropyLoss
import torch.nn as nn
import torch

class transformer_loss(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss = MaskedCrossEntropyLoss(*args, **kwargs)
    
    def forward(self, Y_hat, Y_label, Y_valid_lens):
        # Y_hat, tensor of logits(batch_size, num_steps, vocab_size)
        # Y_label_batch: (batch_size, num_steps)
        # Y_valid_lens_batch: (batch_size,)

        valid_area = torch.arange(Y_label.size(1), dtype=torch.int32, device=Y_valid_lens.device).unsqueeze(0) < Y_valid_lens.unsqueeze(1)

        return self.loss(Y_hat.permute(0,2,1), Y_label, valid_area)