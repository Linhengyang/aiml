import torch.nn as nn
import math
import torch


class skipgramNegSp(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.context_embed = nn.Embedding(vocab_size, embed_size)
        self.center_embed = nn.Embedding(vocab_size, embed_size)

    def forwad(self, centers, context_negatives):
        # centers: (batch_size, 1)
        # context_negatives: (batch_size, 2*(K+1)*max_window_size)
        emb_x = self.center_embed(centers) # (batch_size, 1, embd_size)
        emb_y = self.context_embed(context_negatives) # (batch_size, num_infered_words, embd_size)
        return torch.bmm(emb_x, emb_y.permute(0, 2, 1)).squeeze(1) # (batch_size, 1, num_toinfer_words) -> (bs, num_toinfer_words)
    

class cbowNegSp(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.context_embed = nn.Embedding(vocab_size, embed_size)
        self.center_embed = nn.Embedding(vocab_size, embed_size)

    def forwad(self, contexts, center_negatives, masks):
        # contexts: (batch_size, 2*mask_window_size)
        # context_negatives: (batch_size, K+1)
        # masks: (batch_size, 2*mask_window_size)
        emb_x = self.context_embed(contexts) # (batch_size, 2*mask_window_size, embd_size)
        weights = masks / masks.sum(dim=1, keepdim=True) # (batch_size, 2*mask_window_size)
        weights = weights.unsqueeze(dim=1) # (batch_size, 1, 2*mask_window_size)
        emb_x_pool = torch.bmm(weights, emb_x) # (batch_size, 1, embd_size)
        emb_y = self.center_embed(center_negatives) # (batch_size, K+1, embd_size)
        return torch.bmm(emb_x_pool, emb_y.permute(0, 2, 1)).squeeze(1) # (batch_size, 1, K+1) -> (bs, K+1)