from src.core.models.vit import ViTEncoder, vitConfig
import torch.nn as nn

class ViTClassifier(nn.Module):
    def __init__(self, config: vitConfig):
        
        super().__init__()
        # vit encoder: (batch_size, num_channels, height, width) --encoder--> (batch_size, 1+num_patches, num_hiddens)
        self.encoder = ViTEncoder(config)
        # get-bos: (batch_size, 1+num_patches, num_hiddens) --get class token on dim 1 at index 0-->(batch_size, num_hiddens)
        # classify层: (batch_size, num_hiddens) --dense-->  (batch_size, num_classes)
        self.head = nn.Sequential(nn.LayerNorm(config.num_hiddens), nn.Linear(config.num_hiddens, 10))


    def forward(self, X):
        # input shape: (batch_size, num_channels, h, w)
        X = self.encoder(X) # (batch_size, 1+num_patches, num_hiddens)
        
        return self.head(X[:, 0, :]) #output shape: (batch_size, num_classes)

