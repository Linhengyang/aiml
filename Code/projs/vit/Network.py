from ...Base.MetaFrames import Encoder
from ...Base.RootLayers.PositionalEncodings import LearnAbsPosEnc
from ...Base.SubModules.Patchify import Patchify
from ...Modules._vit import ViTEncoderBlock
import torch.nn as nn
import torch

class ViTEncoder(Encoder):
    def __init__(self,
                 img_shape,
                 patch_size,
                 num_blks,
                 num_heads,
                 num_hiddens,
                 emb_dropout,
                 blk_dropout,
                 mlp_num_hiddens,
                 num_classes=10,
                 use_bias=False):
        
        super().__init__()
        # patch化层: (batch_size, num_channels, height, width) --patchify--> (batch_size, num_patches, num_channels, height, width)
        self.patchify = Patchify(img_shape, patch_size)
        # flatten层: (batch_size, num_patches, num_channels, height, width) --flatten from dim 2--> (batch_size, num_patches, patch_flatlen)
        # dense1层: (batch_size, num_patches, patch_flatlen) --dense--> (batch_size, num_patches, num_hiddens)
        self.dense1 = nn.Linear(self.patchify.patch_flatlen, num_hiddens)
        # append层: (batch_size, num_patches, num_hiddens) --append class token on dim 1 at index 0--> (batch_size, 1+num_patches, num_hiddens)
        self.register_parameter('cls_token', nn.Parameter(torch.zeros(1, 1, num_hiddens)))
        # position encoding层: add position encodings with shape (batch_size, 1+num_patches, num_hiddens)
        self.pos_embedding = LearnAbsPosEnc(self.patchify.num_patches+1, num_hiddens, emb_dropout)

        # (batch_size, 1+num_patches, num_hiddens) --ViTEncoderBlock--> (batch_size, 1+num_patches, num_hiddens)
        self.blks = nn.Sequential()
        for i in range(num_blks):
            cur_blk = ViTEncoderBlock(num_heads, num_hiddens, blk_dropout, mlp_num_hiddens, use_bias)
            self.blks.add_module('vitEncBlk'+str(i), cur_blk)
        # get层: (batch_size, 1+num_patches, num_hiddens) --get class token on dim 1 at index 0-->(batch_size, num_hiddens)
        # predict层: (batch_size, num_hiddens) --dense-->  ((batch_size, num_classes)
        self.head = nn.Sequential(nn.LayerNorm(num_hiddens), nn.Linear(num_hiddens, num_classes))


    def forward(self, X):
        # input shape: (batch_size, num_channels, h, w)
        X = self.patchify(X).flatten(start_dim=2) #(batch_size, num_patches, patch_flatlen)
        X = self.dense1(X) #(batch_size, num_patches, num_hiddens)

        # cls_token (1, 1, num_hiddens) --expand batch_size at dim 0--> (batch_size, 1, num_hiddens)
        # append cls_token to data: (batch_size, num_patches, num_hiddens) --append (batch_size, 1, num_hiddens) on dim 1 at index 0-->
        # (batch_size, 1+num_patches, num_hiddens)
        X = torch.cat([self.cls_token.expand(X.shape[0], -1, -1), X], dim=1)


        X = self.pos_embedding(X) # (batch_size, seq_len=1+num_patches, num_hiddens)
        # (batch_size, seq_len=1+num_patches, num_hiddens) --> (batch_size, seq_len=1+num_patches, num_hiddens)
        for blk in self.blks:
            X = blk(X)
        
        return self.head(X[:, 0, :]) #output shape: (batch_size, num_classes)

