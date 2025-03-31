'''
learn file
'''
import torch

KV_Caches = {
    "1": torch.rand((1, 3, 4)),
    "2": torch.zeros((1, 3, 4))
}

print(KV_Caches)


KV = torch.rand((2, 3, 5))

for i, mat in enumerate(KV):
    ones = torch.ones((1, 5))
    print(mat.shape)
    print(ones.shape)
    KV[i] = torch.cat([mat, ones], dim=0)
    print('yes')

print(KV)