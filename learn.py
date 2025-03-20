'''
learn file
'''
import torch

x = torch.rand(3,4)
prod = x.prod(dim=1)
prod_ = x.prod(dim=1, keepdim=True)

print(f'x as {x}')

print(f'prod as {prod}')
print(f'prod_ as {prod_}')