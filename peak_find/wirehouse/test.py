import torch

a = torch.rand((1, 1, 20))

print(a)

a = torch.nn.functional.interpolate(a*2, scale_factor=2, mode='linear', align_corners=True)
print(a)