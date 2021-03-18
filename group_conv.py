import torch
import numpy as np
import torch.nn as nn

ks = 5
x = torch.rand(7, 6, 256, 256)
n = nn.Conv2d(6, 12, ks)

y0 = (x[0, :, :ks, :ks] * n.weight).sum(dim=(1,2,3)) + n.bias
z0 = n(x)[0,:,0,0]

print(y0.detach())
print(z0.detach())
assert np.allclose(y0.detach(), z0.detach())

n = nn.Conv2d(6, 12, ks, groups=2)

y0 = (x[0, :3, :ks, :ks] * n.weight[:6,...]).sum(dim=(1,2,3)) + n.bias[:6]
y1 = (x[0, 3:, :ks, :ks] * n.weight[6:,...]).sum(dim=(1,2,3)) + n.bias[6:]
y = torch.cat((y0, y1))
z = n(x)[0,:,0,0]
assert np.allclose(y.detach(), z.detach())


