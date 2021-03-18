import torch
import torch.nn.functional as F
import numpy as np
x = torch.rand(5)

a = torch.exp(x)/torch.exp(x).sum()
b = torch.softmax(x, dim=0)
assert np.allclose(a.numpy(), b.numpy())

a = x - torch.log(torch.exp(x).sum())
b = torch.log_softmax(x, dim=0)
assert np.allclose(a.numpy(), b.numpy())

C = torch.LongTensor([0])

a = F.cross_entropy(x.unsqueeze(0), C)
b = F.nll_loss(F.log_softmax(x, dim=0).unsqueeze(0), C)
assert np.allclose(a.numpy(), b.numpy())
