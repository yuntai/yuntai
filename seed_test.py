import torch
import numpy as np
seed=1111

np.random.seed(seed)
torch.manual_seed(seed)

u0 = np.random.random((100, 100))
v0 = torch.rand(100,100)

np.random.random((100, 100))
torch.rand(100,100)
np.random.random((100, 100))
torch.rand(100,100)

np.random.seed(seed)
torch.manual_seed(seed)

u1 = np.random.random((100, 100))
v1 = torch.rand(100,100)

assert np.allclose(u0, u1)
assert v0.equal(v1)
