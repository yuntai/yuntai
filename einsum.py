import torch
bsz = 512
seqlen = 200
nhead = 8
dim = 256

q = torch.rand(bsz,seqlen,nhead,dim).to('cuda')
k = torch.rand(bsz,seqlen,nhead,dim).to('cuda')
v = torch.rand(bsz,seqlen,nhead,dim).to('cuda')

def run0(q,k,v):
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    scores = torch.matmul(q, k.transpose(-2, -1))
    output = torch.matmul(scores, v)
    return output.transpose(1,2).contiguous()
    #return output

def run1(q,k,v):
    scores = torch.einsum('bnhd,bmhd->bhnm', q, k)
    return torch.einsum('bhnm,bmhd->bnhd', scores, v).contiguous()

import time

_st = time.time()
output0 = run0(q,k,v)
print(time.time() - _st)

_st = time.time()
output1 = run1(q,k,v)
print(time.time() - _st)

print(output0.shape)
print(output1.shape)
print(output0.is_contiguous())
print(output1.is_contiguous())

assert torch.all(output0==output1)

