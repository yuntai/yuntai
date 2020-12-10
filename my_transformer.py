import torch.nn as nn
import torch.nn.functional as F
import torch
import time
import numpy as np

def init_weight(weight):
    #nn.init.uniform_(weight, -0.1, 0.1)
    nn.init.constant_(weight, 0.2)

def init_bias(bias):
    nn.init.constant_(bias, 0.1)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1 or classname.find('Conv1d') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, 'bias') and m.bias is not None:
            init_bias(m.bias)

class Transformer(nn.Module):
    def __init__(self, d_model, n_head, n_layers):
        super().__init__()
        self.blocks = nn.Sequential(
            *[EncoderBlock(d_model, n_head) for _ in range(n_layers)])

    def forward(self, input_):
        return self.blocks(input_)

class SelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.proj = nn.Linear(self.d_model, 3 * self.d_model)
        self.sqrtd = np.sqrt(self.d_head)
        self.out = nn.Linear(self.d_model, self.d_model)
        self.attn_dropout = nn.Dropout(0.0)
        self.resid_dropout = nn.Dropout(0.0)

    def forward(self, input_):
        bsz, n_elem, dim = input_.size()
        assert self.d_model == dim, f"input dim({dim}) != model dim({self.d_model})"
        V, Q, K = self.proj(input_).view(bsz, n_elem, self.n_head, 3*self.d_head).chunk(3, dim=3)
        W = torch.einsum('bnhi, bmhi->bhnm', Q, K) / self.sqrtd
        W = F.softmax(W, dim=3)
        W = self.attn_dropout(W)
        O = torch.einsum('bhnm,bmhv->bnhv', W, V).reshape(bsz, n_elem, dim)
        O = self.resid_dropout(self.out(O)) # BACI use leaky_relu before self.out
        return O

class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        d_inner = d_model * 4
        self.ln1 = nn.LayerNorm(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model, d_model)
        self.attn = SelfAttention(d_model, n_head)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_inner),
            nn.GELU(), # GELU
            nn.Linear(d_inner, d_model)
            nn.Dropout(0.2)
        )

    def forward(self, x):
        # https://github.com/huggingface/transformers/pull/3929
        # pre-norm residual
        x += self.attn(self.ln1(x))
        x += self.ff(self.ln2(x))
        return x
        
def run():
    input_dim = 256
    bsz = 32
    seq_len = 55
    dim = 256
    h = 8
    dev = 'cuda'

    input_ = torch.rand(bsz, seq_len, input_dim).to(dev)
    tr = Transformer(dim, 8, 6).to(dev)
    tr.apply(weights_init)

    out3 = tr(input_)

    _st = time.time()
    out3 = tr(input_)
    print(time.time() - _st)
    print(f"{out3.shape=}")

if __name__ == '__main__':
    run()



