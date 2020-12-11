import torch
import torch.nn.functional as F

seqlen=8
peek=1

peek_mask = (torch.triu(torch.ones(seqlen, seqlen), diagonal=1)==0).type(torch.LongTensor)
nopeek_mask = (torch.triu(torch.ones(seqlen, seqlen), diagonal=0)==0).type(torch.LongTensor)

x = torch.rand(8, 8)
mask = torch.tensor([0,0,0,0,1,1,1,1])
mask = torch.einsum('i, j->ij', mask, mask)
print(mask)

mask_peek = mask * peek_mask
new_peek_mask = -1e20 * torch.ones_like(mask_peek)
new_peek_mask[mask_peek > 0] = 0
score_peek = F.softmax(x + new_peek_mask, dim=-1)*mask_peek
print("score_peek")
print(score_peek)

mask_nopeek = mask * nopeek_mask
new_nopeek_mask = -1e20 * torch.ones_like(mask_nopeek)
new_nopeek_mask[mask_nopeek > 0] = 0
score_nopeek = F.softmax(x + new_nopeek_mask, dim=-1)*mask_nopeek
print("score_nopeek")
print(score_nopeek)
