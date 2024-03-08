from regex import B
import torch

x = torch.tensor([[2,3],[4,5]])
mask = torch.tensor([0,1])


print(x.masked_select(mask.unsqueeze(-1)==1))