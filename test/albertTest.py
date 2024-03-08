import os
import torch
from transformers import AlbertModel
from transformers import BertModel
import torch.nn as nn
from torch.optim import Adam

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"

if __name__ == '__main__':
    albert = AlbertModel.from_pretrained('./pretrained_models/albert_large')
    albert.to('cuda')
    albert = nn.DataParallel(albert)
    opt = (albert.parameters(), 2e-5)
    x = torch.randint(0,1000,[50,128], dtype=torch.long, device='cuda')

    out = albert(x)
    print(out)
    out = out[0].unsqueeze(0).repeat([10,1,1,1]) # [10, 24, 512, 768]
    out = torch.sum(out)
    out.backward()
    # opt.step()
    print(222)
    # b = torch.ones([10,24,768,512], dtype=torch.float32, device='cuda')
    # c = torch.matmul(out,b)
    # print(c)