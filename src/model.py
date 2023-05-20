import torch
from torch import nn
# import lightning.pytorch as pl
from torch.nn import functional as F
from torchvision.models import resnet50


class MHABlock(nn.Module):
    def __init__(self, n_heads, d_in, d_k, d_v):
        super(MHABlock, self).__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.wq = nn.Linear(d_in, d_k * n_heads, bias=False)
        self.wk = nn.Linear(d_in, d_k * n_heads, bias=False)
        self.wv = nn.Linear(d_in, d_v * n_heads, bias=False)
        
        
    def forward(self, input_q, input_k):
        # input_q: (batch_size, len_q, d_in)
        # input_k: (batch_size, len_k, d_in)
        batch_size, len_q, d_in = input_q.size()
        _, len_k, _ = input_k.size()
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        # (batch_size, len_q, d_k * n_heads)
        q = self.wq(input_q)
        # (batch_size, len_k, d_k * n_heads)
        k = self.wk(input_k)
        # (batch_size, len_k, d_v * n_heads)
        v = self.wv(input_k)
        # (batch_size, n_heads, len_q, d_k)
        q = q.view(batch_size, len_q, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_k, d_k)
        k = k.view(batch_size, len_k, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_k, d_v)
        v = v.view(batch_size, len_k, n_heads, d_v).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_q, len_k)
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / (d_k ** 0.5)
        # (batch_size, n_heads, len_q, len_k)
        attn = F.softmax(attn, dim=-1)
        # (batch_size, n_heads, len_q, d_v)
        out = torch.matmul(attn, v)
        # (batch_size, len_q, n_heads, d_v)
        out = out.permute(0, 2, 1, 3).contiguous()
        # (batch_size, len_q, d_out)
        out = out.view(batch_size, len_q, d_v * n_heads)
        return out
        

class GlobalSelfAttention(nn.Module):
    def __init__(self, n_heads, d_in, d_k, d_v, d_out, dropout=0.1):
        super(GlobalSelfAttention, self).__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.mha = MHABlock(n_heads, d_in, d_k, d_v)
        self.wo = nn.Linear(d_v * n_heads, d_out)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_out)
        self.shortcut = nn.Linear(d_in, d_out)
        
    def forward(self, x):
        # x: (batch_size, h * w, d_in)
        batch_size, hw, d_in = x.size()
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        # (batch_size, h * w, d_out)
        out = self.mha(x, x)
        # (batch_size, h * w, d_out)
        out = self.wo(out)
        # (batch_size, h * w, d_out)
        out = self.dropout(out)
        # (batch_size, h * w, d_out)
        out = self.layer_norm(out + x) if d_in == self.d_out else self.layer_norm(out + self.shortcut(x))
        return out

    
class Encoder(nn.Module):
    def __init__(self, d):
        super(Encoder, self).__init__()
        resnet = resnet50()
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.upsample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0) 
        self.conv1x1 = nn.Conv2d(512, d, kernel_size=1, stride=1, padding=0)
        self.atten = GlobalSelfAttention(n_heads = 8, d_in = d, d_k = 64, d_v = 64, d_out=d)
        

    def forward(self, x):
        out = self.resnet(x)
        out = self.upsample(out)
        out = self.conv1x1(out)
        batch_size, d, h, w = out.size()
        out = out.view(batch_size, d, h * w).permute(0, 2, 1)
        out = self.atten(out).permute(0, 2, 1).view(batch_size, d, h, w)
        return out
    

