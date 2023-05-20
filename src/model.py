import torch
from torch import nn
# import lightning.pytorch as pl
from torch.nn import functional as F
from torchvision.models import resnet50
class AttentionBlock(nn.Module):
    def __init__(self, n_heads, d_in, d_k, d_v, d_out, dropout=0.1):
        super(AttentionBlock, self).__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.d_out = d_out
        self.dropout = dropout
        self.Wq = nn.Linear(d_in, d_k * n_heads)
        self.Wk = nn.Linear(d_in, d_k * n_heads)
        self.Wv = nn.Linear(d_in, d_v * n_heads)
        self.Wo = nn.Linear(d_v * n_heads, d_out)
        self.layer_norm = nn.LayerNorm(d_out)
        self.dropout = nn.Dropout(dropout)
        
        
    def forward(self, x):
        # x: (batch_size, h * w, d_in)
        batch_size, hw, d_in = x.size()
        d_k, d_v, n_heads = self.d_k, self.d_v, self.n_heads
        # (batch_size, h * w, d_k * n_heads)
        q = self.Wq(x)
        # (batch_size, h * w, d_k * n_heads)
        k = self.Wk(x)
        # (batch_size, h * w, d_v * n_heads)
        v = self.Wv(x)
        # (batch_size, n_heads, h * w, d_k)
        q = q.view(batch_size, hw, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, h * w, d_k)
        k = k.view(batch_size, hw, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, h * w, d_v)
        v = v.view(batch_size, hw, n_heads, d_v).permute(0, 2, 1, 3)
        # (batch_size, n_heads, h * w, h * w)
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / (d_k ** 0.5)
        # (batch_size, n_heads, h * w, h * w)
        attn = F.softmax(attn, dim=-1)
        # (batch_size, n_heads, h * w, d_v)
        out = torch.matmul(attn, v)
        # (batch_size, h * w, n_heads, d_v)
        out = out.permute(0, 2, 1, 3).contiguous()
        # (batch_size, h * w, n_heads * d_v)
        out = out.view(batch_size, hw, n_heads * d_v)
        # (batch_size, h * w, d_out)
        out = self.Wo(out)
        # (batch_size, h * w, d_out)
        out = self.dropout(out)
        # (batch_size, h * w, d_out)
        out = self.layer_norm(out + x)
        return out
        
        
    
class Encoder(nn.Module):
    def __init__(self, d):
        super(Encoder, self).__init__()
        resnet = resnet50()
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
        self.upsample = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0) 
        self.conv1x1 = nn.Conv2d(512, d, kernel_size=1, stride=1, padding=0)
        self.atten = AttentionBlock(n_heads = 8, d_in = d, d_k = 64, d_v = 64, d_out=d)
        

    def forward(self, x):
        out = self.resnet(x)
        out = self.upsample(out)
        out = self.conv1x1(out)
        batch_size, d, h, w = out.size()
        out = out.view(batch_size, d, h * w).permute(0, 2, 1)
        out = self.atten(out).permute(0, 2, 1).view(batch_size, d, h, w)
        return out
    
    
test_img = torch.randn(1, 3, 512, 512)
test_enc = Encoder(256)
print(test_enc(test_img).shape)