from typing import Optional
import torch
from torch import nn

# import lightning.pytorch as pl
from torch.nn import functional as F
from torchvision.models import resnet50, vgg19, VGG19_Weights
from torchvision.ops import roi_align, roi_pool
from mmcv.ops import DeformConv2d as dfconv

class SizeInvariantConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        act_fn="relu",
        bn=False,
    ):
        super(SizeInvariantConv2d, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0) if bn else None
        if act_fn == "relu":
            self.relu = nn.ReLU(inplace=True)
        elif act_fn == "prelu":
            self.relu = nn.PReLU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Decoder(nn.Module):
    def __init__(self, indim):
        super(Decoder, self).__init__()

        self.decoder = nn.Sequential(
            SizeInvariantConv2d(indim, 256),
            SizeInvariantConv2d(256, 256),
            nn.PixelShuffle(8),
            SizeInvariantConv2d(4, 1),
        )

        self.weights_normal_init(self.decoder, dev=0.005)

    def forward(self, x):
        return self.decoder(x)

    def weights_normal_init(self, model, dev=0.01):
        if isinstance(model, list):
            for m in model:
                self.weights_normal_init(m, dev)
        else:
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.data.normal_(0.0, dev)
                    if m.bias is not None:
                        m.bias.data.fill_(0.0)
                elif isinstance(m, nn.Linear):
                    m.weight.data.normal_(0.0, dev)



def nc2dc(nconv):
    w = nconv.weight
    # b = nconv.bias
    outc, inc, _, _ = w.shape
    dconv = dfconv(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1)
    dconv.weight = nn.Parameter(w.clone())
    return dconv, inc

class SizeBlock(nn.Module):
    def __init__(self, conv):
        super(SizeBlock, self).__init__()
        self.conv, inc = nc2dc(conv)
        self.glob = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 32)
        )
        self.local = nn.Sequential(
            nn.Conv2d(inc, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1)
        )

        self.fuse = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3 * 3 * 2, 3, padding=1)
        )
        self.relu = nn.ReLU()


    def forward(self, x, bsize):
        b, c, h, w = x.shape
        g_offset = self.glob(bsize)
        g_offset = g_offset.reshape(b, -1, 1, 1).repeat(1, 1, h, w).contiguous()
        l_offset = self.local(x)
        offset = self.fuse(torch.cat((g_offset, l_offset), dim=1))
        fea = self.conv(x, offset)
        return  self.relu(fea)

class NormBlock(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.conv = nn.Sequential(conv, nn.ReLU(inplace=True))
    
    def forward(self, x, size=None):
        return self.conv(x)

class Vgg19FPN(nn.Module):
    def __init__(self, pretrained=True):
        super(Vgg19FPN, self).__init__()
        vgg = vgg19(weights=VGG19_Weights.DEFAULT if pretrained else None)
        mods = list(vgg.features.children())[:28]

        self.modlist = nn.ModuleList()
        self.selist = nn.ModuleList()
        last = 0
        for i, mod in enumerate(mods):
            if 'MaxPool2d' in str(mod):
                self.modlist.append(nn.Sequential(*mods[last:i-2]))
                self.selist.append(SizeBlock(mods[i-2]))
                last = i
        
        self.clsfc = nn.Conv2d(512, 512, 1, padding=0)
        self.denfc = nn.Conv2d(512, 512, 1, padding=0)
        
    def forward(self, x, msize):
        for mod, sem in zip(self.modlist, self.selist):
            x = mod(x) # b c h w
            x = sem(x, msize)
        fea = x 
        return self.clsfc(fea), self.denfc(fea)
    
    def outdim(self):
        return 512

class GroupOp(nn.Module):
    def forward(self, feature, anchor):
        sim = F.normalize(feature, 2, dim=1) * F.normalize(anchor, 2, dim=1)
        sim = F.relu(sim.sum(dim=1, keepdim=True), inplace=True)
        return sim
    
class SPDCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Vgg19FPN()
        feadim = self.encoder.outdim()
        self.roi = 16
        self.roialign = roi_align
        self.cross = GroupOp()
        self.decoder = Decoder(feadim)

    def forward(self, image, boxes):
        bsize = torch.stack((boxes[:, 4] - boxes[:, 2], boxes[:, 3] - boxes[:, 1]), dim=-1)
        bs_mean = bsize.reshape(-1, 3, 2).float().mean(dim=1)

        b, _, imh, imw = image.shape
        clsfea, denfea = self.encoder(image, bs_mean)

        patches = self.roialign(clsfea, boxes.to(dtype=clsfea.dtype), output_size=self.roi, spatial_scale=1./8)
        anchors_patchs = patches.reshape(b, 3, -1, self.roi, self.roi).mean(dim=1)
        anchor_cls = anchors_patchs.mean(dim=(-1, -2), keepdim=True)
        
        mask = self.cross(clsfea, anchor_cls)
        denmap = self.decoder(denfea * mask)
        
        return denmap
    