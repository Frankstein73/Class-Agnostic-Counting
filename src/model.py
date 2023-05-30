import copy
from typing import Optional
import torch
from torch import nn

# import lightning.pytorch as pl
from torch.nn import functional as F
from torchvision.models import resnet50, vgg16_bn, VGG16_BN_Weights
from torchvision.ops import roi_align, roi_pool
import torchvision

from our_model import Vgg19FPN


class MHABlock(nn.Module):
    def __init__(self, n_heads, d_in, d_k, d_v):
        super(MHABlock, self).__init__()
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_k = d_k
        assert d_v % n_heads == 0, "d_v must be divisible by n_heads"
        d_v = d_v // n_heads
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
        q = q.reshape(batch_size, len_q, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_k, d_k)
        k = k.reshape(batch_size, len_k, n_heads, d_k).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_k, d_v)
        v = v.reshape(batch_size, len_k, n_heads, d_v).permute(0, 2, 1, 3)
        # (batch_size, n_heads, len_q, len_k)
        attn = torch.matmul(q, k.permute(0, 1, 3, 2)) / (d_k**0.5)
        # (batch_size, n_heads, len_q, len_k)
        attn = F.softmax(attn, dim=-1)
        # (batch_size, n_heads, len_q, d_v)
        out = torch.matmul(attn, v)
        # (batch_size, len_q, n_heads, d_v)
        out = out.permute(0, 2, 1, 3)
        # (batch_size, len_q, d_out)
        out = out.reshape(batch_size, len_q, d_v * n_heads)
        return out


class Encoder(nn.Module):
    def __init__(self, d):
        super(Encoder, self).__init__()
        resnet = resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
        vgg16 = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        modules = list(resnet.children())[:-3]
        vgg_modules = list(vgg16.children())[0]
        del vgg_modules[-1]
        self.resnet = nn.Sequential(*modules)
        self.vgg = vgg_modules
        # self.resnet.requires_grad_(False)
        self.upsample = nn.ConvTranspose2d(
            1024, 512, kernel_size=2, stride=2, padding=0
        )
        self.conv1x1 = nn.Conv2d(512, d, kernel_size=1, stride=1, padding=0)
        self.atten = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d, nhead=8, batch_first=True
            ),
            num_layers=3,
        )

    def forward(self, x):
        # out = self.vgg(x)
        out = self.resnet(x)
        out = self.upsample(out)
        out = self.conv1x1(out)
        batch_size, d, h, w = out.size()
        out = out.reshape(batch_size, d, h * w).permute(0, 2, 1)
        out = self.atten(out).permute(0, 2, 1).reshape(batch_size, d, h, w)
        return out


class IterativeAdaptationModule(nn.Module):
    def __init__(self, n_heads, d_in, d_k, d_v, d_hidden, lenq, iterations):
        super(IterativeAdaptationModule, self).__init__()
        self.iterations = iterations
        self.n_heads = n_heads
        self.d_in = d_in
        self.d_k = d_k
        self.d_v = d_v
        self.mha1 = MHABlock(n_heads, d_in, d_k, d_v)
        self.mha2 = MHABlock(n_heads, d_v, d_k, d_v)
        self.layernorm1 = nn.LayerNorm([lenq, d_in])
        self.layernorm2 = nn.LayerNorm([lenq, d_v])
        self.layernorm3 = nn.LayerNorm([lenq, d_v])
        self.fnn = nn.Sequential(
            nn.Linear(d_v, d_hidden),
            nn.ReLU(),
            nn.Linear(d_hidden, d_v),
        )
        self.dropout = nn.Dropout(0.1)

    def forward(self, Qs, Qa, Fe):
        # Qs: (batch_size, n_boxes*s*s = lenq, d)
        # Qa: (batch_size, n_boxes*s*s, d)
        # Fe: (batch_size, h*w, d)
        Ql = Qs
        Ql_list = []
        for i in range(self.iterations):
            Ql_norm = self.layernorm1(Ql)
            Qla = self.dropout(self.mha1(Ql_norm, Qa)) + Ql
            Qla_norm = self.layernorm2(Qla)
            Qlaf = self.dropout(self.mha2(Qla_norm, Fe)) + Qla
            Qlaf_norm = self.layernorm3(Qlaf)
            Ql = self.dropout(self.fnn(Qlaf_norm)) + Qlaf
            Ql_list.append(Ql)
        # (batch_size, L, n_boxes*s*s, d)
        return Ql_list


class OPE(nn.Module):
    def __init__(self, d, s, iterations, n_boxes=3, roi=roi_pool):
        super(OPE, self).__init__()
        self.s = s
        self.d = d
        self.iterations = iterations
        self.n_boxes = n_boxes
        self.roi = roi
        self.phi = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, d),
            nn.ReLU(),
            nn.Linear(d, s * s * d),
            nn.ReLU(),
        )
        self.iam = IterativeAdaptationModule(
            8, d, d, d, 1024, s * s * n_boxes, iterations
        )

    def forward(
        self, feature: torch.Tensor, boxes: list[torch.Tensor], spatial_scale: float
    ):
        # feature: (batch_size, d, h, w)
        # boxes: list of Tensor(n_boxes, 4)
        n_boxes, _ = boxes[0].size()
        assert n_boxes == self.n_boxes
        batch_size, d, h, w = feature.size()
        # (batch_size, n_boxes*s*s, d)
        Qa = (
            # (batch_size*n, d, s, s)
            self.roi(feature, boxes, [self.s, self.s], spatial_scale=spatial_scale)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, n_boxes * self.s * self.s, d)
        )
        tensor_boxes = torch.stack(boxes, dim=0)
        boxes_hw = torch.stack(
            [
                tensor_boxes[:, :, 3] - tensor_boxes[:, :, 1],
                tensor_boxes[:, :, 2] - tensor_boxes[:, :, 0],
            ],
            dim=2,
        )
        # (batch_size, n_boxes*s*s, d)
        Qs = self.phi(boxes_hw).reshape(batch_size, n_boxes * self.s * self.s, d)
        Fe = feature.permute(0, 2, 3, 1).reshape(batch_size, h * w, d)
        out = self.iam(Qs, Qa, Fe)
        return out


class GenshinOPE(nn.Module):
    def __init__(self, d, s, iterations, n_boxes=3, roi=roi_pool):
        super(GenshinOPE, self).__init__()
        self.s = s
        self.d = d
        self.iterations = iterations
        self.n_boxes = n_boxes
        self.roi = roi

    def forward(
        self, feature: torch.Tensor, boxes: list[torch.Tensor], spatial_scale: float
    ):
        # feature: (batch_size, d, h, w)
        # boxes: list of Tensor(n_boxes, 4)
        n_boxes, _ = boxes[0].size()
        assert n_boxes == self.n_boxes
        batch_size, d, h, w = feature.size()
        # (batch_size, n_boxes*s*s, d)
        Qa = (
            # (batch_size*n, d, s, s)
            self.roi(feature, boxes, [self.s, self.s], spatial_scale=spatial_scale)
            .permute(0, 2, 3, 1)
            .reshape(batch_size, n_boxes * self.s * self.s, d)
        )
        return [Qa]


class Decoder(nn.Module):
    def __init__(self, d, h_out, w_out):
        super(Decoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d, 128, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.LeakyReLU(),
        )
        self.upsample = nn.Upsample(size=(h_out, w_out), mode="bilinear")
        self.conv1x1 = nn.Conv2d(32, 1, 1)
        self.leakyrelu = nn.LeakyReLU()
        self.weights_normal_init(self)

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        x = self.conv1x1(x)
        x = self.leakyrelu(x)
        return x
    
    def weights_normal_init(self, model, dev=0.005):
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
    
class Prototype2ResponseMap(nn.Module):
    def __init__(self, d, s, n_boxes):
        super(Prototype2ResponseMap, self).__init__()
        self.d = d
        self.s = s
        self.n_boxes = n_boxes

    def forward(self, prototype, feature):
        batch_size, d, h, w = feature.size()
        prototype = prototype.reshape(batch_size, self.n_boxes, self.s, self.s, self.d).permute(0, 1, 4, 2, 3).unsqueeze(3)
        similarity_maps = []
        for i in range(batch_size):
            for j in range(self.n_boxes):
                similarity = F.conv2d(
                    feature[i],
                    prototype[i, j],
                    groups=self.d,
                    padding=(self.s - 1) // 2,
                )
                assert similarity.shape == (self.d, h, w)
                similarity_maps.append(similarity)
        similarity_maps = torch.stack(similarity_maps, dim=0).reshape(
            batch_size, self.n_boxes, self.d, h, w
        )
        response_maps = similarity_maps.max(dim=1, keepdim=False)[0]
        return response_maps
        

class LOCA(nn.Module):
    def __init__(self, h_in, w_in, d, s, roi, iterations=3, n_boxes=3):
        super(LOCA, self).__init__()
        self.h_in = h_in
        self.w_in = w_in
        self.d = d
        self.s = s
        self.roi = roi
        self.iterations = iterations
        self.encoder = Encoder(d)
        self.n_boxes = n_boxes
        self.ope = GenshinOPE(d=d, s=s, iterations=iterations, n_boxes=3, roi=roi)
        self.decoder = Decoder(d, self.h_in, self.w_in)
        self.prototype2response = Prototype2ResponseMap(d, s, n_boxes)

    def forward(self, img, boxes: list[torch.Tensor]):
        # img: (batch_size, 3, H, W)
        # feature: (batch_size, d, h, w)
        feature = self.encoder(img)
        batch_size, d, h, w = feature.size()
        spatial_scale = feature.size(2) / img.size(2)
        # prototype: [(batch_size, n_boxes*s*s, d)]
        prototype = self.ope(feature, boxes, spatial_scale)
        if self.training is False:
            prototype = [prototype[-1]]
        # prototype: [(batch_size, n_boxes, d, 1, s, s)]
        # prototype = [
        #     ptype.reshape(batch_size, self.n_boxes, self.s, self.s, self.d)
        #     .permute(0, 1, 4, 2, 3)
        #     .unsqueeze(3)
        #     for ptype in prototype
        # ]
        density_maps_list = []
        for ptype in prototype:
            # similarity_maps = []
            # for i in range(batch_size):
            #     for j in range(self.n_boxes):
            #         similarity = F.conv2d(
            #             feature[i],
            #             ptype[i, j],
            #             groups=self.d,
            #             padding=(self.s - 1) // 2,
            #         )
            #         assert similarity.shape == (self.d, h, w)
            #         similarity_maps.append(similarity)
            # similarity_maps = torch.stack(similarity_maps, dim=0).reshape(
            #     batch_size, self.n_boxes, self.d, h, w
            # )
            # response_maps = similarity_maps.max(dim=1, keepdim=False)[0]
            response_maps = self.prototype2response(ptype, feature)
            density_maps = self.decoder(response_maps)
            density_maps_list.append(density_maps)
        return density_maps_list


class EncoderOnlyTransformer(nn.Module):
    def __init__(self, layers=4, dim=512, norm=None):
        super().__init__()
        d_model = dim
        nhead = 2
        dim_feedforward = 2048
        dropout = 0.1
        activation = "relu"

        self.layers = nn.ModuleList([copy.deepcopy(nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, activation, batch_first=True))
                                     for _ in range(layers)])
        self.norm = norm

    def forward(self, src):
        output = src
        for layer in self.layers:
            output = layer(output)
        if self.norm is not None:
            output = self.norm(output)
        return output

class ConvBlock(nn.Module):
    """
    Normal Conv Block with BN & ReLU
    """

    def __init__(self, cin, cout, k_size=3, d_rate=1, batch_norm=True, res_link=False):
        super().__init__()
        self.res_link = res_link
        if batch_norm:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate),
                nn.BatchNorm2d(cout),
                nn.ReLU(inplace=True),
            )
        else:
            self.body = nn.Sequential(
                nn.Conv2d(cin, cout, k_size, padding=d_rate),
                nn.ReLU(inplace=True),
            )

        self.weights_normal_init(self.body)

    def forward(self, x):
        if self.res_link:
            return x + self.body(x)
        else:
            return self.body(x)
        
    def weights_normal_init(self, model, dev=0.005):
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
        
class GroupOp(nn.Module):
    def forward(self, feature, anchor):
        sim = F.normalize(feature, 2, dim=1) * F.normalize(anchor, 2, dim=1)
        sim = F.relu(sim.sum(dim=1, keepdim=True), inplace=True)
        return sim
    
class VGG16Trans(nn.Module):
    def __init__(self, roi=roi_align):
        super().__init__()
        # self.vgg16bn = vgg16_bn(weights=VGG16_BN_Weights.DEFAULT)
        # in fact, [0] is an `nn.Sequential`
        # self.encoder = list(self.vgg16bn.children())[0]
        # # remove last max pooling layer
        # del self.encoder[-1]

        # meet_max_pool = 0
        # for i, mod in enumerate(iter(self.encoder)):
        #     if isinstance(mod, nn.MaxPool2d):
        #         self.encoder[i] = nn.AvgPool2d(kernel_size=2)
        #         meet_max_pool += 1
        #     if meet_max_pool == 4:
        #         break
        # del self.encoder[i:]
        self.encoder = Vgg19FPN()
        # self.cls_iden = ConvBlock(512,512,1,d_rate=0)
        # self.den_iden = ConvBlock(512,512,1,d_rate=0)
        
        self.tran_decoder = EncoderOnlyTransformer(dim=512)
        self.upsampler = nn.Sequential(
            ConvBlock(512, 256),
            ConvBlock(256, 256),
            nn.PixelShuffle(8),
            nn.Conv2d(4, 1, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.cross = GroupOp()
        self.ope = GenshinOPE(d=512, s=3, iterations=3, n_boxes=3, roi=roi)
        self.prototype2response = Prototype2ResponseMap(512, 3, 3)
    
    def forward(self, x, boxes: list[torch.Tensor]):
        # x: (batch_size, 3, H, W)
        _, _, H, W = x.size()

        
        boxes_tensor = torch.stack(boxes)
        bsize = torch.stack([boxes_tensor[:, :, 2] - boxes_tensor[:, :, 0], boxes_tensor[:, :, 3] - boxes_tensor[:, :, 1]], dim=-1)
        # bs_mean (batch_size, num_boxes)
        bs_mean = bsize.float().mean(dim=1)

        # x: (batch_size, c, h, w)
        clsfea, denfea = self.encoder(x, bs_mean)
        batch_size, c, h, w = denfea.size()

        # prototype: [(batch_size, n_boxes*s*s, 3)]
        prototypes = self.ope(clsfea, boxes, spatial_scale=1./8)
        prototype = prototypes[-1]
        
        # response: (batch_size, 3, h, w)
        response = self.prototype2response(prototype, clsfea)

        # re: (batch_size, c, h, w)
        re = response.mean(dim=1, keepdim=True)
        # re = re.mean(dim=(-1, -2), keepdim=True)
        mask = self.cross(clsfea, re)
        x = denfea * mask

        # x: (batch_size, hw, C)
        x = x.flatten(2).permute(0, 2, 1)
        # x: (batch_size, hw, C)
        x = self.tran_decoder(x)
        # x: (batch_size, C, h, w)
        x = x.permute(0, 2, 1).reshape(batch_size, c, h, w)
        # x: (batch_size, 1, H, W)
        x = self.upsampler(x)
        return x

if __name__ == "__main__":
    img = torch.randn(2, 3, 384, 576)
    boxes = [
        torch.tensor(
            [[0, 0, 56, 56], [0, 0, 28, 28], [0, 0, 14, 14]], dtype=torch.float32
        ),
        torch.tensor(
            [[0, 0, 56, 56], [0, 0, 28, 28], [0, 0, 14, 14]], dtype=torch.float32
        ),
    ]
    loca = VGG16Trans()
    # loca.train()
    # print(loca)
    print(loca(img, boxes).shape)
    # loca.eval()
    # print(len(loca(img, boxes)))
