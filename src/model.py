import torch
from torch import nn

# import lightning.pytorch as pl
from torch.nn import functional as F
from torchvision.models import resnet50
from torchvision.ops import roi_align, roi_pool


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
        resnet = resnet50()
        modules = list(resnet.children())[:-3]
        self.resnet = nn.Sequential(*modules)
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
        for i in range(self.iterations):
            Ql_norm = self.layernorm1(Ql)
            Qla = self.dropout(self.mha1(Ql_norm, Qa)) + Ql
            Qla_norm = self.layernorm2(Qla)
            Qlaf = self.dropout(self.mha2(Qla_norm, Fe)) + Qla
            Qlaf_norm = self.layernorm3(Qlaf)
            Ql = self.dropout(self.fnn(Qlaf_norm)) + Qlaf
        return Ql


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
                tensor_boxes[:, :, 2] - tensor_boxes[:, :, 0],
                tensor_boxes[:, :, 3] - tensor_boxes[:, :, 1],
            ],
            dim=2,
        )
        # (batch_size, n_boxes*s*s, d)
        Qs = self.phi(boxes_hw).reshape(batch_size, n_boxes * self.s * self.s, d)
        Fe = feature.permute(0, 2, 3, 1).reshape(batch_size, h * w, d)
        out = self.iam(Qs, Qa, Fe)
        return out


class LOCA(nn.Module):
    def __init__(self, d, s, roi, iterations=3, n_boxes=3):
        super(LOCA, self).__init__()
        self.d = d
        self.s = s
        self.roi = roi
        self.iterations = iterations
        self.encoder = Encoder(d)
        self.n_boxes = n_boxes
        self.ope = OPE(d=d, s=s, iterations=iterations, n_boxes=3, roi=roi)

    def forward(self, img, boxes: list[torch.Tensor]):
        # img: (batch_size, 3, H, W)
        # feature: (batch_size, d, h, w)
        feature = self.encoder(img)
        batch_size, d, h, w = feature.size()
        spatial_scale = feature.size(2) / img.size(2)
        # prototype: (batch_size, n_boxes*s*s, d)
        prototype = self.ope(feature, boxes, spatial_scale)
        # prototype: (batch_size, n_boxes, d, 1, s, s)
        prototype = prototype.reshape(batch_size, self.n_boxes, self.s, self.s, self.d).permute(0, 1, 4, 2, 3).unsqueeze(3)
        similarity_maps = []
        for i in range(batch_size):
            for j in range(self.n_boxes):
                similarity = F.conv2d(feature[i], prototype[i, j], groups=self.d, padding=(self.s - 1)//2)
                assert similarity.shape == (self.d, h, w)
                similarity_maps.append(similarity)
        similarity_maps = torch.stack(similarity_maps, dim=0).reshape(batch_size, self.n_boxes, self.d, h, w)


img = torch.randn(2, 3, 512, 512)
boxes = [
    torch.tensor([[0, 0, 56, 56], [0, 0, 28, 28], [0, 0, 14, 14]], dtype=torch.float32),
    torch.tensor([[0, 0, 56, 56], [0, 0, 28, 28], [0, 0, 14, 14]], dtype=torch.float32),
]
loca = LOCA(256, 7, roi_pool)
loca(img, boxes)
