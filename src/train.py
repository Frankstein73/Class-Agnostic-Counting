from data import FSC147DataModule
from model import LOCA
from torchvision.ops import roi_align, roi_pool
import torch
import lightning.pytorch as pl

dm = FSC147DataModule(
        anno_file='../data/annotation_FSC147_384.json',
        data_split_file='../data/Train_Test_Val_FSC_147.json',
        im_dir='../data/images_384_VarV2',
        gt_dir='../data/gt_density_map_adaptive_384_VarV2',
        batch_size=4
)

model = LOCA(512, 512, 256, 7, roi_pool)

def test_loca():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm.setup()
    model.to(device)
    for batch in dm.train_dataloader():
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']
        print(image.shape, boxes.shape, gt_density.shape)
        print(boxes)

        image = image.to(device)
        boxes = boxes.to(device)
        gt_density = gt_density.to(device)

        # Convert boxes to list of tensors
        boxes = [box for box in boxes]

        output = model(image, boxes)
        print(output.shape)
        break

class LightningLOCA(pl.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, image, boxes):
        return self.model(image, boxes)

    def training_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']

        output = self(image, boxes)
        loss = torch.nn.functional.mse_loss(output, gt_density)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

if __name__ == '__main__':
    test_loca()