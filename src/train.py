from data import FSC147DataModule
from model import LOCA, VGG16Trans
from torchvision.ops import roi_align, roi_pool
import torch
import lightning.pytorch as pl
from torch.nn import functional as F
import torchmetrics



def test_loca(dm, model):
    import matplotlib.pyplot as plt
    import os
    device = torch.device('cpu')
    dm.setup()
    model.to(device)
    for i, batch in enumerate(dm.train_dataloader()):
        if i == 10:
            break
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']

        image = image.to(device)
        boxes = boxes.to(device)
        gt_density = gt_density.to(device)

        outputs = model(image, boxes)

        batch_size = image.shape[0]
        for j in range(batch_size):
            output_len = len(outputs)
            fig = plt.figure(clear=True)
            ax = fig.add_subplot(3, output_len, 1)
            ax.imshow(image[j].permute(1, 2, 0).numpy())
            for box in boxes[j]:
                bx1, by1, bx2, by2 = tuple(box.tolist())
                ax.add_patch(plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, edgecolor='red', linewidth=2))

            ax = fig.add_subplot(3, output_len, output_len+1)
            ax.imshow(gt_density[j].detach().permute(1,2,0).numpy())
            for k in range(output_len):
                ax = fig.add_subplot(3, output_len, output_len*2 + k + 1)
                ax.imshow(outputs[k][j].detach().permute(1,2,0).numpy())

            # make dir test/ if not exist
            os.makedirs('test', exist_ok=True)
            fig.savefig(f'test/{i}_{j}.png')

class LightningVGG16(pl.LightningModule):
    def __init__(self, up_scale=8):
        super().__init__()
        self.model = VGG16Trans(up_scale)
        self.up_scale = up_scale
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, image):
        return self.model(image)

    def training_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']
        count = batch['count']

        output = self.forward(image)

        small_gt_density = F.avg_pool2d(gt_density, self.up_scale) * (self.up_scale ** 2)

        loss = F.mse_loss(output, small_gt_density, reduction="sum")
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        count = batch['count']

        output = self.forward(image)

        predicted_count = output.sum(dim=(1, 2, 3))

        self.mae.update(predicted_count, count)
        self.mse.update(predicted_count, count)

    def on_validation_epoch_end(self):
        self.log('val_mae', self.mae.compute(), prog_bar=True)
        self.log('val_rmse', torch.sqrt(self.mse.compute()), prog_bar=True)
        self.mae.reset()
        self.mse.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-7, weight_decay=1e-3)

class LightningLOCA(pl.LightningModule):
    def __init__(self, aux=0.3):
        super().__init__()
        self.model = LOCA(512, 512, 256, 3, roi_align, iterations=1)
        self.aux = aux

        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()

    def forward(self, image, boxes):
        boxes = [box for box in boxes]
        return self.model(image, boxes)

    def training_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']
        count = batch['count']

        output = self(image, boxes)

        m = count.sum()
        loss = (output[-1] - gt_density).norm(p=2) ** 2 / m
        for i in range(len(output) - 1):
            loss += self.aux * (output[i] - gt_density).norm(p=2) ** 2 / m
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        count = batch['count']

        output = self(image, boxes)[-1]

        predicted_count = output.sum(dim=(1, 2, 3))

        self.mae.update(predicted_count, count)
        self.mse.update(predicted_count, count)

    def on_validation_epoch_end(self):
        self.log('val_mae', self.mae.compute(), prog_bar=True)
        self.log('val_rmse', torch.sqrt(self.mse.compute()), prog_bar=True)
        self.mae.reset()
        self.mse.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-7, weight_decay=1e-3)

if __name__ == '__main__':
    dm = FSC147DataModule(
        anno_file='../data/annotation_FSC147_384.json',
        data_split_file='../data/Train_Test_Val_FSC_147.json',
        im_dir='../data/images_384_VarV2',
        gt_dir='../data/gt_density_map_adaptive_384_VarV2',
        batch_size=8
    )   
    model = LightningLOCA(aux=0.3)
    # model = LightningLOCA.load_from_checkpoint("v1.ckpt")
    test_loca(dm, model)
    
    # dm.setup()
    # demo = next(iter(dm.train_dataloader()))
    # image, boxes, gt_density, count = demo['image'], demo['boxes'], demo['gt_density'], demo['count']
    # # display image with boxes
    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # ax = fig.add_subplot(121)
    # ax.imshow(image[0].permute(1, 2, 0).numpy())
    # for box in boxes[0]:
    #     bx1, by1, bx2, by2 = tuple(box.tolist())
    #     ax.add_patch(plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, edgecolor='red', linewidth=2))

    # ax = fig.add_subplot(122)
    # ax.imshow(gt_density[0].permute(1,2,0).numpy())
    # plt.show()
    trainer = pl.Trainer(max_epochs=200, devices=[1], logger=pl.loggers.WandbLogger('loca'), precision=16, gradient_clip_val=0.1, accumulate_grad_batches=2)
    trainer.fit(model, dm)
