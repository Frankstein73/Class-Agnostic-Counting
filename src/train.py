from data import FSC147DataModule
from datasets.dataset import OurFSC147DataModule
from loss.genloss import GeneralizedLoss
from model import LOCA, VGG16Trans
from torchvision.ops import roi_align, roi_pool
import torch
import lightning.pytorch as pl
from torch.nn import functional as F
import torchmetrics
from lightning.pytorch import seed_everything

import ssim

def save_figs(images, boxes_es, gt_densities, outputs, prefix, save_path="test"):
    import matplotlib.pyplot as plt
    import os
    batch_size = images.shape[0]
    for j in range(batch_size):
        output_len = len(outputs)
        fig = plt.figure(clear=True)
        ax = fig.add_subplot(3, output_len, 1)
        ax.imshow(images[j].permute(1, 2, 0).cpu().numpy())
        for box in boxes_es[j]:
            bx1, by1, bx2, by2 = tuple(box.tolist())
            ax.add_patch(plt.Rectangle((bx1, by1), bx2 - bx1, by2 - by1, fill=False, edgecolor='red', linewidth=2))

        ax = fig.add_subplot(3, output_len, output_len+1)
        if gt_densities is not None:
            ax.imshow(gt_densities[j].detach().permute(1,2,0).cpu().numpy())
        for k in range(output_len):
            ax = fig.add_subplot(3, output_len, output_len*2 + k + 1)
            ax.imshow(outputs[k][j].detach().permute(1,2,0).cpu().numpy())

        # make dir test/ if not exist
        os.makedirs(save_path, exist_ok=True)
        fig.savefig(f'{save_path}/{prefix}_{j}.png')
        plt.close(fig)

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

        save_figs(image, boxes, gt_density, outputs, f"train_{i}", save_path="test_loca")

class LightningVGG16(pl.LightningModule):
    def __init__(self, val_save_figs=True):
        super().__init__()
        self.model = VGG16Trans()
        self.mae = torchmetrics.MeanAbsoluteError()
        self.mse = torchmetrics.MeanSquaredError()
        self.ssim_loss = ssim.SSIM(window_size=11)
        self.gen_loss = GeneralizedLoss()
        self.val_save_figs = val_save_figs

    def forward(self, image, boxes):
        boxes = [box for box in boxes]
        return self.model(image, boxes)

    def training_step(self, batch, batch_idx):
        image, boxes, dotmaps, _ = batch

        output = self.forward(image, boxes)

        # ssim_loss = 1 - self.ssim_loss(output, small_gt_density)
        # loss = F.mse_loss(output, small_gt_density, reduction="mean") + ssim_loss * 0.001
        

        # bsize (batch_size, num_boxes, 2)
        bsize = torch.stack([boxes[:, :, 2] - boxes[:, :, 0], boxes[:, :, 3] - boxes[:, :, 1]], dim=-1)
        # bs_mean (batch_size, num_boxes)
        bs_mean = bsize.float().mean(dim=1)

        loss = self.gen_loss(output, dotmaps, box_size=bs_mean)
        self.log('train_loss', loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, boxes, dotmaps, _ = batch

        count = dotmaps.sum(dim=(1,2,3))
        output = self.forward(image, boxes)
        predicted_count = output.sum(dim=(1, 2, 3))

        self.mae.update(predicted_count, count)
        self.mse.update(predicted_count, count)
        if self.val_save_figs and batch_idx < 10:
            save_figs(image, boxes, None, [output], f"val_{batch_idx}", "val")

    def on_validation_epoch_end(self):
        self.log('val_mae', self.mae.compute(), prog_bar=True)
        self.log('val_rmse', torch.sqrt(self.mse.compute()), prog_bar=True)
        self.mae.reset()
        self.mse.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=4e-5, weight_decay=1e-4)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5),
                # "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2,factor=0.5),
                "interval": "epoch",
                "monitor": "val_mae",
            }
        }

class LightningLOCA(pl.LightningModule):
    def __init__(self, aux=0.3, val_save_figs=False):
        super().__init__()
        self.model = LOCA(512, 512, 256, 3, roi_align, iterations=1)
        self.aux = aux
        self.val_save_figs = val_save_figs

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

        outputs = self(image, boxes)
        output = outputs[-1]

        predicted_count = output.sum(dim=(1, 2, 3))

        if self.val_save_figs and batch_idx < 10:
            save_figs(image, boxes, None, outputs, f"val_{batch_idx}", "val")
        self.mae.update(predicted_count, count)
        self.mse.update(predicted_count, count)

    def on_validation_epoch_end(self):
        self.log('val_mae', self.mae.compute(), prog_bar=True)
        self.log('val_rmse', torch.sqrt(self.mse.compute()), prog_bar=True)
        self.mae.reset()
        self.mse.reset()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=20, weight_decay=1e-4)

if __name__ == '__main__':
    seed_everything(42)
    
    dm = OurFSC147DataModule(
        root_path="../data",
        batch_size=4
    )   
    # model = LightningLOCA(aux=0, val_save_figs=True)
    model = LightningVGG16(val_save_figs=True)
    # model = LightningVGG16.load_from_checkpoint("v1.ckpt")
    # test_loca(dm, model)
    
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
    trainer = pl.Trainer(max_epochs=200, devices=[0], logger=pl.loggers.WandbLogger('vggtrans', project='baseline-v2'), precision=16, gradient_clip_val=0.1)
    trainer.fit(model, dm)
