from data import FSC147DataModule
from model import LOCA
from torchvision.ops import roi_align, roi_pool
import torch
import lightning.pytorch as pl



def test_loca(dm, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dm.setup()
    model.to(device)
    for i, batch in enumerate(dm.train_dataloader()):
        if i == 200:
            break
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']
        print(image.shape, boxes.shape, gt_density.shape)

        image = image.to(device)
        boxes = boxes.to(device)
        gt_density = gt_density.to(device)

        output = model(image, boxes)

class LightningLOCA(pl.LightningModule):
    def __init__(self, aux=0.3):
        super().__init__()
        self.model = LOCA(512, 512, 256, 7, roi_pool)
        self.aux = aux

    def forward(self, image, boxes):
        boxes = [box for box in boxes]
        return self.model(image, boxes)

    def training_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        gt_density = batch['gt_density']
        

        output = self(image, boxes)
        loss = torch.nn.functional.mse_loss(output[-1], gt_density)
        for i in range(len(output) - 1):
            loss += self.aux * torch.nn.functional.mse_loss(output[i], gt_density)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image = batch['image']
        boxes = batch['boxes']
        count = batch['count']

        output = self(image, boxes)[-1]

        predicted_count = output.sum(dim=(1, 2, 3))

        # Calculate MAE
        mae = torch.abs(predicted_count - count).mean()
        self.log('val_mae', mae)

        # Calculate RMSE
        rmse = torch.sqrt(torch.nn.functional.mse_loss(predicted_count, count))
        self.log('val_rmse', rmse)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=1e-4, weight_decay=1e-4)

if __name__ == '__main__':
    dm = FSC147DataModule(
        anno_file='../data/annotation_FSC147_384.json',
        data_split_file='../data/Train_Test_Val_FSC_147.json',
        im_dir='../data/images_384_VarV2',
        gt_dir='../data/gt_density_map_adaptive_384_VarV2',
        batch_size=4
    )   
    model = LightningLOCA(aux=0.3)
    # test_loca(dm, model)

    trainer = pl.Trainer(max_epochs=200, devices=[1], logger=pl.loggers.WandbLogger('loca'), precision=16)
    trainer.fit(model, dm)