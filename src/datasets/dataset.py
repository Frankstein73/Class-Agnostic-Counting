# -*- coding: utf-8 -*-

import torch.utils.data as data
import os
from PIL import Image
import json
from .utils import NormalSample, jpg2id
import torch
import numpy as np
import lightning.pytorch as pl

class FSC147(data.Dataset):
    def __init__(self, root_path, mode):
        super().__init__()
        with open(os.path.join(root_path, 'Train_Test_Val_FSC_147.json')) as f:
            imglist = json.load(f)[mode]
        self.imgids = [jpg2id(imgf) for imgf in imglist]
        with open(os.path.join(root_path, 'fsc147_384x576.json')) as f:
            samples = json.load(f)
        self.samples = {idx: samples[idx] for idx in self.imgids}
        
        self.it2cat = dict()
        with open(os.path.join(root_path, 'ImageClasses_FSC147.txt')) as f:
            catdict = dict()
            for line in f.read().strip().split('\n'):
                a, b = line.split('.jpg')
                a, b = a.strip(), b.strip()
                if b not in catdict:
                    catdict[b] = len(catdict) + 1
                self.it2cat[a] = catdict[b]

        self.root_path = root_path

        self.normalfunc = NormalSample()
        
        self.can_h = 384
        self.can_w = 576
    
    def __getitem__(self, index):
        imgid = self.imgids[index]

        sample = self.getSample(imgid)
        
        return (*sample, imgid)

    def __len__(self):
        return len(self.imgids)

    def getSample(self, imgid):
        sample = self.samples[imgid]
        image = Image.open(os.path.join(self.root_path, sample['imagepath']))
        w, h = image.size
        
        points = torch.tensor(sample['points']).round().long() # N x (w, h)
        boxes = torch.clip(torch.tensor(sample['boxes'][:3]).view(3, 4).round().long(), min=0) # 3 x ((left, top), (right, bottom))
        dotmap = np.zeros((1, h, w), dtype=np.float32)
        points[:, 1] = torch.clip(points[:, 1], min=0, max=h-1)
        points[:, 0] = torch.clip(points[:, 0], min=0, max=w-1)
        dotmap[0, points[:, 1], points[:, 0]] = 1

        image, dotmap = self.normalfunc(image, dotmap)
        for i, box in enumerate(boxes):
            l, t, r, b = box
            b, r = max(t+1, b), max(l+1, r)
            boxes[i] = torch.tensor([l, t, r, b])
        return image, boxes, dotmap
    
    @staticmethod
    def collate_fn(samples):
        images, boxes, dotmaps, imgids = zip(*samples)
        images = torch.stack(images, dim=0)
        boxes = torch.stack(boxes, dim=0).to(dtype=images.dtype)
        dotmaps = torch.stack(dotmaps, dim=0)
        return images, boxes, dotmaps, imgids

class OurFSC147DataModule(pl.LightningDataModule):
    def __init__(self, root_path, batch_size=4, num_workers=4, wh=512):
        super().__init__()
        self.root_path = root_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wh = wh

    def setup(self, stage=None):
        self.train_dataset = FSC147(self.root_path, 'train')
        self.val_dataset = FSC147(self.root_path, 'val')
        self.test_dataset = FSC147(self.root_path, 'test')

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=FSC147.collate_fn)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=FSC147.collate_fn)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, collate_fn=FSC147.collate_fn)
    
if __name__ == '__main__':
    class DeNormalize(object):
        def __init__(self, mean, std):
            self.mean = torch.Tensor(mean)
            self.std = torch.Tensor(std)

        def __call__(self, tensor):
            mean = self.mean.to(tensor.device).view(1, 3, 1, 1)
            std = self.std.to(tensor.device).view(1, 3, 1, 1)
            return tensor * std + mean

    dataset = FSC147('/data/home/elin24/Hercules/coey/gsc147', 'train')
    import tqdm
    denormal = DeNormalize(mean=[0.56347245, 0.50660025, 0.45908741], std=[0.28393339, 0.2804536 , 0.30424776])
    bs_means = []
    for image, boxes, patch, dotmap in tqdm.tqdm(dataset):
        break
