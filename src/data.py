import json
import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
import torch
import cv2
import lightning.pytorch as pl

class ResizeImage(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    """
    
    def __init__(self, MAX_HW=1504):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes = sample['image'], sample['lines_boxes']
        
        W, H = image.size
        if W > self.max_hw or H > self.max_hw:
            scale_factor = float(self.max_hw)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
        else:
            scale_factor = 1
            resized_image = image

        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([y1,x1,y2,x2])

        boxes = torch.Tensor(boxes)
        resized_image = normalize(resized_image)
        # Pad the image and density to wh x wh
        pad = torch.nn.ZeroPad2d((0, self.max_hw - resized_image.shape[2], 0, self.max_hw - resized_image.shape[1]))
        resized_image = pad(resized_image)
        sample = {'image':resized_image,'boxes':boxes}
        return sample


class ResizeImageWithGT(object):
    """
    If either the width or height of an image exceed a specified value, resize the image so that:
        1. The maximum of the new height and new width does not exceed a specified value
        2. The new height and new width are divisible by 8
        3. The aspect ratio is preserved
    No resizing is done if both height and width are smaller than the specified value
    By: Minh Hoai Nguyen (minhhoai@gmail.com)
    Modified by: Viresh
    """
    
    def __init__(self, wh=1504):
        self.wh = wh

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size
        if W > self.wh or H > self.wh:
            scale_factor = float(self.wh)/ max(H, W)
            new_H = 8*int(H*scale_factor/8)
            new_W = 8*int(W*scale_factor/8)
            resized_image = transforms.Resize((new_H, new_W))(image)
            resized_density = cv2.resize(density, (new_W, new_H))
            orig_count = np.sum(density)
            new_count = np.sum(resized_density)

            if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        else:
            scale_factor = 1
            resized_image = image
            resized_density = density
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k*scale_factor) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([y1,x1,y2,x2])

        boxes = torch.Tensor(boxes)
        resized_image = normalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0)
        # Pad the image and density to wh x wh
        pad = torch.nn.ZeroPad2d((0, self.wh - resized_density.shape[2], 0, self.wh - resized_density.shape[1]))
        resized_density = pad(resized_density)
        pad = torch.nn.ZeroPad2d((0, self.wh - resized_image.shape[2], 0, self.wh - resized_image.shape[1]))
        resized_image = pad(resized_image)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample
    
normalize = transforms.Compose([transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def get_transform(split, wh=512):
    if split == 'train':
        custom_transforms = transforms.Compose([ResizeImageWithGT(wh)])
    else:
        custom_transforms = transforms.Compose([ResizeImage(wh)])
    return custom_transforms

def load_fsc147(anno_file, data_split_file, im_dir, gt_dir):
    with open(anno_file) as f:
        annotations = json.load(f)

    with open(data_split_file) as f:
        data_split = json.load(f)

    data = {}
    
    for split in ['train', 'val', 'test']:
        data[split] = []
        im_ids = data_split[split]
        pbar = tqdm.tqdm(total=len(im_ids))
        pbar.set_description('Loading %s' % split)
        for im_id in im_ids:
            anno = annotations[im_id]
            bboxes = anno['box_examples_coordinates']
            dots = np.array(anno['points'])

            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            image = Image.open('{}/{}'.format(im_dir, im_id))
            image.load()
            density_path = gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype('float32')    
            sample = {'image':image,'lines_boxes':rects,'gt_density':density}
            transformed_sample = get_transform(split)(sample)
            data[split].append(transformed_sample)

            pbar.update(1)
        pbar.close()
    return data

class FSC147Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data
        self.num_samples = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return self.num_samples

class FSC147DataModule(pl.LightningDataModule):
    def __init__(self, anno_file, data_split_file, im_dir, gt_dir, batch_size=4, num_workers=4, wh=512):
        super().__init__()
        self.anno_file = anno_file
        self.data_split_file = data_split_file
        self.im_dir = im_dir
        self.gt_dir = gt_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.wh = wh

    def setup(self, stage=None):
        self.data = load_fsc147(self.anno_file, self.data_split_file, self.im_dir, self.gt_dir)
        self.train_dataset = FSC147Dataset(self.data['train'])
        self.val_dataset = FSC147Dataset(self.data['val'])
        self.test_dataset = FSC147Dataset(self.data['test'])

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)