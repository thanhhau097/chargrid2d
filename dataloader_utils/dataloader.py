import glob
import os
import os.path as osp

import albumentations as alb
import cv2
import numpy as np
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image

from dataloader_utils.utils import read_json
from dataloader_utils.onehotencoder import OneHotEncoder
from dataloader_utils.base_dataloader import BaseDataLoader

class BaseDataset(Dataset):
    def __init__(self, root):
        self.corpus_path = osp.join(root, 'corpus.json')
        self.target_path = osp.join(root, 'target.json')
        self.target2idx_path = osp.join(root, 'target2idx.json')

        self.corpus = read_json(self.corpus_path)
        self.target = read_json(self.target_path)
        self.target2idx = read_json(self.target2idx_path)
        self.enc = OneHotEncoder(self.corpus)


class SegDataset(BaseDataset):
    def __init__(self, root, size=(512, 512), transform=None, type='train'):
        super().__init__(root)
        self.root = osp.join(root, type)
        self.size = size
        self.transform = transform

        # input
        self.lbl_fol = osp.join(self.root, 'labels')
        self.img_fol = osp.join(self.root, 'images')

        self.tensor_fol = osp.join(self.root, 'tensor_input')
        self.semantic_fol = osp.join(self.root, 'semantic_gt')
        self.obj_fol = osp.join(self.root, 'obj_gt')

        self.img_lst = glob.glob(osp.join(self.img_fol,  '*.png'))
        self.tensor_lst = glob.glob(osp.join(self.tensor_fol,  '*.pt'))
        self.semantic_lst = glob.glob(osp.join(self.semantic_fol,  '*.png'))
        self.obj_lst = glob.glob(osp.join(self.obj_fol,  '*.json'))

        self.idx2name = {}
        for idx, path in enumerate(self.tensor_lst):
            name = osp.basename(path).split('.')[0]
            self.idx2name[idx] = name

    def __len__(self):
        return len(self.tensor_lst)

    def __getobjcoor__(self, obj):
        boxes = []
        labels = []

        for line in obj:
            coor = line['box']
            c_x, c_y, w, h = coor
            w += 0.001
            h += 0.001
            boxes.append([c_x, c_y, w, h])
            labels.append(self.target2idx[line['class']])

        return boxes, labels

    def __getitem__(self, idx):
        name = self.idx2name[idx]
        # print(name)
        tensor_path = osp.join(self.tensor_fol, name + '.pt')
        semantic_path = osp.join(self.semantic_fol, name + '.png')
        obj_path = osp.join(self.obj_fol, name + '.json')

        tensor = torch.load(tensor_path)
        semantic = Image.open(semantic_path)
        obj = read_json(obj_path)

        img = transforms.functional.to_pil_image(tensor)
        img = np.asarray(img)
        mask = np.asarray(semantic)
        ori_boxes, label_boxes = self.__getobjcoor__(obj)

        if self.transform:
            augmented = self.transform(image=img, mask=mask, bboxes=ori_boxes, lbl_id=label_boxes)
            img = augmented['image'].astype('int16')
            mask = augmented['mask'].astype('int16')
            boxes = augmented['bboxes']
            lbl_boxes = augmented['lbl_id']

            img, mask = torch.from_numpy(img).type(torch.LongTensor), torch.from_numpy(mask)
            # boxes = np.swapaxes(boxes, 0, 1)  # x_min, y_min, width, height -> we need to return 4 coordinates
            # boxes, lbl_boxes = torch.from_numpy(np.array(boxes)).type(torch.LongTensor), \
            #                     torch.from_numpy(np.array(lbl_boxes))

            img = img.unsqueeze(0)
            img = self.enc.process(img)

        return img, mask, torch.tensor([]), torch.tensor([])  # boxes, lbl_boxes

    def collate_fn(self, batch):
        images = list()
        mask = list()
        boxes = list()
        lbl_boxes = list()

        for b in batch:
            images.append(b[0])
            mask.append(b[1])
            boxes.append(b[2])
            lbl_boxes.append(b[3])

        images = torch.stack(images, dim=0)
        mask = torch.stack(mask, dim=0)

        return images, mask, boxes, lbl_boxes

class ChargridDataloader(BaseDataLoader):
    def __init__(self, root, list_file_name_path, image_size, batch_size, validation_split, num_workers=0, collate_fn=None, shuffle=True):
        """
        Generate batch of items for training and validating

        :param root: data directory
        :param image_size: size of image with training with batch
        :param batch_size: number of images in one batch
        :param shuffle: shuffle after each epoch
        """
        self.root = root
        self.size = image_size
        self.aug = alb.Compose([
            alb.LongestMaxSize(self.size + 24, interpolation=0),
            alb.PadIfNeeded(self.size + 24, self.size + 24, border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(self.size, self.size, p=0.3),
            alb.Resize(self.size, self.size, interpolation=0)
        ], alb.BboxParams(format='coco', label_fields=['lbl_id'], min_area=2.0))

        dataset = SegDataset('./data', list_file_name_path, transform=self.aug)

        kwarg = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': shuffle,
            'validation_split': validation_split,
            'num_workers': num_workers,
            'collate_fn': dataset.collate_fn
        }

        super(ChargridDataloader, self).__init__(**kwarg)


if __name__ == "__main__":
    size = 512
    aug = alb.Compose([
            alb.LongestMaxSize(size + 24, interpolation=0),
            alb.PadIfNeeded(size + 24, size + 24, border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(size, size, p=0.3),
            alb.Resize(size, size, interpolation=0)
        ], alb.BboxParams(format='coco', label_fields=['lbl_id'], min_area=2.0))

    train_dataset = SegDataset('./data', transform=aug, type='train')
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, collate_fn=train_dataset.collate_fn)
    print(len(train_dataloader))

    val_dataset = SegDataset('./data', transform=aug, type='val')
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=True, collate_fn=val_dataset.collate_fn)
    print(len(val_dataloader))

    for idx, sample in enumerate(val_dataloader):
        img, mask, boxes, lbl_boxes = sample
        print(img.size())
        print(mask.size())
        print(lbl_boxes)