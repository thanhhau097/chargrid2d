import glob
import os.path as osp

import albumentations as alb
import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from dataloader_utils.onehotencoder import OneHotEncoder
from dataloader_utils.utils import read_json


class SegDataset(Dataset):
    def __init__(self, root, size=(1024, 1024), transform=None):
        self.root = root
        self.size = size
        self.transform = transform
        self.img_fol = osp.join(root, 'images')
        self.tensor_fol = osp.join(root, 'tensor_input')
        self.semantic_fol = osp.join(root, 'semantic_gt')
        self.obj_fol = osp.join(root, 'obj_gt')
        self.corpus_path = osp.join(root, 'corpus.json')
        self.target_path = osp.join(root, 'target.json')

        self.img_lst = glob.glob(osp.join(self.img_fol,  '*.png'))
        self.tensor_lst = glob.glob(osp.join(self.tensor_fol,  '*.pt'))
        self.semantic_lst = glob.glob(osp.join(self.semantic_fol,  '*.png'))
        self.obj_lst = glob.glob(osp.join(self.obj_fol,  '*.json'))

        self.idx2name = {}
        for idx, path in enumerate(self.tensor_lst):
            name = osp.basename(path).split('.')[0]
            self.idx2name[idx] = name
        self.corpus = read_json(self.corpus_path)
        self.target = read_json(self.target_path)
        self.enc = OneHotEncoder(self.corpus)

    def __len__(self):
        return len(self.tensor_lst)

    def __transform__(self, tensor):
        pass

    def __getitem__(self, idx):
        name = self.idx2name[idx]

        tensor_path = osp.join(self.tensor_fol, name + '.pt')
        semantic_path = osp.join(self.semantic_fol, name + '.png')
        obj_path = osp.join(self.obj_fol, name + '.json')

        tensor = torch.load(tensor_path)
        semantic = Image.open(semantic_path)
        obj = read_json(obj_path)

        img = transforms.functional.to_pil_image(tensor)
        img = np.asarray(img)
        mask = np.asarray(semantic)

        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image'].astype('int16')
            mask = augmented['mask'].astype('int16')
            img, mask = torch.from_numpy(img).type(torch.LongTensor), torch.from_numpy(mask)
            img = img.unsqueeze(0)
            img = self.enc.process(img)

        return img, mask

    def visualize(self, img, mask):
        imgs = np.asarray(img)
        masks = np.asarray(mask)

        debug = np.zeros((imgs.shape[1]*2, imgs.shape[0]*imgs.shape[2]))

        for idx, img in enumerate(imgs):
            debug[:img.shape[0], idx*img.shape[0]:(idx+1)*img.shape[1]] = img
        for idx, mask in enumerate(masks):
            debug[mask.shape[0]:, idx*mask.shape[0]:(idx+1)*mask.shape[1]] = 255 - mask

        plt.imshow(debug)
        plt.show()

if __name__ == "__main__":
    aug = alb.Compose([
        alb.LongestMaxSize(512),
        alb.PadIfNeeded(512, 512, border_mode=cv2.BORDER_CONSTANT)
    ])
    dataset = SegDataset('./data', transform=aug)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for sample in data_loader:
        img, mask = sample
        print(img.size())
        print(mask.size())
        # dataset.visualize(img, mask)