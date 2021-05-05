import argparse
import glob
import os.path as osp

import albumentations as alb
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from chargrid2d.dataloader_utils.onehotencoder import OneHotEncoder
from chargrid2d.dataloader_utils.utils import read_json, make_folder
from chargrid2d.model import Chargrid2D


class PredictProcedure():
    def __init__(self, corpus_path, target_path, model_path, **kwargs):
        self.char2idx = read_json(kwargs['char2idx_path'])
        self.corpus = read_json(corpus_path)
        self.target = read_json(target_path)
        self.model = Chargrid2D(len(self.corpus) + 1, len(self.target))
        if kwargs['device'] == 'cpu':
            self.model.load_state_dict(torch.load(model_path, map_location='cpu'))
        else:
            self.model.load_state_dict(torch.load(model_path))
        self.device = kwargs['device']
        self.model.to(self.device)

        self.size = 512
        self.aug = alb.Compose([
            alb.LongestMaxSize(self.size + 24, interpolation=0),
            alb.PadIfNeeded(self.size + 24, self.size + 24, border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(self.size, self.size, p=0.3),
            alb.Resize(self.size, self.size, 0)
        ])
        self.enc = OneHotEncoder(self.corpus)

        self.all_color = [
            (0, 0, 0),
            (0, 255, 0),
            (0,0,255),
            (0,255,255),
            (255, 0, 255),
            (255, 255, 0),
            (127,255,212),
            (69,139,116),
            (131,139,139),
            (227,207,87),
            (139,125,107),
            (138,43,226),
            (156,102,31),
            (165,42,42),
            (255,64,64),
            (255,97,3),
            (127,255,0),
            (238,18,137),
            (128,128,128),
            (34,139,34),
            (139,105,20),
            (255,105,180),
            (60,179,113),
            (139,0,0),
            (0, 139, 0),
            (0, 0, 139),
        ]
        self.all_color = self.all_color * (len(self.target) // len(self.all_color) + 1)

    def get_char2idx(self, char):
        if char in self.char2idx:
            return self.char2idx[char] + 1
        else:
            return 0

    def process(self, img_path, textline_path):
        img = cv2.imread(img_path)
        textlines = read_json(textline_path)
        doc_h, doc_w = img.shape[0], img.shape[1]
        mask = np.zeros((doc_h, doc_w), dtype='int16')

        for item in textlines:
            w, h  = item['location'][2], item['location'][3]
            char_w, char_h = int(w / (len(item['value'])+1)), int(h)
            cur_x, cur_y = int(item['location'][0]), int(item['location'][1])
              
            for char in item['value']:
                mask[cur_y: cur_y + char_h, cur_x: cur_x + char_w] = self.get_char2idx(char)
                cur_x += char_w
        tensor = torch.from_numpy(mask)
  
        img = transforms.functional.to_pil_image(tensor)
        img = np.asarray(img)
        augmented = self.aug(image=img)
        img = augmented['image'].astype('int16')
        img = torch.from_numpy(img).type(torch.LongTensor)
        img = img.unsqueeze(0)
        img = self.enc.process(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        output = self.model.forward(img)
        # print(output.shape())
        pred = output[0].data.max(1)[1].cpu().numpy().reshape(512, 512)

        return pred
    
    def decode_segmap(self, temp, img_name, plot=False):
        PALETTE = {
        }

        for name, color in zip(self.target, self.all_color):
            PALETTE[name] = color
        
        r = np.zeros_like(temp).astype(np.uint8)
        g = np.zeros_like(temp).astype(np.uint8)
        b = np.zeros_like(temp).astype(np.uint8)
        for l, name in enumerate(self.target):
            r[temp == l] = PALETTE[name][0]
            g[temp == l] = PALETTE[name][1]
            b[temp == l] = PALETTE[name][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3)).astype(np.uint8)
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        print(type(rgb))
        cv2.imwrite(osp.join('./data/debug_segment', img_name + '.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--gpu', default=False, type=bool)

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    corpus_path = './data/corpus.json'  #args.corpus_path
    target_path = './data/target.json'  #args.target_path
    model_path =  './weights/model_epoch_35.pth'  #args.model_path
    make_folder('./data/debug_segment')

    predictor = PredictProcedure(corpus_path, target_path, model_path, **{'device': device, 'char2idx_path': './data/char2idx.json'})


    def squarify(M, val):
        (a, b) = M.shape
        if a > b:
            padding = ((0, 0), ((a - b) // 2, (a - b) // 2))
        else:
            padding = (((b - a) // 2, (b - a) // 2), (0, 0))
        return np.pad(M, padding, mode='constant', constant_values=val)

    for img_path in glob.glob('/home/thanh/projects/axaocr/data/axa_kv/train/images/*.png')[:5]:
        name = osp.basename(img_path).replace('.png', '')
        print(name)
        txtline_path = osp.join('./data/standard_lbl', name + '.json')
        mask_path = osp.join('./data/semantic_gt', name + '.png')
        tensor_path = osp.join('./data/tensor_input', name + '.pt')
        if not osp.exists(txtline_path) or not osp.exists(mask_path):
            continue
        # mask = cv2.imread(mask_path, 0)
        # # padding and resize
        # mask = squarify(mask, 0)
        # # mask = cv2.resize(mask, (512, 512), 0)
        #
        # # augmented = predictor.aug(image=mask)
        # # mask = augmented['image'].astype('int16')
        output = predictor.process(img_path, txtline_path)
        print(output)
        predictor.decode_segmap(output, name, True)

        # predictor.decode_segmap(mask, name, True)
        # # plt.imshow(mask)
        # # plt.show()


        # tensor = torch.load(tensor_path)
        # input_arr = tensor.numpy()
        # input_arr = squarify(input_arr, 0)
        # # plt.imshow(input_arr)
        # # plt.show()
        # print(tensor.max())
