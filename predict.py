import argparse

import albumentations as alb
import os.path as osp
import cv2
import numpy as np
import glob
import torch
from torchvision import transforms
from matplotlib import pyplot as plt

from models.fscnn import FastSCNN
from dataloader_utils.utils import read_json, make_folder
from dataloader_utils.onehotencoder import OneHotEncoder

all_color = [
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

class PredictProcedure():
    def __init__(self, corpus_path, target_path, model_path, **kwargs):
        self.char2idx = read_json(kwargs['char2idx_path'])
        self.corpus = read_json(corpus_path)
        self.target = read_json(target_path)
        self.model = FastSCNN(len(self.corpus) + 1, len(self.target) + 1, True)
        self.model.load_state_dict(torch.load(model_path))
        self.device = kwargs['device']
        self.model.to(self.device)

        self.size = 512
        self.aug = alb.Compose([
            alb.LongestMaxSize(self.size + 24, interpolation=0),
            alb.PadIfNeeded(self.size + 24, self.size + 24, border_mode=cv2.BORDER_CONSTANT),
            alb.RandomCrop(self.size, self.size, p=0.3),
            alb.Resize(self.size, self.size, interpolation=0)
        ])

        self.enc = OneHotEncoder(self.corpus)

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
            char_w, char_h = int(w / (len(item['value'])+1)), int(h) // 2
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

        outputs = self.model.forward(img)
        pred = torch.argmax(outputs[0], 1)
        pred = pred.cpu().data.numpy()

        return pred[0]

    def draw_legend(self):
        PALETTE = {
        }
        for name, color in zip(self.target, all_color):
            PALETTE[name] = color
        
        img = np.zeros((1500, 800, 3), np.uint8)
        img[:] = (255, 255, 255)

        loc = [25, 25]
        width, height = 50, 25

        for k, v in PALETTE.items():
            color = PALETTE[k]
            overlay = img.copy()
            overlay = cv2.rectangle(
                overlay, (loc[0], loc[1]), (loc[0] + width, loc[1] + height), color, -1)
            img = cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, k, (loc[0] + 50, loc[1] + 17),
                        font, 0.8, color, 1, cv2.LINE_AA)
            # loc[0] += 25*2
            loc[1] += 25 * 2

        cv2.imwrite('legend.png', img)
    
    def decode_segmap(self, temp, img_name, plot=False):
        PALETTE = {
        }
        for name, color in zip(self.target, all_color):
            PALETTE[name] = color
        
        r = np.zeros_like(temp).astype(np.uint8)
        g = np.zeros_like(temp).astype(np.uint8)
        b = np.zeros_like(temp).astype(np.uint8)
        for l, name in enumerate(self.target):
            idx = l + 1
            r[temp == idx] = PALETTE[name][0]
            g[temp == idx] = PALETTE[name][1]
            b[temp == idx] = PALETTE[name][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3)).astype(np.uint8)
        rgb[:, :, 0] = b
        rgb[:, :, 1] = g
        rgb[:, :, 2] = r
        print(type(rgb))
        cv2.imwrite(osp.join('./data/debug_segment', img_name + '.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        if plot:
            plt.imshow(temp)
            # plt.imshow(rgb)
            plt.show()
        else:
            return rgb



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--corpus_path', type=str)
    parser.add_argument('--target_path', type=str)
    parser.add_argument('--gpu', type=bool, default=True)

    args = parser.parse_args()

    if args.gpu and torch.cuda.is_available():
        device = 'cuda:0'
    else:
        device = 'cpu'

    corpus_path = './data/corpus.json' #args.corpus_path
    target_path = './data/target.json' #args.target_path
    model_path =  './weights/best_model.pth' #args.model_path
    make_folder('./data/debug_segment')

    predictor = PredictProcedure(corpus_path, target_path, model_path, **{'device': device, 'char2idx_path': './data/char2idx.json'})
    data_root = './data/train'

    predictor.draw_legend()

    for img_path in glob.glob(osp.join(data_root, 'images/*.png')):
        name = osp.basename(img_path).replace('.png', '')
        print(name)
        txtline_path = osp.join(data_root, 'standard_lbl', name + '.json')
        mask_path = osp.join(data_root, 'semantic_gt', name + '.png')
        if not osp.exists(txtline_path) or not osp.exists(mask_path):
            continue

        mask = cv2.imread(mask_path, 0)
        augmented = predictor.aug(image=mask)
        mask = augmented['image'].astype('int16')
        output = predictor.process(img_path, txtline_path)
        print(output.shape)

        debug = np.zeros((512, 512*2))
        debug[:, :512] = output
        debug[:, 512:] = mask

        predictor.decode_segmap(debug, name, False)

        # predictor.decode_segmap(mask, name, False)
        # plt.imshow(mask)
        # plt.show()
