import argparse
import os
import os.path as osp
import operator
import glob

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

from dataloader_utils.utils import make_folder, read_json, write_json, extract_info

class MaskGenerator():
    def __init__(self):
        super().__init__()
        self.__corpus__ = {}
        self.char2idx = {}
        self.target2idx = {}
        self.corpus = []
        self.path_lbls = []
        self.path_std_lbls = []
        self.path_imgs = []
        self.path_lcs = []
        self.target = []

    def __convert_data(self, label_json):
        std_out = []
        regions = {}
        try:
            for fpath, item in label_json['_via_img_metadata'].items():
                regions = item['regions']
        except:
            regions = label_json['attributes']['_via_img_metadata']['regions']
        
        for region in regions:
            if 'x' not in region['shape_attributes']:
                continue
            x = region['shape_attributes']['x']
            y = region['shape_attributes']['y']
            width = region['shape_attributes']['width']
            height = region['shape_attributes']['height']

            try:
                key_type = region['region_attributes']['type']
            except:
                key_type = region['region_attributes']['key_type']

            try:
                fm_key = region['region_attributes']['formal_key'].strip()
            except:
                fm_key = region['region_attributes']['key'].strip()
            
            ocr = region['region_attributes']['label']
            for char in ocr:
                if char not in self.__corpus__:
                    self.__corpus__[char] = 1
                else:
                    self.__corpus__[char] += 1
                    
            std_out.append({
                'value': ocr,
                'formal_key': fm_key,
                'key_type': key_type,
                'location': [x, y, width, height]
            })
        
        return std_out

    def __make_corpus(self):
        x = self.__corpus__
        
        sorted_x = sorted(x.items(), key=operator.itemgetter(1))[::-1]
        for idx, item in enumerate(sorted_x):
            self.corpus.append(item[0])
            self.char2idx[item[0]] = idx
            if len(self.corpus) > 300:
                break
        write_json('data/corpus.json', self.corpus)
        write_json('data/char2idx.json', self.char2idx)

        return self.corpus

    def get_char2idx(self, char):
        if char in self.char2idx:
            return self.char2idx[char] + 1
        else:
            return 0

    def get_corpus(self):
        if len(self.corpus) == 0:
            self.corpus = self.__make_corpus()
        return self.corpus

    def convert_lbl(self, lbl_fol, std_lbl_fol):
        all_files = glob.glob(osp.join(lbl_fol, '*.json'))

        for idx, file_path in enumerate(all_files):
            name = osp.basename(file_path)
            print(name)
            lbl_path = file_path
            img_path = osp.join(img_fol, name.replace('.json', '.png'))
            if not osp.exists(img_path):
                img_path = osp.join(img_fol, name.replace('.json', '.jpg'))  

            print(f'Converting data......File {idx}/Total {len(all_files)}....Progress: {int(idx/len(all_files)*100)}%')
            print(lbl_path)
            print(img_path)
            if not osp.exists(lbl_path) or not osp.exists(img_path):
                print('Image or Label is not exist')
                exit
            
            self.path_lbls.append(lbl_path)
            self.path_imgs.append(img_path)

            lbl_data = self.__convert_data(read_json(lbl_path))
            write_json(osp.join(std_lbl_fol, name), lbl_data)
            self.path_std_lbls.append(osp.join(std_lbl_fol, name))
    
    def __generate_object(self, img_path, lbl_path):
        name = osp.basename(img_path)
        name = name.split('.')[0]
        
        img = cv2.imread(img_path, 0)
        doc_h, doc_w = img.shape
        mask = np.zeros((doc_h, doc_w), dtype='int16')
        gt = np.zeros((doc_h, doc_w), dtype='int16')
        obj = []

        lbl_data = read_json(lbl_path)
        for item in lbl_data:
            w, h  = item['location'][2], item['location'][3]
            char_w, char_h = int(w / (len(item['value'])+1)), int(h) // 2
            cur_x, cur_y = int(item['location'][0]), int(item['location'][1])
            fm_key = item['formal_key']
            k_type = item['key_type']
            if fm_key == 'other':
                cl = 'other'
            else:
                cl = k_type + '_' + fm_key
            for char in item['value']:
                mask[cur_y: cur_y + char_h, cur_x: cur_x + char_w] = self.get_char2idx(char)
                gt[cur_y: cur_y + char_h, cur_x: cur_x + char_w] = self.target2idx[cl]
                cur_x += char_w

            std_item = {
                'text': item['value'],
                'box': [int(item['location'][0]), int(item['location'][1]), int(item['location'][2]), int(item['location'][3])],
                'class': cl
            }
            obj.append(std_item)

        tensor = torch.from_numpy(mask)
        debug_img = np.zeros((doc_h, doc_w*3))
        debug_img[:, :doc_w] = img
        debug_img[:, doc_w: doc_w*2] = mask
        debug_img[:, doc_w*2: doc_w*3] = 255 - gt

        return debug_img, tensor, gt, obj

    def generate_mask(self, out_fol):
        for lbl_path, img_path in zip(self.path_std_lbls, self.path_imgs):
            debug_img, chargrid, semantic_gt, obj_gt = self.__generate_object(img_path, lbl_path)
            name = osp.basename(img_path)
            name = name.split('.')[0]
            print(name)
            cv2.imwrite(osp.join('./data/debug', name + '.png'), debug_img)
            cv2.imwrite(osp.join('./data/semantic_gt', name + '.png'), semantic_gt)
            write_json(osp.join('./data/obj_gt', name + '.json'), obj_gt)
            torch.save(chargrid, osp.join('./data/tensor_input', name + '.pt'))

            # if __debug__:
            #     plt.imshow(debug_img)
            #     plt.show()
    
    def generate_target(self):
        for lbl_path in self.path_std_lbls:
            lbl_data = read_json(lbl_path)
            for item in lbl_data:
                fm_key = item['formal_key']
                k_type = item['key_type']
                if fm_key == 'other':
                    cl = 'other'
                else:
                    cl = k_type + '_' + fm_key

                if cl not in self.target:
                    self.target.append(cl)
        self.target = sorted(self.target)
        for idx, target in enumerate(self.target):
            self.target2idx[target] = idx

        write_json('./data/target.json', self.target)
        write_json('./data/target2idx.json', self.target2idx)

    def online_process(self, path):
        pass

    def process(self, lbl_fol, img_fol, out_fol):
        make_folder(out_fol)
        make_folder('./data/debug')
        make_folder('./data/semantic_gt')
        make_folder('./data/tensor_input')
        make_folder('./data/obj_gt')

        self.convert_lbl(lbl_fol, out_fol)
        self.get_corpus()
        self.generate_target()
        # self.generate_mask('.')
        

if __name__ == "__main__":
    root = 'D:/cinnamon/dataset/kyocera/S3/data/20190924'
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='../data', type=str)

    args = parser.parse_args()
    root = args.root_folder
    root = 'D:/cinnamon/dataset/kyocera/S3/data/20190924'

    lbl_fol = osp.join(root, 'processed_labels')
    img_fol = osp.join(root, 'images')
    out_fol = osp.join('./data', 'standard_lbl')

    mask_generator = MaskGenerator()
    mask_generator.process(lbl_fol, img_fol, out_fol)
    mask_generator.generate_mask('.')