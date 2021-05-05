import argparse
import glob
import json
import os

import albumentations as alb
import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import transforms

from chargrid2d.dataloader_utils.generate_mask import MaskGenerator
from chargrid2d.dataloader_utils.onehotencoder import OneHotEncoder
from chargrid2d.model import Chargrid2D


class Chargrid2DInformationExtraction:
    def __init__(self, config_folder, weights_path):
        # ---------- TEST -----------
        self.mask_generator = MaskGenerator()
        corpus_path = os.path.join(config_folder, 'corpus.json')
        char2idx_path = os.path.join(config_folder, 'char2idx.json')
        target_path = os.path.join(config_folder, 'target.json')
        target2idx_path = os.path.join(config_folder, 'target2idx.json')
        self.mask_generator.load_config(corpus_path, char2idx_path, target_path, target2idx_path)

        # --------- LOAD MODEL ----------------
        print("Loading information extraction model...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = Chargrid2D(len(self.mask_generator.corpus) + 1, len(self.mask_generator.target))
        self.model = self.model.to(self.device)
        self.model.eval()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))

        # UTILS
        size = 512
        self.aug = alb.Compose([
            alb.LongestMaxSize(size, interpolation=0),
            alb.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
        ])

        self.enc = OneHotEncoder(self.mask_generator.corpus)

    def extract_box_data(self, label_data):
        for item in label_data:
            location = item['location']
            x, y = location[0]
            w = location[2][0] - location[0][0]
            h = location[2][1] - location[0][1]
            item['box'] = [x, y, w, h]

        return label_data

    def process(self, image, label_data):
        """
        :param image: numpy array image
        :param label_data: [{'text': 'hello', 'box': [x, y, w, h]}]
        :return:
        """
        label_data = self.extract_box_data(label_data)
        tensor = self.mask_generator.generate_test_file(image, label_data)

        img = transforms.functional.to_pil_image(tensor)
        img = np.asarray(img)
        augmented = self.aug(image=img)
        img = augmented['image'].astype('int16')

        # plt.imshow(img)
        # plt.show()

        img = torch.from_numpy(img).type(torch.LongTensor)

        img = img.unsqueeze(0)
        img = self.enc.process(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)

        output = self.model.forward(img)
        pred = output[0].data.max(1)[1].cpu().numpy()
        pred = pred.reshape(512, 512)

        pred = self.resize_segmentation_map(image, pred).astype('int16')
        res = self.cut_out_textlines(label_data, pred)
        # res = self.get_final_result(res, self.mask_generator.target)

        return res

    def resize_segmentation_map(self, image, segmentation_image):
        h, w = image.shape[:2]
        # cut out segmentation image
        segmentation_image = segmentation_image.astype('float32')

        if h > w:
            segmentation_image = cv2.resize(segmentation_image, (h, h), interpolation=cv2.INTER_NEAREST)
            result = segmentation_image[:, (h - w) // 2: (h - w) // 2 + w]
        else:
            segmentation_image = cv2.resize(segmentation_image, (w, w), interpolation=cv2.INTER_NEAREST)
            result = segmentation_image[:, (w - h) // 2: (w - h) // 2 + h]

        return result

    def cut_out_textlines(self, label_data, segmentation_image):
        results = []
        for item in label_data:
            x, y, w, h = item['box']
            textline_image = segmentation_image[y: y + h, x: x + w]
            cl = self.get_class_of_segmentation_textline(textline_image, threshold=0.1)
            class_name = self.mask_generator.target[cl]
            if 'key' in class_name:
                formal_key = class_name.split('key_')[-1]
                key_type = 'key'
            elif 'value' in class_name:
                formal_key = class_name.split('value_')[-1]
                key_type = 'value'
            else:
                formal_key = 'other'
                key_type = 'other'
            results.append({
                'text': item['text'],
                'formal_key': formal_key, 
                'class': formal_key,
                'key_type': key_type,
                'location': [(x, y), (x + w, y), (x+w, y+h), (x, y+h)]
            })

        return results

    def get_class_of_segmentation_textline(self, textline_image, threshold=0.2):
        """
        :param textline_image:
        :param threshold: threshold of 0 is 1 - threshold
        :return:
        """
        textline_image = textline_image.astype('int16')
        # cl = mode(textline_image.reshape(-1)).mode[0]
        (values, counts) = np.unique(textline_image, return_counts=True)

        # we need to find the best except 0
        best = 0
        cl = 0

        zero_count = 0
        for value, count in zip(values, counts):
            if value == 0:
                zero_count = count
            else:
                if count > best:
                    cl = value
                    best = count
        try:
            if zero_count / sum(counts) < 1 - threshold:
                return cl
        except:
            pass

        return 0

    # def get_final_result(self, data, target):
    #     results = [''] * len(target)
    #     for item in data:
    #         if item['class'] != 0:
    #             if results[item['class']] == '':
    #                 results[item['class']] = item['text']
    #             else:
    #                 results[item['class']] = results[item['class']] + ' ' + item['text']

    #     d = {}
    #     for i in range(len(target)):
    #         d[target[i]] = results[i]

    #     return d


if __name__ == '__main__':
    def convert_via_to_standard_format(label_data, add_label=True):
        """Standard Input Format is list with each region is a dict that map:
            - "location": Four point coordinate in clockwise position
            - "text": Text data
        Args:
            - label_data: DataPile format label file
            - add_label: Add "key_type" and "type" to kv_input return
        Return:
            - list of tuple class_name and key_type from label
            - list of Standard format for input
        """

        standard_input = []
        list_label = []
        regions = label_data["attributes"]["_via_img_metadata"]["regions"]

        for region in regions:
            shape = region["shape_attributes"]
            if shape["name"] == "polygon":
                xs = shape["all_points_x"]
                ys = shape["all_points_y"]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)

                location = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            elif shape["name"] == "rect":
                x, y = shape["x"], shape["y"]
                w, h = shape["width"], shape["height"]

                location = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
            else:
                warnings.warn(f"Not supported {shape['name']}")

            text = region["region_attributes"]["label"]
            class_name = region["region_attributes"]["formal_key"]
            key_type = region["region_attributes"]["key_type"]

            info = {"location": location, "text": text}

            if add_label:
                info["key_type"] = key_type
                info["type"] = class_name

            standard_input.append(info)

            list_label.append((class_name, key_type))

        return list_label, standard_input


    root = '/home/thanh/projects/axaocr/data/axa_kv/train/'
    config_folder = './data/'
    model_path = 'weights/model_epoch_35.pth'

    model = Chargrid2DInformationExtraction(config_folder, model_path)

    for img_path in glob.glob('/home/thanh/projects/axaocr/data/axa_kv/train/images/*.png')[:2]:
        image_name = os.path.basename(img_path).replace('.png', '')

        # image_name = 'X51006557194'
        image_path = os.path.join(root, 'images', image_name + '.png')
        label_path = os.path.join(root, 'labels', image_name + '.json')

        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            _, label_data = convert_via_to_standard_format(json.load(f))

        # --------- MAIN ---------
        print(model.process(image, label_data))