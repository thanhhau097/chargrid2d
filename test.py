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


def decode_segmap(temp, target, plot=False):
    PALETTE = {
    }
    for name, color in zip(target, all_color):
        PALETTE[name] = color

    r = np.zeros_like(temp).astype(np.uint8)
    g = np.zeros_like(temp).astype(np.uint8)
    b = np.zeros_like(temp).astype(np.uint8)
    for l, name in enumerate(target):
        r[temp == l] = PALETTE[name][0]
        g[temp == l] = PALETTE[name][1]
        b[temp == l] = PALETTE[name][2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3)).astype(np.uint8)
    rgb[:, :, 0] = b
    rgb[:, :, 1] = g
    rgb[:, :, 2] = r
    print(type(rgb))
    # cv2.imwrite(osp.join('./data/debug_segment', img_name + '.png'), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    if plot:
        plt.imshow(rgb)
        plt.show()
    else:
        return rgb


# TODO:
# 1. resize predicted segmentation map to original
# 2. Cut text lines
# 3. Post processing
def resize_segmentation_map(image, segmentation_image):
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


def cut_out_textlines(label_data, segmentation_image):
    results = []
    for item in label_data:
        x, y, w, h = item['box']
        textline_image = segmentation_image[y: y + h, x: x + w]
        cl = get_class_of_segmentation_textline(textline_image, threshold=0.1)
        results.append({
            'text': item['text'],
            'class': cl
        })
    return results


def get_class_of_segmentation_textline(textline_image, threshold=0.2):
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


def get_final_result(data, target):
    results = [''] * len(target)
    for item in data:
        if item['class'] != 0:
            if results[item['class']] == '':
                results[item['class']] = item['text']
            else:
                results[item['class']] = results[item['class']] + ' ' + item['text']

    d = {}
    for i in range(len(target)):
        d[target[i]] = results[i]

    return d


def information_extraction(image, label_data):
    """

    :param image: numpy array image
    :param label_data: [{'text': 'hello', 'box': [x, y, w, h]}]
    :return:
    """
    tensor = mask_generator.generate_test_file(image, label_data)

    img = transforms.functional.to_pil_image(tensor)
    img = np.asarray(img)
    augmented = aug(image=img)
    img = augmented['image'].astype('int16')

    img = torch.from_numpy(img).type(torch.LongTensor)

    img = img.unsqueeze(0)
    img = enc.process(img)
    img = img.unsqueeze(0)
    # img = img.to(self.device)

    output = model.forward(img)
    pred = output[0].data.max(1)[1].cpu().numpy()
    pred = pred.reshape(512, 512)

    pred = resize_segmentation_map(image, pred).astype('int16')
    res = cut_out_textlines(label_data, pred)
    res = get_final_result(res, mask_generator.target)

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_folder', default='./data/sroie/test/', type=str)

    args = parser.parse_args()
    root = args.root_folder

    # ---------- TEST -----------
    mask_generator = MaskGenerator()
    corpus_path = './data/sroie/corpus.json'
    char2idx_path = './data/sroie/char2idx.json'
    target_path = './data/sroie/target.json'
    target2idx_path = './data/sroie/target2idx.json'
    mask_generator.load_config(corpus_path, char2idx_path, target_path, target2idx_path)

    # --------- LOAD MODEL ----------------
    model_path = 'weights/model_epoch_80.pth'
    model = Chargrid2D(len(mask_generator.corpus) + 1, len(mask_generator.target))
    model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # UTILS
    size = 512
    aug = alb.Compose([
        alb.LongestMaxSize(size, interpolation=0),
        alb.PadIfNeeded(size, size, border_mode=cv2.BORDER_CONSTANT),
        # alb.RandomCrop(size, size, p=0.3),
        # alb.Resize(size, size, 0)
    ])

    enc = OneHotEncoder(mask_generator.corpus)

    for img_path in glob.glob('./data/sroie/test/images/*.jpg'):
        image_name = os.path.basename(img_path).replace('.jpg', '')

        # image_name = 'X51006557194'
        image_path = os.path.join(root, 'images', image_name + '.jpg')
        label_path = os.path.join(root, 'labels', image_name + '.json')

        image = cv2.imread(image_path)
        with open(label_path, 'r') as f:
            label_data = json.load(f)

        # --------- MAIN ---------
        print(information_extraction(image, label_data))

