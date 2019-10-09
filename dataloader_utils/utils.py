import os
import os.path as osp
import json
import shutil


def make_folder(path):
    if osp.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)


def read_json(path):
    with open(path, 'r', encoding='utf-8-sig') as fi:
        data = json.load(fi)

    return data


def write_json(path, data):
    with open(path, 'w', encoding='utf-8-sig') as fo:
        json.dump(data, fo, ensure_ascii=False, indent=4)


def overlap_element(pred, ele):
    xA = max(pred[0], ele[0])
    yA = max(pred[1], ele[1])
    xB = min(pred[2], ele[2])
    yB = min(pred[3], ele[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    pred_locArea = (pred[2] - pred[0] + 1) * (pred[3] - pred[1] + 1)
    gt_locArea = (ele[2] - ele[0] + 1) * (ele[3] - ele[1] + 1)

    iou = interArea / float(pred_locArea + gt_locArea - interArea)

    return iou


def check_align(pred, gt):
    return abs(pred[1] - gt[1]) <= 20


def _overlap(pred, list_loc):
    ious = []
    idxs = []
    for idx, ele in enumerate(list_loc):
        iou = overlap_element(pred, ele)
        if iou > 0.0001 and check_align(pred, ele):
            ious.append(iou)
            idxs.append(idx)
    return ious, idxs


def overlap_ratio(pred, target):
    ratio, idx = _overlap(pred, target)
    return ratio, idx


def extract_info(label_json):
    regions = {}
    try:
        for fpath, item in label_json['_via_img_metadata'].items():
            regions = item['regions']
    except:
        regions = label_json['attributes']['_via_img_metadata']['regions']

    coors = []
    texts = []
    fm_keys = []
    key_types = []

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

        if key_type not in ['key', 'value', 'common_key', 'master']:
            continue

        try:
            fm_keys.append(region['region_attributes']['formal_key'].strip())
        except:
            fm_keys.append(region['region_attributes']['key'].strip())

        coors.append([x, y, x + width, y + height])
        texts.append(region['region_attributes']['label'])
        key_types.append(key_type)
        
    return coors, texts, fm_keys, key_types
