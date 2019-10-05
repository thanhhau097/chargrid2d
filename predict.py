import torch
import cv2


def predict(image_path):
    image = cv2.imread(image_path)
