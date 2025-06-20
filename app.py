import torch
import sys
import os

# Add yolov5 path
sys.path.append(os.path.join(os.getcwd(), 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
from yolov5.utils.torch_utils import select_device
from yolov5.utils.augmentations import letterbox
import cv2

# Load image


# Load model
device = select_device('')
model = DetectMultiBackend('tick_model.pt', device=device)
