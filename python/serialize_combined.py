import argparse
from os.path import realpath, dirname, join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

from yolort.models import YOLO

# Parse command-line arguments
parser = argparse.ArgumentParser()
# Define arguments for model weights and thresholds
parser.add_argument("--weights_det_th", type=str, default="../weights/pytorch/yolov5s_det_th.pt")
parser.add_argument("--conf_thres",  type=float, default=0.3) 
parser.add_argument("--iou_thres",  type=float, default=0.4) 
args = parser.parse_args()

# Determine if CUDA (GPU) is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# Directory configuration
ROOT = dirname(realpath(__file__))

# Object detection
# Load the pre-trained YOLO model for combined thermal detections (blood/face)
model_det_th = YOLO.load_from_yolov5(args.weights_det_th, args.conf_thres, args.iou_thres)
model_det_th.eval()
model_det_th = model_det_th.to(device)

# Convert the model to TorchScript
tmodel_det_th = torch.jit.script(model_det_th)
tmodel_det_th.save("../weights/torchscript/traced_det_th-{}.pt".format(device).replace(":0",""))