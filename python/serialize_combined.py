'''
create a serialize version of Resnet101 model
'''
import argparse
from os.path import realpath, dirname, join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

from yolort.models import YOLO

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--weights_det_th", type=str, default="../weights/pytorch/Detection_Yolov5s_th.pt")
parser.add_argument("--score_thresh",  type=float, default=0.3) 
parser.add_argument("--nms_thresh",  type=float, default=0.4) 
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# configuration
ROOT = dirname(realpath(__file__))

# Object detection
# Combined thermal detections blood/face
model_det_th = YOLO.load_from_yolov5(args.weights_det_th, args.score_thresh, args.nms_thresh)
model_det_th.eval()
model_det_th = model_det_th.to(device)

tmodel_det_th = torch.jit.script(model_det_th)
tmodel_det_th.save("../weights/torchscript/traced_det_th-{}.pt".format(device).replace(":0",""))

# print blood_det number of parameters
print("Combined Detection Model Parameters: {}".format(sum(p.numel() for p in model_det_th.parameters() if p.requires_grad)))
