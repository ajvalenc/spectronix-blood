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

parser = argparse.ArgumentParser()
parser.add_argument("--weights_det_th", type=str, default="/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Detection_Yolov5s_th.pt")
parser.add_argument("--score_thresh",  type=float, default=0.2) 
parser.add_argument("--nms_thresh",  type=float, default=0.4) 
args = parser.parse_args()

# configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Object detection
# Combined thermal detections blood/face
model_det_th = YOLO.load_from_yolov5(args.weights_det_th, args.score_thresh, args.nms_thresh)
model_det_th.eval()
model_det_th = model_det_th.to(device)

tmodel_det_th = torch.jit.script(model_det_th)
tmodel_det_th.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_det_th-{}.pt".format(device).replace(":0",""))

# print blood_det number of parameters
print("Combined Detection Model Parameters: {}".format(sum(p.numel() for p in model_det_th.parameters() if p.requires_grad)))
