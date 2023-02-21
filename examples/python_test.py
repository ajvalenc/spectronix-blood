import sys
import argparse
import time
from os.path import join
from copy import deepcopy

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

sys.path.append("../../yolov7")
from models.experimental import attempt_load
from utils.general import scale_coords, non_max_suppression
from utils.plots import plot_one_box

# configuration
parser = argparse.ArgumentParser()
parser.add_argument("--weights_bcls", type=str, default="../models/Blood_Classification_resnet101.pth", help="blood classification model")
parser.add_argument("--weights_bdet", type=str, default="../models/Blood_Detection_Yolov7.pt", help="blood detection model")
parser.add_argument("--weights_fdet", type=str, default="../models/Face_Detection_Yolov7.pt", help="face detection model")
args = parser.parse_args()

feature_extract = False
num_classes = 2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# load model parameters
model_bcls = torch.load(args.weights_bcls, map_location=device)
model_bdet = attempt_load(args.weights_bdet, map_location=device)
model_fdet = attempt_load(args.weights_fdet, map_location=device)

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/BinClass_Test/Pos/T_IM_45.png"
im0 = cv2.imread(filename) #original
img = cv2.imread(filename) #input

img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
img /= 255.
img = img.unsqueeze(0)

#img = torch.rand((1,3,480,640), dtype=torch.float32).to(device)

for i in range(10):
    with torch.no_grad(): #ensures autograd is off
        # evaluate classifier
        start = time.time()
        out_bcls = model_bcls(img)
        end = time.time()
        print("Blood classification runtime: ", end-start)

        start = time.time()
        out_bdet = model_bdet(img)[0]
        end = time.time()
        print("Blood detection runtime: ", end-start)

        start = time.time()
        out_fdet = model_fdet(img)[0]
        end = time.time()
        print("Face detection runtime: ", end-start)

# Process detections
print("Blood Classification Output: ", out_bcls)

im0_cp = deepcopy(im0)
out_bdet = non_max_suppression(out_bdet, 0.2, 0.45, None, True)
for i, det in enumerate(out_bdet):  # detections per image
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, im0_cp, line_thickness=3)

#cv2.imshow("Blood Detection", im0_cp)
#cv2.waitKey()
cv2.imwrite("../output/bdet.jpg", im0_cp)

im0_cp = deepcopy(im0)
out_fdet = non_max_suppression(out_fdet, 0.2, 0.45, None, True)
for i, det in enumerate(out_fdet):  # detections per image
    if len(det):
        # Rescale boxes from img_size to im0 size
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

        # Write results
        for *xyxy, conf, cls in reversed(det):
            plot_one_box(xyxy, im0_cp, line_thickness=3)

#cv2.imshow("Face/Forehead Detection", im0_cp)
#cv2.waitKey()
cv2.imwrite("../output/fdet.jpg", im0_cp)
