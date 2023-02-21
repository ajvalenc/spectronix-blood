import sys
import argparse
import time
from copy import deepcopy

import cv2
import torch
import torch.nn as nn
from torchvision.io import read_image 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

sys.path.append("../../yolov7")
from utils.datasets import LoadImages
from pathlib import Path
from utils.general import scale_coords, non_max_suppression
from utils.plots import plot_one_box

# configuration
parser = argparse.ArgumentParser()
parser.add_argument("--weights_bcls", type=str, default="../weights/traced_bcls-cpu.pt", help="blood classification model")
parser.add_argument("--weights_bdet", type=str, default="../weights/traced_bdet-cpu.pt", help="blood detection model")
parser.add_argument("--weights_fdet", type=str, default="../weights/traced_fdet-cpu.pt", help="face detection model")
args = parser.parse_args()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/BinClass_Test/Pos/T_IM_45.png"
im0 = cv2.imread(filename) #original
img = cv2.imread(filename) #input
#img = cv2.resize(img, (480,640), interpolation=cv2.INTER_LINEAR)
img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
img /= 255.
img = img.unsqueeze(0)

# load serialized blood classification model
print("Loading binary model...")
model_bcls = torch.jit.load(args.weights_bcls)
print("Binary model loaded successfully")

# load serialized blood detection model
print("Loading detection model...")
model_bdet = torch.jit.load(args.weights_bdet)
print("Detection model loaded successfully")

# load serialized face detection model
print("Loading detection model...")
model_fdet = torch.jit.load(args.weights_fdet)
print("Detection model loaded successfully")

# random input data
#img = torch.rand((3,480,640)).to(device)

for i in range(1):
    with torch.no_grad(): #ensures autograd is off
        start = time.time()
        out_bcls = model_bcls(img)
        end = time.time()
        print("Blood classification runtime: ", end-start)

        start = time.time()
        out_bdet = model_bdet(img)
        end = time.time()
        print("Blood detection runtime: ", end-start)

        start = time.time()
        out_fdet = model_fdet(img)
        end = time.time()
        print("Face detection runtime: ", end-start)
        
# Process detections
print("Blood Classification Output: ", out_bcls)

im0_cp = deepcopy(im0)
out_bdet = non_max_suppression(out_bdet[0], 0.2, 0.45, None)
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
out_fdet = non_max_suppression(out_fdet[0], 0.2, 0.45, None)
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
