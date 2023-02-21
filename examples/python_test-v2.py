import time
from os.path import join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

from yolort.models.yolo import YOLO

# configuration
models_bcls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/models/Blood_Classification_resnet101.pth" #blood cls
models_bdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/models/Blood_Detection_Yolov5.pt" #blood det
models_fdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/models/Face_Detection_Yolov5.pt" #face det

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# load model parameters
score_thresh = 0.20
nms_thresh = 0.45

model_bcls = torch.load(models_bcls, map_location=device)

model_bdet = YOLO.load_from_yolov5(models_bdet, score_thresh, nms_thresh)
model_bdet = model_bdet.eval()
model_bdet = model_bdet.to(device)

model_fdet = YOLO.load_from_yolov5(models_fdet, score_thresh, nms_thresh)
model_fdet = model_fdet.eval()
model_fdet = model_fdet.to(device)

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/BinClass_Test/Pos/T_IM_45.png"
img_raw = cv2.imread(filename) #original
img = cv2.imread(filename) #input

img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
img /= 255.
img = img.unsqueeze(0)

#img = torch.rand((1,3,480,640), dtype=torch.float32).to(device)

# process outputs
with torch.no_grad(): #ensures autograd is off
    # evaluate classifier
    start = time.time()
    out_bcls = model_bcls(img)
    end = time.time()
    print("Blood classification runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_bdet = model_bdet(img)
    end = time.time()
    print("Blood detection runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_fdet = model_fdet(img)
    end = time.time()
    print("Face detection runtime [ms]: ", 1000*(end-start))
