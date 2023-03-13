import time
from os.path import join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

from yolort.models.yolo import YOLO

# configuration
weights_blood_cls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Blood_Classification_resnet101.pth"
weights_blood_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Blood_Detection_Yolov5.pt"
weights_face_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Face_Detection_Yolov5.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# load model parameters
score_thresh = 0.20
nms_thresh = 0.45

model_blood_cls = torch.load(weights_blood_cls, map_location=device)

model_blood_det = YOLO.load_from_yolov5(weights_blood_det, score_thresh, nms_thresh)
model_blood_det = model_blood_det.eval()
model_blood_det = model_blood_det.to(device)

model_face_det = YOLO.load_from_yolov5(weights_face_det, score_thresh, nms_thresh)
model_face_det = model_face_det.eval()
model_face_det = model_face_det.to(device)

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/blood/8bit/Pos/T_IM_45.png"
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
    out_blood_cls = model_blood_cls(img)
    end = time.time()
    print("Blood classification runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_bloood_det = model_blood_det(img)
    end = time.time()
    print("Blood detection runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_face_det = model_face_det(img)
    end = time.time()
    print("Face detection runtime [ms]: ", 1000*(end-start))
