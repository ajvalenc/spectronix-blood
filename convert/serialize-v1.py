'''
create a serialize version of Resnet101 model
'''
from os.path import realpath, dirname, join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

from yolort.models import YOLO

# configuration
weights_bcls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/pytorch/Blood_Classification_resnet101.pth"
weights_bdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/pytorch/Blood_Detection_Yolov5.pt"
weights_fdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/pytorch/Face_Detection_Yolov5.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model parameters
score_thresh = 0.20
nms_thresh = 0.45

# blood binary classifer
model_bcls = torch.load(weights_bcls, map_location=device)
model_bcls.eval()

x = torch.rand((1,3,480,640), dtype=torch.float32).to(device)
traced_model_bcls = torch.jit.trace(model_bcls, x)
traced_model_bcls.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_bcls-{}.pt".format(device).replace(":0",""))

# blood object detection
model_bdet = YOLO.load_from_yolov5(weights_bdet, score_thresh, nms_thresh)
model_bdet.eval()
model_bdet = model_bdet.to(device)

traced_model_bdet = torch.jit.script(model_bdet)
traced_model_bdet.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_bdet-{}.pt".format(device).replace(":0",""))

# face object detection
model_fdet = YOLO.load_from_yolov5(weights_fdet, score_thresh, nms_thresh)
model_fdet.eval()
model_fdet = model_fdet.to(device)

traced_model_fdet = torch.jit.script(model_fdet)
traced_model_fdet.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_fdet-{}.pt".format(device).replace(":0",""))

