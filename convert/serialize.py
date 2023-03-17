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
weights_blood_cls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Blood_Classification_resnet101.pth"
weights_blood_det_th = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Blood_Detection_Yolov5_th.pt"
weights_blood_det_ir = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Blood_Detection_Yolov5_ir.pt"
weights_face_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/pytorch/Face_Detection_Yolov5.pt"

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# load model parameters
score_thresh = 0.20
nms_thresh = 0.45

# blood binary classifer
## define pretrained model
#num_classes = 2
#model_blood_cls = resnet101(weights="IMAGENET1K_V2")
#
## edit last layer
#num_ftrs = model_blood_cls.fc.in_features
#model_blood_cls.fc = nn.Linear(num_ftrs, num_classes)
#
## load weights
#model_bcls.load_state_dict(torch.load(weights_blood_cls, map_location=device))
## load model and weights
model_blood_cls = torch.load(weights_blood_cls, map_location=device)
model_blood_cls.to(device)

x = torch.rand((1,3,480,640), dtype=torch.float32).to(device)
tmodel_blood_cls = torch.jit.trace(model_blood_cls, x)
tmodel_blood_cls.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_cls-{}.pt".format(device).replace(":0",""))

# blood object detection
## thermal camera
model_blood_det_th = YOLO.load_from_yolov5(weights_blood_det_th, score_thresh, nms_thresh)
model_blood_det_th.eval()
model_blood_det_th = model_blood_det_th.to(device)

tmodel_blood_det_th = torch.jit.script(model_blood_det_th)
tmodel_blood_det_th.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_th-{}.pt".format(device).replace(":0",""))

## ir camera
model_blood_det_ir = YOLO.load_from_yolov5(weights_blood_det_ir, score_thresh, nms_thresh)
model_blood_det_ir.eval()
model_blood_det_ir = model_blood_det_ir.to(device)

tmodel_blood_det_ir = torch.jit.script(model_blood_det_ir)
tmodel_blood_det_ir.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_ir-{}.pt".format(device).replace(":0",""))

# face object detection
model_face_det = YOLO.load_from_yolov5(weights_face_det, score_thresh, nms_thresh)
model_face_det.eval()
model_face_det = model_face_det.to(device)

tmodel_face_det = torch.jit.script(model_face_det)
tmodel_face_det.save("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_face_det-{}.pt".format(device).replace(":0",""))
