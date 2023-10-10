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
parser.add_argument("--weights_blood_cls", type=str, default="../weights/pytorch/Blood_Classification_resnet101.pth")
parser.add_argument("--weights_blood_det_th", type=str, default="../weights/pytorch/Blood_Detection_Yolov5_th.pt")
parser.add_argument("--weights_blood_det_ir", type=str, default="../weights/pytorch/Blood_Detection_Yolov5_ir.pt")
parser.add_argument("--weights_face_det", type=str, default="../weights/pytorch/Face_Detection_Yolov5.pt")
parser.add_argument("--score_thresh_blood_th",  type=float, default=0.4) #0.4
parser.add_argument("--nms_thresh_blood_th", type=float, default=0.5)
parser.add_argument("--score_thresh_blood_ir",  type=float, default=0.4) #0.4
parser.add_argument("--nms_thresh_blood_ir", type=float, default=0.5)
parser.add_argument("--score_thresh_face",  type=float, default=0.3) #0.3 
parser.add_argument("--nms_thresh_face",  type=float, default=0.4) 
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# configuration
ROOT = dirname(realpath(__file__))

# blood binary classifer
model_blood_cls = torch.load(args.weights_blood_cls, map_location=device)  # load model and weights
model_blood_cls.to(device)

x = torch.rand((1,3,480,640), dtype=torch.float32).to(device)
tmodel_blood_cls = torch.jit.trace(model_blood_cls, x)
tmodel_blood_cls.save("../weights/torchscript/traced_blood_cls-{}.pt".format(device).replace(":0",""))

# blood object detection
# thermal camera
model_blood_det_th = YOLO.load_from_yolov5(args.weights_blood_det_th, args.score_thresh_blood_th, args.nms_thresh_blood_th)
model_blood_det_th.eval()
model_blood_det_th = model_blood_det_th.to(device)

tmodel_blood_det_th = torch.jit.script(model_blood_det_th)
tmodel_blood_det_th.save("../weights/torchscript/traced_blood_det_th-{}.pt".format(device).replace(":0",""))

# ir camera
model_blood_det_ir = YOLO.load_from_yolov5(args.weights_blood_det_ir, args.score_thresh_blood_ir, args.nms_thresh_blood_ir)
model_blood_det_ir.eval()
model_blood_det_ir = model_blood_det_ir.to(device)

tmodel_blood_det_ir = torch.jit.script(model_blood_det_ir)
tmodel_blood_det_ir.save("../weights/torchscript/traced_blood_det_ir-{}.pt".format(device).replace(":0",""))

# face object detection
model_face_det = YOLO.load_from_yolov5(args.weights_face_det, args.score_thresh_face, args.nms_thresh_face)
model_face_det.eval()
model_face_det = model_face_det.to(device)

tmodel_face_det = torch.jit.script(model_face_det)
tmodel_face_det.save("../weights/torchscript/traced_face_det-{}.pt".format(device).replace(":0",""))
