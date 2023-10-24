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
parser.add_argument("--weights_blood_cls", type=str, default="../weights/pytorch/resnet101_blood_cls.pth")
parser.add_argument("--weights_blood_det_th", type=str, default="../weights/pytorch/yolov5_blood_det_th.pt")
parser.add_argument("--weights_blood_det_ir", type=str, default="../weights/pytorch/yolov5_blood_det_ir.pt")
parser.add_argument("--weights_face_det", type=str, default="../weights/pytorch/yolov5_face_det.pt")
parser.add_argument("--conf_thres_blood_th",  type=float, default=0.4) #0.4
parser.add_argument("--iou_thres_blood_th", type=float, default=0.5)
parser.add_argument("--conf_thres_blood_ir",  type=float, default=0.4) #0.4
parser.add_argument("--iou_thres_blood_ir", type=float, default=0.5)
parser.add_argument("--conf_thres_face",  type=float, default=0.3) #0.3 
parser.add_argument("--iou_thres_face",  type=float, default=0.4) 
args = parser.parse_args()

# Determine if CUDA (GPU) is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# Directory configuration
ROOT = dirname(realpath(__file__))

# Blood binary classifer
# Load the pre-trained blood classifier and convert it to TorchScript
model_blood_cls = torch.load(args.weights_blood_cls, map_location=device)  # Load model and weights
model_blood_cls.to(device)

x = torch.rand((1,3,480,640), dtype=torch.float32).to(device)
tmodel_blood_cls = torch.jit.trace(model_blood_cls, x)
tmodel_blood_cls.save("../weights/torchscript/traced_blood_cls-{}.pt".format(device).replace(":0",""))

# Blood object detection
# Thermal camera
# Load the pre-trained YOLO model for blood detection in thermal images and convert it to TorchScript
model_blood_det_th = YOLO.load_from_yolov5(args.weights_blood_det_th, args.conf_thres_blood_th, args.iou_thres_blood_th)
model_blood_det_th.eval()
model_blood_det_th = model_blood_det_th.to(device)

tmodel_blood_det_th = torch.jit.script(model_blood_det_th)
tmodel_blood_det_th.save("../weights/torchscript/traced_blood_det_th-{}.pt".format(device).replace(":0",""))

# Ir camera
# Load the pre-trained YOLO model for blood detection in IR images and convert it to TorchScript
model_blood_det_ir = YOLO.load_from_yolov5(args.weights_blood_det_ir, args.conf_thres_blood_ir, args.iou_thres_blood_ir)
model_blood_det_ir.eval()
model_blood_det_ir = model_blood_det_ir.to(device)

tmodel_blood_det_ir = torch.jit.script(model_blood_det_ir)
tmodel_blood_det_ir.save("../weights/torchscript/traced_blood_det_ir-{}.pt".format(device).replace(":0",""))

# Face object detection
# Load the pre-trained YOLO model for face detection and convert it to TorchScript
model_face_det = YOLO.load_from_yolov5(args.weights_face_det, args.conf_thres_face, args.iou_thres_face)
model_face_det.eval()
model_face_det = model_face_det.to(device)

tmodel_face_det = torch.jit.script(model_face_det)
tmodel_face_det.save("../weights/torchscript/traced_face_det-{}.pt".format(device).replace(":0",""))
