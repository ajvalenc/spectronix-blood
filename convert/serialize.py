'''
create a serialize version of Resnet101 model
'''
import sys
import argparse
import copy
from os.path import realpath, dirname, join

import numpy as np
import cv2
import torch
import torch.nn as nn
from torchvision.models import resnet101

sys.path.append("yolov7-script")
import models
from models.experimental import attempt_load
from models.yolo import PostProcess
from utils.activations import SiLU, Hardswish

def export_yolo(weights, device):
    """
    enable tracing for yolo models [modified from export.py]
    """
    model = attempt_load(weights, map_location=device)

    # Update model
    for k, m in model.named_modules():
        m._non_presistent_buffers_set = set()

        if isinstance(m, models.common.Conv): #assign export-friendly funcs
            if isinstance(m.act, nn.Hardswish):
                m.act = Hardswish()
            elif isinstance(m.act, nn.SiLU):
                m.act = SiLU()
    
    model.model[-1].export = True #set Detect() layer grid export
    
    # script Detect() layer (non-traceable)
    model.model[-1] = torch.jit.script(model.model[-1])

    return model

# configuration
parser = argparse.ArgumentParser()
parser.add_argument("--weights_bcls", type=str, default="./models/Blood_Classification_resnet101.pth", help="blood binary classifier model")
parser.add_argument("--weights_bdet", type=str, default="./models/Blood_Detection_Yolov7.pt", help="blood detection model")
parser.add_argument("--weights_fdet", type=str, default="./models/Face_Detection_Yolov7.pt", help="blood detection model")

args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#####################NOT USED################################
#feature_extract = False
#num_classes = 2

# load pre-trained model
#model = resnet101(weights="IMAGENET1K_V1") #avoid warning and should be same as pretrained=True
#model.to(device)

# edit model
#num_ftrs = model.fc.in_features
#model.fc = nn.Linear(num_ftrs, num_classes)
#model.fc = model.fc.to(device)
############################################################

# blood binary classifer
model_bcls = torch.load(args.weights_bcls, map_location=device)
model_bcls.eval()

x = torch.rand((1,3,480,640), dtype=torch.float32).to(device)
traced_model_bcls = torch.jit.trace(model_bcls, x)
traced_model_bcls.save("./weights/traced_bcls-{}.pt".format(device).replace(":0",""))

# blood object detection
model_bdet = export_yolo(args.weights_bdet, device)
nms = PostProcess(0.2,0.45,None)
script_nms = torch.jit.script(nms)
#####################NOT USED################################
#ll_bdet = Detect(nc=5, #number of classes
#        anchors=([12,16,19,36,40,28],
#            [36,75,76,55,72,146],
#            [142,110,192,243,459,401]),
#        ch=(256,512,1024)) #customized layer for scripting
#prev = model_bdet.model[-1]
#ll_bdet.load_state_dict(prev.state_dict())
#ll_bdet.stride = prev.stride
#ll_bdet.to(device)
#model_fdet.traced = True #disable tracing for last layer
############################################################

x = torch.rand((1,3,640,640), dtype=torch.float32).to(device)
y = model_bdet(x) #dry run
traced_model_bdet = torch.jit.trace(model_bdet, x, strict=False)
traced_model_bdet.save("./weights/traced_bdet-{}.pt".format(device).replace(":0",""))

# face object detection
model_fdet = export_yolo(args.weights_fdet, device)

x = torch.rand((1,3,640,640), dtype=torch.float32).to(device)
y = model_bdet(x) #dry run
traced_model_fdet = torch.jit.trace(model_fdet, x, strict=False)
traced_model_fdet.save("./weights/traced_fdet-{}.pt".format(device).replace(":0",""))

