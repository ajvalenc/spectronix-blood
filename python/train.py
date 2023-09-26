import torch
import os

import yolov5
from yolov5 import train


# load model if exists
weights_blood_cls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_cls-{}.pt".format(device).replace(":0","")
weights_blood_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det-{}.pt".format(device).replace(":0","")
weights_face_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_face_det-{}.pt".format(device).replace(":0","")