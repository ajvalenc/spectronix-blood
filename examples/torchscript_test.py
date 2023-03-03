import time

import cv2
import torch
import torch.nn as nn
from torchvision.io import read_image 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

# configuration
weights_blood_cls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_blood_cls-cpu.pt"
weights_blood_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_blood_det-cpu.pt"
weights_face_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_face_det-cpu.pt"

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load torchscript models
print("Loading binary model...")
tmodel_blood_cls = torch.jit.load(weights_blood_cls)
print("Binary model loaded successfully")

print("Loading detection model...")
tmodel_blood_det = torch.jit.load(weights_blood_det)
print("Detection model loaded successfully")

print("Loading detection model...")
tmodel_face_det = torch.jit.load(weights_face_det)
print("Detection model loaded successfully")

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/blood/8bit/Pos/T_IM_45.png"
img_raw = cv2.imread(filename) #original
img = cv2.imread(filename) #input

#img = cv2.resize(img, (480,640), interpolation=cv2.INTER_LINEAR)
img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
img /= 255.
img = img.unsqueeze(0)

# random input data
#img = torch.rand((3,480,640)).to(device)

with torch.no_grad(): #ensures autograd is off
    start = time.time()
    out_blood_cls = tmodel_blood_cls(img)
    end = time.time()
    print("Blood classification runtime: ", end-start)

    start = time.time()
    out_blood_det = tmodel_blood_det(img)
    end = time.time()
    print("Blood detection runtime: ", end-start)

    start = time.time()
    out_face_det = tmodel_face_det(img)
    end = time.time()
    print("Face detection runtime: ", end-start)
