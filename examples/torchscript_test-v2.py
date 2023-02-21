import time

import cv2
import torch
import torch.nn as nn
from torchvision.io import read_image 
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

# configuration
weights_bcls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_bcls-cpu.pt" #blood cls
weights_bdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_bdet-cpu.pt" #blood det
weights_fdet = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_fdet-cpu.pt" #face det

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# load torchscript models
print("Loading binary model...")
model_bcls = torch.jit.load(weights_bcls)
print("Binary model loaded successfully")

print("Loading detection model...")
model_bdet = torch.jit.load(weights_bdet)
print("Detection model loaded successfully")

print("Loading detection model...")
model_fdet = torch.jit.load(weights_fdet)
print("Detection model loaded successfully")

# input data and transform (using opencv)
filename = "/home/ajvalenc/Datasets/spectronix/thermal/BinClass_Test/Pos/T_IM_45.png"
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
