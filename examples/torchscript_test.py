import time

import cv2
import torch
import torch.nn as nn
from torchvision.io import read_image
from torchvision.datasets import ImageFolder
from torchvision.transforms import Compose, ToTensor

from yolort.utils import Visualizer

from PIL import Image
import numpy as np

def sixteen_bits2eight_bits(pixel):
    # Pseudo colouring the 16 bit images
    pixel = (pixel - np.min(pixel)) / (30100-np.min(pixel))
    pixel = np.rint(pixel * 255)
    return pixel.astype("uint8")
    
def visualize(image, normalize=False):

    if normalize:
        image = cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 

    # rescale (16-bit image only)
    if (image.dtype) == np.uint16:
        image = image / 257
        image = image.astype(np.uint8)

    #cv2.normalize(image, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U) 
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image

# configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True
weights_blood_cls = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_cls-{}.pt".format(device).replace(":0","")
weights_blood_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_th-{}.pt".format(device).replace(":0","")
weights_face_det = "/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_face_det-{}.pt".format(device).replace(":0","")

# load torchscript models 
print("Loading binary model...")
model_blood_cls = torch.jit.load(weights_blood_cls)
print("Binary model loaded successfully")

print("Loading detection model...")
model_blood_det = torch.jit.load(weights_blood_det)
print("Detection model loaded successfully")

print("Loading detection model...")
model_face_det = torch.jit.load(weights_face_det)
print("Detection model loaded successfully")

# input data and transform (using opencv)
#filename = "/home/ajvalenc/Datasets/spectronix/thermal/reference/T_IM_45.png"
#filename = "/home/ajvalenc/OneDrive - University of Ottawa/Datasets/spectronix/thermal/fever/16bit/M337/a1_M337.tiff"
filename = "/home/ajvalenc/Datasets/spectronix/thermal/blood/16bit/s01_thermal_cloth_01_MicroCalibir_M0000334/0001.png"
img = cv2.imread(filename,cv2.IMREAD_ANYDEPTH) #input
    
# conversion
img = sixteen_bits2eight_bits(img)

# channels
img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
img /= 255.
img = img.unsqueeze(0)

# dry run
for i in range(3):
    model_blood_cls(img)
    model_blood_det(img)
    model_face_det(img)

with torch.no_grad(): #ensures autograd is off
    start = time.time()
    out_bcls = model_blood_cls(img)
    end = time.time()
    print("Blood classification runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_bdet = model_blood_det(img)
    end = time.time()
    print("Blood detection runtime [ms]: ", 1000*(end-start))

    start = time.time()
    out_fdet = model_face_det(img)
    end = time.time()
    print("Face detection runtime [ms]: ", 1000*(end-start))
    
print("Output bdet:", out_bdet[1][0])
print("Output fdet:", out_fdet[1][0])

# verify results
img_raw = cv2.imread(filename) #original
v = Visualizer(visualize(img_raw, True), ["Cold_Background", "Cold_Body", "Warm_Background", "Warm_Body", "Warm_Dripping"])
v.draw_instance_predictions(out_bdet[1][0])
v.imshow(scale=0.5)

# verify results
img_raw = cv2.imread(filename) #original
v = Visualizer(visualize(img_raw,True), ["Face", "ForeHead"])
v.draw_instance_predictions(out_fdet[1][0])
v.imshow(scale=0.5)
