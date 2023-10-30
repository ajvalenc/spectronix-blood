import os
from os.path import join, expanduser

import cv2
import numpy as np
from PIL import Image

def sixteen_bits2eight_bits(pixel, max_value):
    # Pseudo colouring the 16 bit images
    pixel = (pixel - np.min(pixel)) / (max_value-np.min(pixel))
    pixel = np.rint(pixel * 255)
    return pixel.astype("uint8")


HOME = expanduser("~")
DATA_DIR = join(HOME, "Downloads/slack_spectronix/investigate/thermal")
IMAGE_DIR = join(DATA_DIR, "16bit/9")
OUTPUT_DIR = join(DATA_DIR, "8bit/9")

for image_name in os.listdir(IMAGE_DIR):
    image_path = join(IMAGE_DIR, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_ANYDEPTH)
    image = sixteen_bits2eight_bits(image, 30100)
    #image.save(join(OUTPUT_DIR, image_name))
    cv2.imwrite(join(OUTPUT_DIR, image_name), image)
