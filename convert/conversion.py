from PIL import Image
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

def sixteen_bits2eight_bits_IR(path):
    #input_img = Image.open(path)
    #pixel = np.array(input_img)

    input_img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    pixel = np.array(input_img)

    q1 = np.quantile(pixel.flatten(), 0.25)
    q3 = np.quantile(pixel.flatten(), 0.75)
    threshold = q3+3*(q3-q1)

    pixel = 255. * (pixel - np.min(pixel)) / (threshold-np.min(pixel))
    pixel = np.clip(pixel, 0, 255)
    pixel =np.uint8(pixel)

    return pixel, threshold, q1, q3

dir_in = "/home/ajvalenc/Datasets/spectronix/ir/blood/raw/s21_thermal_cloth_01_000028493212_ir"

filenames = sorted(os.listdir(dir_in))

for filename in filenames:
    path = os.path.join(dir_in, filename)
    img, threshold, q1, q3 = sixteen_bits2eight_bits_IR(path)

    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imshow("img", img)
    if (cv2.waitKey(5) >0):
        break

    print(threshold, q1, q3)

