import argparse
import os
from os.path import join, dirname, realpath

import cv2
import torch
from yolov5 import detect
from yolort.models.yolo import YOLO

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="../data/images", help="file/dir/URL/glob")
parser.add_argument("--project", type=str, default="../output/runs/detect", help="save results to project/name")
parser.add_argument("--weights", type=str, default="", help="model.pt path(s) leave blank to use original pretrained weights")
parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.4, help="NMS IOU threshold")
parser.add_argument("--yolort", action="store_true", help="use yolort model otherwise use yolov5 model")
parser.add_argument("--model", type=str, default="yolov5", help="yolo model name")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# directory configuration
ROOT = dirname(realpath(__file__))
dir_weights = join(ROOT, args.weights)
if not os.path.isfile(dir_weights):
    dir_weights = args.model + "s.pt"
    print("Error: weights file does not exist. Using default weights " + dir_weights)

dir_source = join(ROOT, args.source)
if not os.path.isdir(dir_source):
    print("Error: source directory does not exist")
    exit()

dir_project = join(ROOT, args.project)
os.makedirs(dir_project, exist_ok=True)

# inference
if not args.yolort:
    detect.run(weights=dir_weights,  # model.pt path(s)
            source=dir_source,  # file/dir/URL/glob, 0 for webcam
            imgsz=640,  # inference size (pixels)
            conf_thres=args.conf_thres,  # confidence threshold
            iou_thres=args.iou_thres,  # NMS IOU threshold
            max_det=1000,  # maximum detections per image
            device=device,  # cuda device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # show results
            save_txt=True,  # save results to *.txt
            save_conf=True,  # save confidences in --save-txt labels
            save_crop=False,  # save cropped prediction boxes
            nosave=False,  # do not save images/videos
            classes=None,  # filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # class-agnostic NMS
            augment=False,  # augmented inference
            visualize=False,  # visualize features
            update=False,  # update all models
            project=dir_project,  # save results to project/name
            )
else:
    model = YOLO.load_from_yolov5(args.weights, args.conf_thres, args.iou_thres)
    model = model.eval()
    model = model.to(device)

    filenames = sorted(os.listdir(dir_source))

    for filename in filenames:
        # input data and transform (using opencv)
        img_raw = cv2.imread(join(dir_source, filename))
        img = img_raw.copy()

        img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)
        img /= 255.
        img = img.unsqueeze(0)

        # process outputs
        with torch.no_grad():
            out = model(img)
            print(out)




