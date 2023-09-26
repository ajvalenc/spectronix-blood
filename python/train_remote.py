import argparse
import os
from os.path import join, dirname, realpath

import cv2
import torch
from yolov5 import train
from roboflow import Roboflow

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="../data/images", help="file/dir/URL/glob")
parser.add_argument("--project", type=str, default="runs/detect", help="save results to project/name")
parser.add_argument("--weights", type=str, default="", help="model.pt path(s) leave blank to use original pretrained weights")
parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.4, help="NMS IOU threshold")
parser.add_argument("--api-key", type=str, default="fI0NwCgOFaNOoDDHvDbs", help="Roboflow API key")
parser.add_argument("--username", type=str, default="uospec-pya0l", help="Roboflow username")
parser.add_argument("--project-id", type=str, default="thermal_face_forehead", help="Roboflow project ID")
parser.add_argument("--num_version", type=int, default=4, help="Roboflow project number version")
parser.add_argument("--model", type=str, default="yolov5", help="yolo model name")
parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("--batch-size", type=int, default=16, help="batch size")
args = parser.parse_args()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# configuration
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

# setup environment
os.environ["DATASET_DIRECTORY"] = join("/home/ajvalenc/Downloads/content", args.model)

# roboflow dataset
rf = Roboflow(api_key=args.api_key)
project = rf.workspace(args.username).project(args.project_id)
dataset = project.version(args.num_version).download(args.model)

train.run(weights=dir_weights,  # model.pt path(s)
          data=join(dataset.location, "data.yaml"),  # data.yaml path
          epochs=args.epochs,  # number of epochs
          batch_size=args.batch_size,  # batch size
          imgsz=640,  # train image size
          rect=False,  # rectangular training
          resume=False,  # resume training from checkpoint
          nosave=False,  # do not save checkpoints
          noval=False,  # do not save checkpoints
          noautoanchor=False,  # disable autoanchor check
          evolve=False,  # evolve hyperparameters
          bucket="",  # parent bucket of dataset
          cache_images=True,  # cache images for faster training
)




