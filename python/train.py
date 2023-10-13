import argparse
import os
from os.path import join, dirname, realpath

import cv2
import torch
from yolov5 import train
from roboflow import Roboflow

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="", help="file/dir/URL/glob")
parser.add_argument("--weights", type=str, default="", help="model.pt path(s) leave blank to use original pretrained weights")
parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.4, help="Non-Maximum Suppression IOU threshold")
parser.add_argument("--epochs", type=int, default=500, help="number of epochs")
parser.add_argument("--batch-size", type=int, default=16, help="batch size")
parser.add_argument("--api-key", type=str, default="fI0NwCgOFaNOoDDHvDbs", help="Roboflow API key")
parser.add_argument("--username", type=str, default="uospec-pya0l", help="Roboflow username")
parser.add_argument("--project-id", type=str, default="thermal_face_forehead", help="Roboflow project ID")
parser.add_argument("--num-version", type=int, default=4, help="Roboflow project number version")
parser.add_argument("--annotation-format", type=str, default="yolov5", help="Roboflow export annotation format")
args = parser.parse_args()

# Determine if CUDA (GPU) is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = False if str(device) == "cpu" else True

# Directory configuration
ROOT = dirname(realpath(__file__))
fn_weights = args.weights # Path where weights are saved
if not os.path.isfile(fn_weights): # Download default pre-trained weights if not exist locally
    dir_weights = join(ROOT, "../weights/pytorch")
    if not os.path.isdir(dir_weights):
        os.makedirs(dir_weights, exist_ok=True)
    fn_weights = join(dir_weights, "yolov5s.pt")
    print("Warning: Weights file does not exist. Using default pre-trained weights: ", fn_weights)

dir_source = args.source # Path where source images are saved
if not os.path.isdir(dir_source):
    print("Warning: Source directory does not exist. Attempting to download from Roboflow.")

    # Initialize the Roboflow API client with your API key
    rf = Roboflow(api_key=args.api_key)

    # Access your Roboflow workspace and project
    project = rf.workspace(args.username).project(args.project_id)

    # Access the desired model version
    model = project.version(args.num_version).model

    # Download the dataset locally
    os.environ["DATASET_DIRECTORY"] = join(os.path.expanduser("~"), "Downloads/content", args.annotation_format)  # Path where dataset is saved
    dataset = project.version(args.num_version).download(args.annotation_format)
    dir_source = dataset.location
    
# Training
train.run(weights=fn_weights,  # model.pt path(s)
          data=join(dir_source, "data.yaml"),  # data.yaml path
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
          cache_images=True)  # cache images for faster training




