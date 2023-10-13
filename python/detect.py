import argparse
import os
from os.path import join, dirname, realpath

import cv2
import torch
from yolov5 import detect
from yolort.models.yolo import YOLO

from roboflow import Roboflow

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="", help="file/dir/URL/glob")
parser.add_argument("--weights", type=str, default="", help="model.pt path(s) leave blank to use original pretrained weights")
parser.add_argument("--project", type=str, default="../output/runs", help="save results to project/name")
parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.4, help="Non-Maximum Suppression IOU threshold")
parser.add_argument("--yolort", action="store_true", help="use yolort model otherwise use standard yolov5 model")
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
    dir_source = join(dataset.location, "test/images")

dir_project = join(ROOT, args.project) # Path where results are saved
os.makedirs(dir_project, exist_ok=True)

# Inference
if not args.yolort:
    # Using YOLOv5 for inference
    detect.run(weights=fn_weights,  # Model.pt path(s)
            source=dir_source,  # File/dir/URL/glob, 0 for webcam
            imgsz=640,  # Inference size (pixels)
            conf_thres=args.conf_thres,  # Confidence threshold
            iou_thres=args.iou_thres,  # Non-maximum suppression IOU threshold
            max_det=1000,  # Maximum detections per image
            device=device,  # CUDA device, i.e. 0 or 0,1,2,3 or cpu
            view_img=False,  # Show results
            save_txt=True,  # Save results to *.txt
            save_conf=True,  # Save confidences in --save-txt labels
            save_crop=False,  # Save cropped prediction boxes
            nosave=False,  # Do not save images/videos
            classes=None,  # Filter by class: --class 0, or --class 0 2 3
            agnostic_nms=False,  # Class-agnostic Non-Maximum Suppression
            augment=False,  # Augmented inference
            visualize=False,  # Visualize features
            update=False,  # Update all models
            project=dir_project,  # Save results to project/name
            )
else:
    # Using YOLORT for inference
    model = YOLO.load_from_yolov5(args.weights, args.conf_thres, args.iou_thres)
    model = model.eval()
    model = model.to(device)

    # Loop through the images, make predictions, and print the results
    filenames = sorted(os.listdir(dir_source))
    for filename in filenames:
        
        img = cv2.imread(join(dir_source, filename))

        img = torch.as_tensor(img.astype("float32").transpose(2,0,1)).to(device)  # Prepare image for model input
        img /= 255.
        img = img.unsqueeze(0)

        with torch.no_grad():
            out = model(img)
            print(out)




