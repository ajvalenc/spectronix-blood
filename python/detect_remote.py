import argparse
import os
from os.path import join, dirname, realpath
from roboflow import Roboflow

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--source", type=str, default="../data/images", help="file/dir/URL/glob")
parser.add_argument("--conf-thres", type=float, default=0.5, help="confidence threshold")
parser.add_argument("--iou-thres", type=float, default=0.4, help="NMS IOU threshold")
parser.add_argument("--api-key", type=str, default="fI0NwCgOFaNOoDDHvDbs", help="Roboflow API key")
parser.add_argument("--project-id", type=str, default="thermal_face_forehead", help="Roboflow project ID")
parser.add_argument("--num_version", type=int, default=4, help="Roboflow project number version")
parser.add_argument("--data-format", type=str, default="yolov5", help="model data format")
args = parser.parse_args()

# directory configuration
ROOT = dirname(realpath(__file__))
dir_source = join(ROOT, args.source)
if not os.path.isdir(dir_source):
    print("Error: source directory does not exist")
    exit()

# roboflow dataset
rf = Roboflow(api_key=args.api_key)
project = rf.workspace(args.data_format).project(args.project_id)
model = project.version(args.num_version).model

# inference
filenames = sorted(os.listdir(dir_source))

for filename in filenames:
    print(model.predict(join(dir_source, filename), confidence=args.conf_thres, overlap=args.iou_thres).json())
    #model.predict(filename, confidence=conf_thres, overlap=iou_thres).save("prediction.jpg")  # visualize prediction