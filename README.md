# Real-Time Blood and Fever Detection

## Models
The detection models are based on Ultralytics YOLOv5. These models can be adjusted with additional arguments using the Python scripts provided in `python/` directory. The `src` contains the C++ code for real-time inference and decision making system.

### Training
To train the YOLOv5 model with custom dataset, run the following command:

```bash
python python/train.py --source=path/to/data/images --weights=path/to/weights --epochs=100 --batch-size=16
```

- `--source`: Specifies the path to the directory containing the training images. If not provided, the script will attempt to retrieve the data from Roboflow.
- `--weights`: Specifies the path to the pre-trained weights file. If not provided, it will use the default `yolov5s.pt` pre-trained weights.
- `--epochs`: Specifies the number of training epochs to run (default is 300).
- `--batch-size`: Specifies the batch size to use during training (default is 16).

### Inference

### On Python
To perform inference on test images using the retrained YOLOv5 model, run the following command:

```bash
python python/detect.py --source=path/to/data/images --weights=path/to/weights --conf-thres=0.5 --iou-thres=0.4
```
- `--source`: Specifies the path to the directory containing the test images. If not provided, the script will attempt to retrieve the data from Roboflow.
- `--weights`: Specifies the path to the retrained weights file. If not provided, it will use the default Yolov5.pt pre-trained weights.
- `--conf-thres`: Specifies the confidence threshold for object detection (default is 0.5).
- `--iou-thres`: Specifies the Non-Maximum Suppression IoU threshold for object detection (default is 0.4).

### On C++
Once the project is build, the executables can be found in `build/` directory. The weights files must be placed in `weights/torchscript/` directory. To perform real-time inference on test images using detection models, run the following command:

```bash
./demo_fever --source path/to/data/images
```
- `--source`: Specifies the path to the directory containing the test images.
- `traced_face_det.pt`: Face detection torchscript weights file.

```bash
./demo_blood --source_th path/to/data/images_th --source_ir path/to/data/images_ir
```
- `--source_th`: Specifies the path to the directory containing the test images from thermal camera.
- `--source_ir`: Specifies the path to the directory containing the test images from IR camera.
- `traced_blood_det_th.pt`: Blood detection torchscript weights file for thermal images.
- `traced_blood_det_ir.pt`: Blood detection torchscript weights file for IR images.

### Serialization
To convert the native PyTorch weights to torchscript format, run the following command:

```bash
python python/serialize.py
```

- `--weights-blood-cls`: Specifies the path to the retrained weights file for blood classification model. If not provided, it will use the default `resnet101_blood_cls.pt` pre-trained weights.
- `--weights-blood-det-th`: Specifies the path to the retrained weights file for blood detection model in thermal images. If not provided, it will use the default `yolov5_blood_det_th.pt` pre-trained weights.
- `--weights-blood-det-ir`: Specifies the path to the retrained weights file for blood detection model in IR images. If not provided, it will use the default `yolov5_blood_det_ir.pt` pre-trained weights.
- `--weights-face-det`: Specifies the path to the retrained weights file for face detection model. If not provided, it will use the default `yolov5_face_det.pt` pre-trained weights.
- `conf-thres-blood-th`: Specifies the confidence threshold for blood detection in thermal images (default is 0.4).
- `iou-thres-blood-th`: Specifies the Non-Maximum Suppression IoU threshold for blood detection in thermal images (default is 0.5).
- `conf-thres-blood-ir`: Specifies the confidence threshold for blood detection in IR images (default is 0.4).
- `iou-thres-blood-ir`: Specifies the Non-Maximum Suppression IoU threshold for blood detection in IR images (default is 0.5).
- `conf-thres-face`: Specifies the confidence threshold for face detection (default is 0.3).
- `iou-thres-face`: Specifies the Non-Maximum Suppression IoU threshold for face detection (default is 0.5).

## Datasets
### Raw Sensors

Cloud service [WD MyCloud](https://os5.mycloud.com/) is used to store images from the multiple cameras. 

### Labels

The labeled data for the detection models are generated using [Roboflow](https://universe.roboflow.com/uospec-pya0l/thermal_all_classes-jqkur) service. Data is available in a zip folder and can be downloaded from the cloud
- Blood samples: `\\SPECTRONIX-D1\U Ottawa Shared Data\Blood Detection Dataset\Annotated`


## Build Instructions
The following instructions are used to build the `src/` files into an executable. The project can be built using the provided `CMakeLists.txt` file. The following dependencies are required:

### Local Install

Download [TorchVision](https://github.com/pytorch/vision). Make sure to use the compatible version with libTorch (tested vision 0.13.1 for libtorch 1.12.1 main versions). Requires CMAKE > 3.12 (Ubuntu 18.04 update [see discussion](https://github.com/orgs/robotology/discussions/364))

```bash
mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch -DCMAKE_INSTALL_PREFIX=/pth/to/installation -DUSE_PYTHON=on -DWITH_CUDA=on  ..
```

Note: GPU detection must be set to compute 8.6 if AdaLover GPUs are used since cu113 versions does not support compute 8.9.

##### On Linux

Target must be compiled using C++14. Otherwise linking errors occurs ([see issue](https://github.com/pytorch/vision/issues/3833)).

```bash
cmake -DCUDA_GPU_DETECT_OUTPUT=8.6 ..
cmake --build . -j nproc
```

##### On Windows

```bash
 cmake -DCMAKE_DISABLE_FIND_PACKAGE_MKL=ON ..
 cmake --config . --build Release -j nproc
```

### Containers

#### Conda Environment
Create the environment from the `environment.yml` file.

```bash
conda env create -f environment.yml
```

If the environment is created from scratch, the [yolort](https://github.com/zhiqwang/yolov5-rt-stack) package must be installed from source. There is a bug in the current PyPi release v0.6.3 (since v0.4.3) that requires some editing in the source code ([see issue](https://github.com/zhiqwang/yolov5-rt-stack/issues/466)).Therefore, it is recommended to install the package from source as the bug is fixed in the latest commits.

```bash
git clone
cd yolov5-rt-stack
pip install -e .
```

#### Docker Container

This is used to isolate the development and avoid potential conflicts with dependencies. Thera are a few docker images that can be used to create containers based on what software is pre-installed:

- nvidia/cuda:11.3.0-cudnn8-devel-ubuntu20.04 (currently used)
- nvidia/cuda:12.1.0-devel-ubuntu22.04
- pytorch/pytorch:1.12.1-cuda11.3-cudnn8-devel

The following `Dockerfile` can be used to create an image of the environment. Important details are:
- Install [openCV](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) using distro packages
- Install [libTorch](https://pytorch.org/cppdocs/installing.html) from pre-compiled binaries ("cxx11 ABI" version is used as it avoids problems such as [linking error](https://stackoverflow.com/questions/61456607/cmake-linking-error-finding-library-but-undefined-reference) caused by conda environements).
- Install [TorchVision](https://github.com/pytorch/vision) from source. This is required in order to use `nms` models used by YOLO (make sure to download a compatible version with libTorch).

Enable display (should be done before entering the container)

```bash
# host
export DISPLAY=:1.0
xhost +
```

```bash
#container
export DISPLAY=:1.0
./demo_fever
```

##### Manual Setup

Create container from official Nvidia with CUDA and CUDNN libraries. Mount the appropriate volumes and set the environment variables using the `-v` and `-e` flags respectively.

```bash
docker run -it --rm --runtime=nvidia --gpus all --name spectronix --net=host cuda:11.3.0-cudnn8-devel-ubuntu20.04 nvidia-smi bash
```

Execute an interactive shell on the container

```bash
docker exec -it spectronix bash
```

Install Dependencies

```bash
apt-get update && apt-get install -y --no-install-recommends \    
    build-essential \                                                 
    cmake \
    wget \
    unzip \
	libopencv-dev \ 
	python3-dev \                                            
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
```

Proceed to install libTorch and build TorchVision for C++ development

##### Automatic Build

Build image from Dockerfile

```bash
docker build -t spectronix:11.3.0-v0.1 .
```

Run container. Mount the appropriate volumes and set the environment variables using the `-v` and `-e` flags respectively.

```bash
docker run -it --rm --runtime=nvidia --gpus all --name spectronix --net=host spectronix:11.3.0-v0.1 bash
```
