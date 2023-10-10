# Real-Time Blood and Fever Detection Models

## Model
The detection models are based on Ultralytics YOLOv5(see python scripts to adjust input parameters).

Train: Fine tuning YOLOv5 model with custom dataset (retrains using Yolov5s weights by default).

```bash
python train.py --source=path/to/data/images
```

Detect: Inference YOLOv5 model with retrained weights on test image data.

```bash
python detect.py --source=path/to/data/images --weights=path/to/weights
```

Serialize: Convert native PyTorch weights to torchscript serialized `.pt` files.

```bash
python serialize.py
```

## Data
### Raw Sensors

Cloud service [WD MyCloud](https://os5.mycloud.com/) is used to store images from the multiple cameras. 

### Labels

The labeled data for the detection models are generated using [Roboflow](https://universe.roboflow.com/uospec-pya0l/thermal_all_classes-jqkur) service. Data is available in a zip folder and can be downloaded from the cloud
- Blood samples: `\\SPECTRONIX-D1\U Ottawa Shared Data\Blood Detection Dataset\Annotated`


## Build Instructions

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
