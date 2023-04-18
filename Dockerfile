ARG BASE=ubuntu20.04
ARG CUDA=11.3.0
ARG CUDNN=8

FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-${BASE}
ENV DEBIAN_FRONTEND=noninteractive

# Install dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    wget \
    unzip \
    libopencv-dev \
    python3-dev \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Build versions
ARG TORCH=1.12.1
ARG TORCHVISION=0.13.1
ARG CPP=17
ARG GPUCOMPUTE=8.6

# PyTorch library
ARG CUDA
RUN CUDA_MAJOR_MINOR=$(echo ${CUDA} | perl -pe 's/(\d+)\.(\d+)\.\d+/$1$2/') && \
    wget https://download.pytorch.org/libtorch/cu${CUDA_MAJOR_MINOR}/libtorch-cxx11-abi-shared-with-deps-${TORCH}%2Bcu${CUDA_MAJOR_MINOR}.zip -O libtorch.zip && \
    unzip libtorch.zip -d /usr/local/ && \
    rm libtorch.zip
ENV Torch_DIR=/usr/local/libtorch

# Torchvision library
RUN wget https://github.com/pytorch/vision/archive/refs/tags/v${TORCHVISION}.zip && \
    unzip v${TORCHVISION}.zip && \
    rm v${TORCHVISION}.zip && \ 
    cmake -B vision-${TORCHVISION}/build -S vision-${TORCHVISION} -DCMAKE_INSTALL_PREFIX=/usr/local/torchvision -DCMAKE_CXX_STANDARD=${CPP} -DCUDA_GPU_DETECT_OUTPUT=${GPUCOMPUTE} -DUSE_PYTHON=on -DWITH_CUDA=on && \
    cmake --build vision-${TORCHVISION}/build -j $(nproc) && \
    cmake --install vision-${TORCHVISION}/build
ENV TorchVision_DIR=/usr/local/torchvision
