# Spectronix Real-Time Blood Recognition Models


## Build instruction

cmake ..
make


## serialize models using cuda (Fuser produce errors, disable!)
!PYTORCH_JIT_USE_NNC_NOT_NVFUSER=1 serialize.py


## Build and install libtorch, torchvision 
Follow this guide https://www.neuralception.com/settingupopencv/

For torchvision use the following to avoid having to install with sudo

cmake -D CMAKE_INSTALL_PREFIX:PATH=/home/ajvalenc/opt/torchvision ..

