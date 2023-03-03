cmake_minimum_required(VERSION 3.12 FATAL_ERROR) 
project(main)

set(CMAKE_PREFIX_PATH "/home/ajvalenc/opt/libtorch") 
list(APPEND CMAKE_PREFIX_PATH "/home/ajvalenc/opt/torchvision")

find_package(Torch REQUIRED) 
find_package(TorchVision REQUIRED) 
find_package(OpenCV REQUIRED)

include_directories(include)

add_library(utils src/utils.cpp)

add_executable(main main.cpp)

target_link_libraries(utils ${OpenCV_LIBS} TorchVision::TorchVision)

target_compile_features(main PUBLIC cxx_range_for)
target_link_libraries(main utils)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET deserialize
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:deserialize>)
endif (MSVC)

set_property(TARGET main PROPERTY CXX_STANDARD 14)
