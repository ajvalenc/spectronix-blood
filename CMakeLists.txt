cmake_minimum_required(VERSION 3.12 FATAL_ERROR) 
project(main)

set(CMAKE_PREFIX_PATH "/home/ajvalenc/opt/libtorch") 
list(APPEND CMAKE_PREFIX_PATH "/home/ajvalenc/opt/torchvision")

find_package(Torch REQUIRED) 
find_package(TorchVision REQUIRED) 
find_package(OpenCV REQUIRED)

include_directories(include)

add_library(utils src/utils.cpp)
add_library(fever src/fever.cpp)
add_library(blood src/blood.cpp)

add_executable(demo_fever demo_fever.cpp)
add_executable(demo_blood demo_blood.cpp)

target_link_libraries(utils ${OpenCV_LIBS} TorchVision::TorchVision)
target_link_libraries(fever ${OpenCV_LIBS} TorchVision::TorchVision)
target_link_libraries(blood ${OpenCV_LIBS} TorchVision::TorchVision)

target_compile_features(demo_fever PUBLIC cxx_range_for)
target_compile_features(demo_blood PUBLIC cxx_range_for)
target_link_libraries(demo_fever utils fever blood)
target_link_libraries(demo_blood utils fever blood)

if (MSVC)
  file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
  add_custom_command(TARGET deserialize
                     POST_BUILD
                     COMMAND ${CMAKE_COMMAND} -E copy_if_different
                     ${TORCH_DLLS}
                     $<TARGET_FILE_DIR:deserialize>)
endif (MSVC)

set_property(TARGET demo_fever PROPERTY CXX_STANDARD 14)
set_property(TARGET demo_blood PROPERTY CXX_STANDARD 14)
