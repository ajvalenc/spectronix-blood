#pragma once

#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Extract face detection predictions from model output
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getFeverPredictions(torch::IValue &predictions);

// Calculate the transformation from radiometry to temperature
double transferFunction(double &radiometry, bool is_linear);

// Compute the maximum, minimum and mean temperature of a region defined by a bounding box
std::tuple<double, double, double> getTemperature(cv::Mat &image, torch::Tensor boxes, int camera_id);

// Detect the presence of fever in thermal images
bool detectFever(torch::IValue &output, cv::Mat &image, int camera_id, int face_thresh, int forehead_thresh);



