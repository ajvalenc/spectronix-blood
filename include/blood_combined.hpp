#pragma once

#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// Extract blood detection predictions from model output
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getBloodPredictions(torch::IValue &predictions);

// Calculate the mean pixel value of a region defined by a bounding box
double getMeanPatch(cv::Mat &image, torch::Tensor box);

// Calculate the intersection over union of two bounding boxes
double getIOU(torch::Tensor box_1, torch::Tensor box_2);

// Detect the presence of dark and warm blood in thermal and infrared images
bool detectBlood(torch::IValue &output_th, torch::IValue &output_ir, cv::Mat &image_ir, double region_thresh, double brightness_thresh);

