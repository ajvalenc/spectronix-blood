#pragma once

#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// model predictions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getBloodPredictions(torch::IValue &predictions);

// blood detection
double getMeanPatch(cv::Mat &image, torch::Tensor box);
double getIOU(torch::Tensor box_1, torch::Tensor box_2);
std::tuple<bool,bool> detectBlood(torch::IValue &output_th, torch::IValue &output_ir, cv::Mat &image_ir, double region_thresh, double brightness_thresh);

// decision making
namespace dm {
  extern const int num_frames;
  extern std::vector<int> detection_dark_his;
  extern std::vector<int> detection_warm_his;
};

void isBloodAlarm(std::tuple<bool,bool> blood, float rate);
