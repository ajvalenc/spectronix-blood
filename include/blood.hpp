#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// model predictions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getPredictions(torch::IValue &predictions);

// blood detection
double getMeanPatch(cv::Mat &image, torch::Tensor box);
double getIOU(torch::Tensor box_1, torch::Tensor box_2);
bool detectBlood(torch::IValue &output_th, torch::IValue &output_ir, cv::Mat &image_ir, double region_thresh, double brightness_thresh);

