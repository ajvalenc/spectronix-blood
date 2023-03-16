#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// model predictions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getPredictions(torch::IValue &predictions);

// blood detection
bool detectBlood(torch::IValue &output_th, torch::IValue &output_ir, float region_thresh, int brightness_thresh);

