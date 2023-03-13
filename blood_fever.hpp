#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

// image conversion (16bits -> 8bits)
cv::Mat sixteenBits2EightBits(cv::Mat &image);

// image processing block
torch::Tensor processImage(cv::Mat &image_raw);

// model predictions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getPredictions(torch::IValue predictions);

// temperature calculation
double transferFunction(double &radiometry, bool is_linear);

std::tuple<double, double, double> getTemperature(cv::Mat &image, torch::Tensor boxes, int camera_id);

// fever detection
bool detectFever(torch::jit::script::Module &model, cv::Mat &image, int camera_id);

