#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>

// model predictions
std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> getPredictions(torch::IValue &predictions);

// temperature calculation
double transferFunction(double &radiometry, bool is_linear);

std::tuple<double, double, double> getTemperature(cv::Mat &image, torch::Tensor boxes, int camera_id);

// fever detection
bool detectFever(torch::IValue &output, cv::Mat &image, int camera_id, int face_thresh, int forehead_thresh);



