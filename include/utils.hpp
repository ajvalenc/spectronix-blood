#include <tuple>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <torch/script.h>

torch::Tensor processImage(cv::Mat &image);

// 16 bits to 8 bits image conversion
cv::Mat sixteenBits2EightBits(cv::Mat &image);

// load torchscript model
torch::jit::script::Module getModule(const char *file_path, torch::Device device);

// get bouding boxes from model predictions
torch::Tensor getBoundingBoxes(torch::IValue predictions);


class DifferentialTemperature {
public:

	DifferentialTemperature(int camera);

	double transferFunction(double &radiometry, bool is_linear);

	std::tuple<double, double, double> getTemperature(cv::Mat &image, torch::Tensor boxes);

private:
	
	int camera_id;
	float reference_temperature;
	cv::Rect reference_region;
};
