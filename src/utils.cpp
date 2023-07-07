#include "utils.hpp"

cv::Mat sixteenBits2EightBits(cv::Mat &image, double &max_value) {
	double min_value;  
  cv::minMaxIdx(image, &min_value);
	image.convertTo(image, CV_8UC1, 255.0f / (max_value - min_value), - 255.0f * min_value / (max_value - min_value));

 	// less exprensive to compute but compute already normalize values
	//image.convertTo(image, CV_32FC1, 1.0 / (max_value - min_value), - min_value / (max_value - min_value));

	return image;
}

cv::Mat processImageThermal(cv::Mat &image) {
 cv::Mat image_prc = image.clone();

 // fix image type
  if (image_prc.depth() == CV_16U) {
     double max_value = 30100.0f;
     image_prc = sixteenBits2EightBits(image_prc, max_value);
  }
  // scale image content
  cv::cvtColor(image_prc, image_prc, cv::COLOR_GRAY2BGR);
  image_prc.convertTo(image_prc, CV_32FC3, 1.0f / 255.0f);

 return image_prc;
}

cv::Mat processImageIR(cv::Mat &image) {
 cv::Mat image_prc = image.clone();

 // fix image type
  if (image_prc.depth() == CV_16U) {
     double max_value = 395.0f;
     image_prc = sixteenBits2EightBits(image_prc, max_value);
  }
  // scale image content
  cv::cvtColor(image_prc, image_prc, cv::COLOR_GRAY2BGR);
  image_prc.convertTo(image_prc, CV_32FC3, 1.0f / 255.0f);

  // apply camera transformation
  //cv::Mat m = (cv::Mat_<double>(2,3) << 1.0306, 0.01071, -45.058,   -0.01071, 1.0306, -49.872);
  //cv::warpAffine(image_prc, image_prc, m, image_prc.size());

 return image_prc;
}

torch::Tensor toTensor(cv::Mat &image, torch::Device device) {

  // Convert image to torch tensor
  torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}).to(device);
  tensor_image = tensor_image.permute({2, 0, 1});
  tensor_image.unsqueeze_(0);

  return tensor_image;
}

std::vector<torch::jit::IValue> toInput(torch::Tensor &tensor_image) {

	return std::vector<torch::jit::IValue>{tensor_image};
}
