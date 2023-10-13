#include "utils.hpp"

cv::Mat sixteenBits2EightBits(cv::Mat &image, double &max_value) {
	double min_value;  
  cv::minMaxIdx(image, &min_value);

  // Convert to 8-bit and normalize the image
	image.convertTo(image, CV_8UC1, 255.0f / (max_value - min_value), - 255.0f * min_value / (max_value - min_value));

 	// Alternatively, you can use the following line for normalization. Less expressive but compute already normalized values
	//image.convertTo(image, CV_32FC1, 1.0 / (max_value - min_value), - min_value / (max_value - min_value));

	return image;
}

cv::Mat processImageThermal(cv::Mat &image) {
 cv::Mat image_prc = image.clone();

 // Check image type and convert it to 8-bit if it's 16-bit
  if (image_prc.depth() == CV_16U) {
     double max_value = 30100.0f;
     image_prc = sixteenBits2EightBits(image_prc, max_value);
  }
  // Scale image content and change its type
  cv::cvtColor(image_prc, image_prc, cv::COLOR_GRAY2BGR);
  image_prc.convertTo(image_prc, CV_32FC3, 1.0f / 255.0f);

 return image_prc;
}

cv::Mat processImageIR(cv::Mat &image) {
 cv::Mat image_prc = image.clone();

 // Check image type and convert it to 8-bit if it's 16-bit
  if (image_prc.depth() == CV_16U) {
     double max_value = 395.0f;
     image_prc = sixteenBits2EightBits(image_prc, max_value);
  }
  // Scale image content and change its type.
  cv::cvtColor(image_prc, image_prc, cv::COLOR_GRAY2BGR);
  image_prc.convertTo(image_prc, CV_32FC3, 1.0f / 255.0f);

  // Optionally, apply a camera transformation if needed
  //cv::Mat m = (cv::Mat_<double>(2,3) << 1.0306, 0.01071, -45.058,   -0.01071, 1.0306, -49.872);
  //cv::warpAffine(image_prc, image_prc, m, image_prc.size());

 return image_prc;
}

torch::Tensor toTensor(cv::Mat &image, torch::Device device) {

  // Convert image to a torch tensor and rearrange dimensions.
  torch::Tensor tensor_image = torch::from_blob(image.data, {image.rows, image.cols, 3}).to(device);
  tensor_image = tensor_image.permute({2, 0, 1});
  tensor_image.unsqueeze_(0);

  return tensor_image;
}

std::vector<torch::jit::IValue> toInput(torch::Tensor &tensor_image) {

	return std::vector<torch::jit::IValue>{tensor_image};
}
