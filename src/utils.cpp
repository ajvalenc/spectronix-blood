#include "utils.hpp"

torch::Tensor processImage(cv::Mat &image) {

  if (image.depth() == CV_16U) {
     image = sixteenBits2EightBits(image);
  }
  // scale image
  cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
  image.convertTo(image, CV_32FC3, 1.0f / 255.0f);

  // Convert image to torch tensor
  torch::Tensor image_tensor = torch::from_blob(image.data, {image.rows, image.cols, 3});
  image_tensor = image_tensor.permute({2, 0, 1});
  image_tensor.unsqueeze_(0);

  return image_tensor;
}

cv::Mat sixteenBits2EightBits(cv::Mat &image) {
	double max_value = 30100.0;
	double min_value;
 
    cv::minMaxIdx(image, &min_value);
	image.convertTo(image, CV_8UC1, 255.0 / (max_value - min_value), - 255.0 * min_value / (max_value - min_value));

 	// less exprensive to compute but compute already normalize values
	//image.convertTo(image, CV_32FC1, 1.0 / (max_value - min_value), - min_value / (max_value - min_value));

	return image;
}

torch::jit::script::Module getModule(const char *file_path, torch::Device device) {
  torch::jit::script::Module module;

  std::ifstream input(file_path, std::ios::binary);

  if (input)
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(input, device = device);
  else
    std::cerr << "ifstream not provided ";

  std::cout << "Module successfully loaded \n";

  return module;
}

torch::Tensor getBoundingBoxes(torch::IValue predictions){
	
	torch::Tensor boxes = predictions.toTuple()->elements().at(1).toList().get(0).toGenericDict().at("boxes").toTensor(); 

	return boxes;
}

DifferentialTemperature::DifferentialTemperature(int camera) {
	camera_id = camera;
	
	if (camera_id == 337) {
		reference_temperature = 21.5;
		reference_region = cv::Rect{600,40,15,15};
	}
	else if (camera_id == 334) {
		reference_temperature = 21.1;
		reference_region = cv::Rect{400,50,10,10};
	}

}

double DifferentialTemperature::transferFunction(double &radiometry, bool is_linear) {
	double a, b, c, T;
	if (camera_id == 337) {
		if (is_linear) {
			T = (0.02309 * radiometry) - 648.80056;
		}
		else {
			a = 0.5548024811500611;
    		b = -8.931341192663892e-06;
    		c = -8562.309528172134;

	    	T = (a * radiometry) + (b * (radiometry * radiometry)) + c;
		}
	}
	else if (camera_id == 334) {
		if (is_linear) {
			T = (0.01989 * radiometry) - 552.21746;
		}
		else {
			a = 0.4931711470645102;
    		b = -7.968154638301412e-06;
    		c = -7579.7939226811395;

    		T = (a * radiometry) + (b * (radiometry * radiometry)) + c;
		}
	}
	return T;
}

std::tuple<double, double, double> DifferentialTemperature::getTemperature(cv::Mat &image, torch::Tensor box) {

	// extract region points
	float left = box[0].item().toFloat();
	float top = box[1].item().toFloat();
	float right = box[2].item().toFloat();
	float bottom = box[3].item().toFloat();

	// define region of interest
	cv::Rect detect_region(left, top, (right-left), (bottom-top));
	
	// crop image and patch
	cv::Mat detect_patch= image(detect_region);
	cv::Mat reference_patch = image(reference_region);

	// get radiometry stats
	double max_radiometry_detect, mean_radiometry_detect, mean_radiometry_reference;

	cv::minMaxIdx(detect_patch, NULL, &max_radiometry_detect);
	cv::Scalar u1_dim = cv::mean(detect_patch);
	mean_radiometry_detect =  u1_dim.val[0];
	cv::Scalar u2_dim = cv::mean(reference_patch);
	mean_radiometry_reference = u2_dim.val[0];
	// compute temperature stats
	double max_temperature, mean_temperature, patch_temperature;

	max_temperature = transferFunction(max_radiometry_detect,false);
	mean_temperature = transferFunction(mean_radiometry_detect, false);
	patch_temperature = transferFunction(mean_radiometry_reference, true);
	// compute differential temperature
	double maxdiff_temperature = max_temperature - patch_temperature + reference_temperature;
	double meandiff_temperature = mean_temperature - patch_temperature + reference_temperature;

	std::tuple<double, double, double> output(maxdiff_temperature,meandiff_temperature,patch_temperature);

	return output;
}


