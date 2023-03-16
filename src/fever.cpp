#include "fever.hpp"

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> getPredictions(torch::IValue &predictions){
	
	auto detections = predictions.toTuple()->elements().at(1).toList().get(0).toGenericDict();
	torch::Tensor boxes = detections.at("boxes").toTensor();
	torch::Tensor scores = detections.at("scores").toTensor();
	torch::Tensor labels = detections.at("labels").toTensor();

	return {scores, boxes, labels};
}

double transferFunction(double &radiometry, int camera_id, bool is_linear) {
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

std::tuple<double, double, double> getTemperature(cv::Mat &image, torch::Tensor box, int camera_id) {

	float reference_temperature;
	cv::Rect reference_region;
	
	// differential temperature
	if (camera_id == 337) {
		reference_temperature = 21.5;
		reference_region = cv::Rect{600,40,15,15};
	}
	else if (camera_id == 334) {
		reference_temperature = 21.1;
		reference_region = cv::Rect{400,50,10,10};
	}

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

	max_temperature = transferFunction(max_radiometry_detect, camera_id, false);
	mean_temperature = transferFunction(mean_radiometry_detect, camera_id, false);
	patch_temperature = transferFunction(mean_radiometry_reference, camera_id, true);
	// compute differential temperature
	double maxdiff_temperature = max_temperature - patch_temperature + reference_temperature;
	double meandiff_temperature = mean_temperature - patch_temperature + reference_temperature;

	//std::tuple<double, double, double> output(maxdiff_temperature,meandiff_temperature,patch_temperature);

	return {maxdiff_temperature, meandiff_temperature, patch_temperature};
}

bool detectFever(torch::IValue &output, cv::Mat &image, int camera_id, int face_thresh, int forehead_thresh) {

	bool fever = false;
	auto [scores, boxes, labels] = getPredictions(output);
	
	for (size_t i =0; i < boxes.sizes()[0]; ++i) {
           auto [maxdiff_temp, meandiff_temp, patch_temp] = getTemperature(image, boxes[i], camera_id);
           int category = labels[i].item<int>();
           if (category == 0) {
           	std::cout << "Face detected ";
           	
           	if (maxdiff_temp > face_thresh) { //treshold for the warmest region on the face
           	   std::cout << "(Fever): ";
           	   fever = fever || true;
           	}
           	else {
           	   std::cout << "(No Fever): ";
           	   fever = fever || false;
           	}
           }	
           else if (category == 1) {
                std::cout << "Forehead detected ";
                
                if (maxdiff_temp > forehead_thresh) { //treshold for the warmest region on the forehead
           	   std::cout << "(Fever): ";
           	   fever = fever || true;
           	}
           	else {
           	   std::cout << "(No Fever): ";
           	   fever = fever || false;
           	}  	
            }
        std::cout << maxdiff_temp << "  " << meandiff_temp << "  " << patch_temp << "\n";
        }
        return fever;
}
