#include "blood.hpp"

const double EPSILON = 1e-5;

namespace dm{
  const int num_frames = 30;
  std::vector<int> detection_dark_his(num_frames, 0);
  std::vector<int> detection_warm_his(num_frames, 0);
};

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> getBloodPredictions(torch::IValue &predictions){
	
	auto detections = predictions.toTuple()->elements().at(1).toList().get(0).toGenericDict();
	torch::Tensor boxes = detections.at("boxes").toTensor();
	torch::Tensor scores = detections.at("scores").toTensor();
	torch::Tensor labels = detections.at("labels").toTensor();
	
	return {scores, boxes, labels};
}

double getMeanPatch(cv::Mat &image, torch::Tensor box) {

	// extract region points
	float left = box[0].item().toFloat();
	float top = box[1].item().toFloat();
	float right = box[2].item().toFloat();
	float bottom = box[3].item().toFloat();

	// define region of interest
	cv::Rect region(left, top, (right - left), (bottom - top));
	
	// crop patch
	cv::Mat detect_patch= image(region);

	// get mean patch
    cv::Scalar mean_patch = cv::mean(detect_patch);

	return mean_patch.val[0];
}

double getIOU(torch::Tensor box_1, torch::Tensor box_2) {

	// extract region points
	float left_1 = box_1[0].item().toFloat();
	float top_1 = box_1[1].item().toFloat();
	float right_1 = box_1[2].item().toFloat();
	float bottom_1 = box_1[3].item().toFloat();

	float left_2 = box_2[0].item().toFloat();
	float top_2 = box_2[1].item().toFloat();
	float right_2 = box_2[2].item().toFloat();
	float bottom_2 = box_2[3].item().toFloat();

	// intersection coordinates
	float left = std::max(left_1, left_2);
	float top = std::max(top_1, top_2);
    float right = std::min(right_1, right_2);
    float bottom = std::min(bottom_1, bottom_2);

	// overlapped area
	float overlap_area = std::max((right - left), 0.0f) * std::max((bottom - top), 0.0f);

	// no overlapped case
	float area_1 = (right_1 - left_1) * (bottom_1 - top_1);
	float area_2 = (right_2 - left_2) * (bottom_2 - top_2);

	float total_area = area_1 + area_2 - overlap_area; 

	double iou = overlap_area / (total_area + EPSILON);

	return iou;
}

std::tuple<bool,bool> detectBlood(torch::IValue &output_th, torch::IValue &output_ir, cv::Mat &image_ir, double region_thresh, double brightness_thresh) {

	// initialize flags
	bool dark_liquid = false, warm_liquid = false;
	auto [scores_th, boxes_th, labels_th] = getBloodPredictions(output_th);

	if (labels_th.sizes() == 0) return {dark_liquid, warm_liquid}; //early break when no thermal detection is found
	
	auto [scores_ir, boxes_ir, labels_ir] = getBloodPredictions(output_ir);
	for (size_t i=0; i < boxes_th.sizes()[0]; ++i) {

		int category = labels_th[i].item<int>();
		if (labels_ir.sizes() == 0) { //no liquid ir
      if (category >= 2) { //warm liquid thermal
				double mean_patch_ir = getMeanPatch(image_ir, boxes_th[i]);
				if (mean_patch_ir < brightness_thresh) { //dark region ir
					warm_liquid = true;
				}
			}
		}
		else { //dark liquid ir
			for (size_t j=0; j < boxes_ir.sizes()[0]; ++j) {
				double iou = getIOU(boxes_th[i], boxes_ir[j]);
				if (iou >= region_thresh) {
					dark_liquid = true;
				}
			}
		}

	}

	return {dark_liquid, warm_liquid};
}


std::tuple<bool,bool> detectBloodThermal(torch::IValue &output_th, double region_thresh, double brightness_thresh) {

	// initialize flags
	bool dark_liquid = false, warm_liquid = false;
	auto [scores_th, boxes_th, labels_th] = getBloodPredictions(output_th);

	if (labels_th.sizes() == 0) return {dark_liquid, warm_liquid}; //early break when no thermal detection is found
	
	for (size_t i=0; i < boxes_th.sizes()[0]; ++i) {

		int category = labels_th[i].item<int>();
    if (category >= 2) { //warm liquid thermal
					warm_liquid = true;
		}

	}

	return {dark_liquid, warm_liquid};
}

void isBloodAlarm(std::tuple<bool,bool> blood, float rate) {

	// extract detection results
	bool dark_liquid = std::get<0>(blood);
	bool warm_liquid = std::get<1>(blood);

	// initialize flags
  int threshold = int (dm::num_frames * rate / 100.0f);

  // update detection history
  dm::detection_dark_his.erase(dm::detection_dark_his.begin());
	dm::detection_warm_his.erase(dm::detection_warm_his.begin());
  dm::detection_dark_his.push_back(dark_liquid);
  dm::detection_warm_his.push_back(warm_liquid);
 
  // activate alarm based on detection history
  int counter_dark = std::accumulate(dm::detection_dark_his.begin(), dm::detection_dark_his.end(), 0);
  int counter_warm = std::accumulate(dm::detection_warm_his.begin(), dm::detection_warm_his.end(), 0);

  if (counter_dark >= threshold) {
		std::cout << "\nWARNING! **blood event** dark liquid detected";}
	if (counter_warm >= threshold) {
		std::cout << "\nATTENTION! **possible blood event** warm liquid detected";}

}
