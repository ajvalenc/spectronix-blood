#include "blood.hpp"

std::tuple<torch::Tensor,torch::Tensor,torch::Tensor> getPredictions(torch::IValue &predictions){
	
	auto detections = predictions.toTuple()->elements().at(1).toList().get(0).toGenericDict();
	torch::Tensor boxes = detections.at("boxes").toTensor();
	torch::Tensor scores = detections.at("scores").toTensor();
	torch::Tensor labels = detections.at("labels").toTensor();

	return {scores, boxes, labels};
}

bool detectBlood(torch::IValue &output_th, torch::IValue &output_ir, float region_thresh, int brightness_thresh) {

	bool blood = false;

	return blood;
}


