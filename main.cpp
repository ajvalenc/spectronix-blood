#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

#include "utils.hpp"

torch::Device device(torch::kCPU);

int main(int argc, char **argv) {
  
  // create models
  torch::jit::script::Module tmodel_blood_cls = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_blood_cls-cpu.pt", device);
  torch::jit::script::Module tmodel_blood_det = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_blood_det-cpu.pt", device);
  torch::jit::script::Module tmodel_face_det = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_face_det-cpu.pt", device);

  // read input
  std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/blood/16bit/s01_thermal_cloth_01_MicroCalibir_M0000334/"};
  //std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/blood/8bit/Pos/"};

  std::vector<cv::String> filenames;
  cv::glob(directory, filenames, false);


  int i = 0;
  while (i < filenames.size()) {

	  // read image (16-bit)
	  cv::Mat img_raw = cv::imread(filenames[i], cv::IMREAD_ANYDEPTH);
	  cv::Mat img = img_raw.clone();

	  // process image
	  torch::Tensor img_tensor = processImage(img);
  	  
	  // format tensor to list ivalue required by torchscript model
  	  std::vector<torch::jit::IValue> inputs;
  	  inputs.push_back(img_tensor);

	  // dry run 
	  torch::NoGradGuard no_grad; // ensure autograd is off
	  for (size_t i = 0; i < 2; ++i){
		  tmodel_blood_cls.forward(inputs);
		  tmodel_blood_det.forward(inputs);
		  tmodel_face_det.forward(inputs);
	  }

	  // inference
	  torch::IValue out_blood_cls = tmodel_blood_cls.forward(inputs);
	  std::cout << "\n\nBlood Classification\n"<< out_blood_cls << "\n";

	  torch::IValue out_blood_det = tmodel_blood_det.forward(inputs);
	  std::cout << "\nBlood detection\n" << out_blood_det << "\n";
		
	  torch::IValue out_face_det = tmodel_face_det.forward(inputs);
	  std::cout << "\nFace detection\n" << out_face_det << "\n";

	  // fever detection
	  DifferentialTemperature diff_temp_m334(334);
      torch::Tensor boxes = getBoundingBoxes(out_face_det);
	
	  for (size_t i =0; i < boxes.sizes()[0]; ++i) {

		  auto [maxdiff_temp, meandiff_temp, patch_temp] = diff_temp_m334.getTemperature(img_raw, boxes[i]);

		 std::cout << "Fever temps " << maxdiff_temp << "  " << meandiff_temp << "  " << patch_temp << "\n";

		//cv::Rect box(left, top, (right-left), (bottom-top));
		//cv::rectangle(img, box, cv::Scalar(0,255,0), 2);
	  }

	  // display results
	  cv::imshow("Thermal Camera", img_raw);
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
