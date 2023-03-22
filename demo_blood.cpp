#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

#include "utils.hpp"
#include "blood.hpp"

torch::Device device(torch::kCPU);

torch::jit::script::Module getModule(const char *file_path) {
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

int main(int argc, char **argv) {
  
  // create models
  torch::jit::script::Module tmodel_blood_det_th = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_th-cpu.pt");
  torch::jit::script::Module tmodel_blood_det_ir = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_ir-cpu.pt");


  // read input
  std::string directory_th{"/home/ajvalenc/Datasets/spectronix/thermal/blood/16bit/s21_thermal_cloth_01_MicroCalibir_M0000334/"};
  std::string directory_ir{"/home/ajvalenc/Datasets/spectronix/ir/blood/registered/s21_thermal_cloth_01_000028493212_ir/"};

  std::vector<cv::String> filenames;
  cv::utils::fs::glob_relative(directory_ir, "", filenames, false); //ir has less entries

  // dry run 
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<torch::jit::IValue> input;
  auto img_rand = torch::rand({1,3,640,480}).to(device);
  input.push_back(img_rand);
  torch::NoGradGuard no_grad; // ensure autograd is off
  for (size_t i = 0; i < 3; ++i){
	  tmodel_blood_det_th.forward(input);
	  tmodel_blood_det_ir.forward(input);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
  std::cout << "\nWarmuptime:  " << duration.count() << " Fps: " << 1000.0f / duration.count() << "\n";

  int i = 0;
  float avg_runtime = 0.0f;
  while (i < filenames.size()) {

	  // set camera
	  int cam337 = 337;
	  double iou_thresh = 0.2, detectable_blood_thresh = 100;
	
	  // read images
	  cv::Mat img_th = cv::imread(directory_th + "/" + filenames[i], cv::IMREAD_ANYDEPTH);
	  cv::Mat img_ir = cv::imread(directory_ir + "/" + filenames[i], cv::IMREAD_ANYDEPTH);

	  // process input
	  cv::Mat img_prc_th = processImage(img_th);
	  torch::Tensor ts_img_th = toTensor(img_prc_th, device);
      std::vector<torch::jit::IValue> input_th = toInput(ts_img_th);

	  cv::Mat img_prc_ir = processImage(img_ir);
	  torch::Tensor ts_img_ir = toTensor(img_prc_ir, device);
      std::vector<torch::jit::IValue> input_ir = toInput(ts_img_ir);

	  // inference
	  auto start = std::chrono::high_resolution_clock::now();

	  torch::IValue out_blood_det_th = tmodel_blood_det_th.forward(input_th);
	  torch::IValue out_blood_det_ir = tmodel_blood_det_ir.forward(input_ir);

      std::cout << "\nBlood detection\n";
	  std::cout << "Thermal: " << out_blood_det_th << "\n";
	  std::cout << "IR: " << out_blood_det_ir << "\n";
	  bool blood = detectBlood(out_blood_det_th, out_blood_det_ir, img_ir, iou_thresh, detectable_blood_thresh);

	  auto end = std::chrono::high_resolution_clock::now();
	  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
      avg_runtime += duration.count();
	  std::cout << "\nRuntime:  " << avg_runtime / (i+1.0f) << " Fps: " << 1000.0f * (i+1.0f) /  avg_runtime << "\n";


	  // display results
	  cv::imshow("Thermal Camera", img_ir);
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
