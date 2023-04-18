#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

#include "utils.hpp"
#include "blood_combined.hpp"
#include "fever_combined.hpp"

torch::Device device(torch::kCUDA);

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
  torch::jit::script::Module tmodel_det_th = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_det_th-cuda.pt");
  torch::jit::script::Module tmodel_det_ir = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det_ir-cuda.pt");


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
	  tmodel_det_th.forward(input);
	  tmodel_det_ir.forward(input);
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
std::cout << "\nWarmuptime:  " << duration.count() << " Fps: " << 1000.0f / duration.count() << "\n";

  int i = 0;
  float avg_runtime_prc = 0.0f;
  float avg_runtime_det = 0.0f;
  float avg_runtime_dm = 0.0f;
  float avg_runtime_total = 0.0f;
  while (i < filenames.size()) {

	  // set camera
	  int cam_id = 337;
    int face_thresh = 37, forehead_thresh = 35;
	  double iou_thresh = 0.2, detectable_blood_thresh = 100;
	
	  // read images
	  cv::Mat img_th = cv::imread(directory_th + "/" + filenames[i], cv::IMREAD_ANYDEPTH);
	  cv::Mat img_ir = cv::imread(directory_ir + "/" + filenames[i], cv::IMREAD_ANYDEPTH);
     std::cout << "\nDetection - frame " << i;
     
    // process input
    auto start = std::chrono::high_resolution_clock::now();
	  cv::Mat img_prc_th = processImage(img_th);
	  torch::Tensor ts_img_th = toTensor(img_prc_th, device);
      std::vector<torch::jit::IValue> input_th = toInput(ts_img_th);

	  cv::Mat img_prc_ir = processImage(img_ir);
	  torch::Tensor ts_img_ir = toTensor(img_prc_ir, device);
      std::vector<torch::jit::IValue> input_ir = toInput(ts_img_ir);

    // inference
    auto mid1 = std::chrono::high_resolution_clock::now();
	  torch::IValue out_det_th = tmodel_det_th.forward(input_th);
	  torch::IValue out_det_ir = tmodel_det_ir.forward(input_ir);
    
    // decision making
    auto mid2 = std::chrono::high_resolution_clock::now();
    bool fever = detectFever(out_det_th, img_th, cam_id, face_thresh, forehead_thresh);
	  bool blood = detectBlood(out_det_th, out_det_ir, img_ir, iou_thresh, detectable_blood_thresh);

	  // display results
	  auto end = std::chrono::high_resolution_clock::now();
    auto duration_prc = std::chrono::duration_cast<std::chrono::milliseconds>(mid1-start);
    auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(mid2-mid1);
    auto duration_dm = std::chrono::duration_cast<std::chrono::milliseconds>(end-mid2);
	  auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    avg_runtime_prc += duration_prc.count();
    avg_runtime_det += duration_det.count();
    avg_runtime_dm += duration_dm.count();
    avg_runtime_total += duration_total.count();
	  std::cout << "\nThermal: " << out_det_th;
	  std::cout << "\nIR: " << out_det_ir;
	  std::cout << "\nRuntime Processing:  " << avg_runtime_prc / (i+1.0f) << "\nRuntime detection:  " << avg_runtime_det / (i+1.0f) << "\nRuntime Decision Making:  " << avg_runtime_dm / (i+1.0f);
	  std::cout << "\nRuntime:  " << avg_runtime_total / (i+1.0f) << " Fps: " << 1000.0f * (i+1.0f) /  avg_runtime_total << "\n";
    
	  cv::imshow("Thermal Camera", img_ir);
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
