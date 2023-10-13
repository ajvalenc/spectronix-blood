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
  
  // Create models
  torch::jit::script::Module tmodel_blood_det_th = getModule("../weights/torchscript/traced_blood_det_th-cuda.pt");
  torch::jit::script::Module tmodel_blood_det_ir = getModule("../weights/torchscript/traced_blood_det_ir-cuda.pt");

  // Initialize variables
  std::string directory_th;
  std::string directory_ir;

  if (argc != 5) {
      std::cerr << "Usage: " << argv[0] << " --source_th <directory_path_thermal> --source_ir <directory_path_ir" << std::endl;
      return 1;
  }
  if (std::string(argv[1]) == "--source_th" && std::string(argv[3]) == "--source_ir") {
      directory_th = argv[2];
      directory_ir = argv[4];
  } else {
      std::cerr << "Usage: " << argv[0] << " --source_th <directory_path> --source_ir <directory_path_ir" << std::endl;
      return 1;
  }

  // Read input
  std::vector<cv::String> filenames;
  cv::utils::fs::glob_relative(directory_ir, "", filenames, false); // IR has less entries

  // Dry run to warm up
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<torch::jit::IValue> input;
  auto img_rand = torch::rand({1,3,640,480}).to(device);
  input.push_back(img_rand);
  torch::NoGradGuard no_grad; // Ensure autograd is off
  for (size_t i = 0; i < 3; ++i){
	  tmodel_blood_det_th.forward(input);
	  tmodel_blood_det_ir.forward(input);
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

	  // Set camera parameters
	  int cam_id = 337;
	  double iou_thresh = 0.5;
    double detectable_blood_thresh = 100;
	
	  // Read images
	  cv::Mat img_th = cv::imread(directory_th + "/" + filenames[i], cv::IMREAD_ANYDEPTH);
	  cv::Mat img_ir = cv::imread(directory_ir + "/" + filenames[i], cv::IMREAD_ANYDEPTH);
    std::cout << "\nFrame: " << i;

	  // Process input
    auto start = std::chrono::high_resolution_clock::now();
	  cv::Mat img_prc_th = processImageThermal(img_th);
	  torch::Tensor ts_img_th = toTensor(img_prc_th, device);
    std::vector<torch::jit::IValue> input_th = toInput(ts_img_th);

	  cv::Mat img_prc_ir = processImageIR(img_ir);
	  torch::Tensor ts_img_ir = toTensor(img_prc_ir, device);
    std::vector<torch::jit::IValue> input_ir = toInput(ts_img_ir);

	  // Inference
    auto mid1 = std::chrono::high_resolution_clock::now();
	  torch::IValue out_blood_det_th = tmodel_blood_det_th.forward(input_th);
	  torch::IValue out_blood_det_ir = tmodel_blood_det_ir.forward(input_ir);
    
    // Process detections
    auto mid2 = std::chrono::high_resolution_clock::now();
	  auto blood = detectBlood(out_blood_det_th, out_blood_det_ir, img_ir, iou_thresh, detectable_blood_thresh);
    //auto blood = detectBloodThermal(out_blood_det_th, iou_thresh, detectable_blood_thresh);

    // Decision making module
    float det_rate = 50.0f;
    isBloodAlarm(blood, det_rate);

	  // Display results
	  auto end = std::chrono::high_resolution_clock::now();
    auto duration_prc = std::chrono::duration_cast<std::chrono::milliseconds>(mid1-start);
    auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(mid2-mid1);
    auto duration_dm = std::chrono::duration_cast<std::chrono::milliseconds>(end-mid2);
	  auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    avg_runtime_prc += duration_prc.count();
    avg_runtime_det += duration_det.count();
    avg_runtime_dm += duration_dm.count();
    avg_runtime_total += duration_total.count();
	      
    bool blood_th = std::get<0> (blood);
    bool blood_ir = std::get<1> (blood);
    if (blood_th || blood_ir){
      std::cout << "\nThermal: " << out_blood_det_th;
      std::cout << "\nIR: " << out_blood_det_ir;
    }

	  // display results
    std::cout << "\nRuntime Processing:  " << avg_runtime_prc / (i+1.0f) << "\nRuntime detection:  " << avg_runtime_det / (i+1.0f) << "\nRuntime Decision Making:  " << avg_runtime_dm / (i+1.0f);
	  std::cout << "\nRuntime:  " << avg_runtime_total / (i+1.0f) << " Fps: " << 1000.0f * (i+1.0f) /  avg_runtime_total << "\n";
	  cv::imshow("Ir Camera", img_prc_ir);
    cv::imshow("Thermal Camera", img_prc_th);
	  
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
