#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

#include "utils.hpp"
#include "fever.hpp"

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
  torch::jit::script::Module tmodel_face_det = getModule("/home/ajvalenc/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_face_det-cuda.pt");

  // read input
  //std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/fever/16bit/M337/"};
  std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/blood/8bit/Pos/"};

  std::vector<cv::String> filenames;
  cv::utils::fs::glob(directory, "", filenames, false);

  // dry run 
  auto start = std::chrono::high_resolution_clock::now();
  std::vector<torch::jit::IValue> input;
  auto img_rand = torch::rand({1,3,640,480}).to(device);
  input.push_back(img_rand);
  torch::NoGradGuard no_grad; // ensure autograd is off
  for (size_t i = 0; i < 3; ++i){
	  tmodel_face_det.forward(input);
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

	  // set camera and fever thresholds
	  int cam_id = 337;
	  int face_thresh = 37, forehead_thresh = 35;

	  // read images
	  cv::Mat img_th = cv::imread(filenames[i], cv::IMREAD_ANYDEPTH);

	  // process input
	  auto start = std::chrono::high_resolution_clock::now();
    double max_value_th = 30100.0;
	  cv::Mat img_prc_th = processImage(img_th, max_value_th);
	  torch::Tensor ts_img_th = toTensor(img_prc_th, device);
      std::vector<torch::jit::IValue> input_th = toInput(ts_img_th);

	  // inference
	  auto mid1 = std::chrono::high_resolution_clock::now();
	  torch::IValue out_face_det = tmodel_face_det.forward(input_th);

    // decision making
    auto mid2 = std::chrono::high_resolution_clock::now();
	  bool fever = detectFever(out_face_det, img_th, cam_id, face_thresh, forehead_thresh);

	  auto end = std::chrono::high_resolution_clock::now();
    auto duration_prc = std::chrono::duration_cast<std::chrono::milliseconds>(mid1-start);
    auto duration_det = std::chrono::duration_cast<std::chrono::milliseconds>(mid2-mid1);
    auto duration_dm = std::chrono::duration_cast<std::chrono::milliseconds>(end-mid2);
    auto duration_total = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	  avg_runtime_prc += duration_prc.count();
	  avg_runtime_det += duration_det.count();
	  avg_runtime_dm += duration_dm.count();
	  avg_runtime_total += duration_total.count();
	  std::cout << "\nFace detection:\n" << out_face_det << "\n";
	  std::cout << "\nRuntime Processing:  " << avg_runtime_prc / (i+1.0f) << "\nRuntime detection:  " << avg_runtime_det / (i+1.0f) << "\nRuntime Decision Making:  " << avg_runtime_dm / (i+1.0f);
	  std::cout << "\nRuntime:  " << avg_runtime_total / (i+1.0f) << " Fps: " << 1000.0f * (i+1.0f) /  avg_runtime_total << "\n";

	  // display results
	  cv::imshow("Thermal Camera", img_th);
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
