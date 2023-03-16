#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

#include "blood_fever.cpp"

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
  torch::jit::script::Module tmodel_blood_cls = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_cls-cpu.pt");
  torch::jit::script::Module tmodel_blood_det = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_blood_det-cpu.pt");
  torch::jit::script::Module tmodel_face_det = getModule("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood_fever/weights/torchscript/traced_face_det-cpu.pt");

  // read input
  std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/fever/16bit/M337/"};
  //std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/blood/8bit/Pos/"};

  std::vector<cv::String> filenames;
  cv::glob(directory, filenames, false);

  // dry run 
	  std::vector<torch::jit::IValue> input;
	  auto img_rand = torch::rand({1,3,640,480});
	  input.push_back(img_rand);
	  torch::NoGradGuard no_grad; // ensure autograd is off
	  for (size_t i = 0; i < 2; ++i){
		  tmodel_blood_cls.forward(input);
		  tmodel_blood_det.forward(input);
		  tmodel_face_det.forward(input);
	  }

  int i = 0;
  while (i < filenames.size()) {

	  // read image (16-bit)
	  cv::Mat img_raw = cv::imread(filenames[i], cv::IMREAD_ANYDEPTH);
	  int cam337 = 337;

	  // process input
	  cv::Mat img = processImage(img_raw);
	  torch::Tensor ts_img = toTensor(img);
      std::vector<torch::jit::IValue> input = toInput(ts_img);

	  // inference
	  //torch::IValue out_blood_cls = tmodel_blood_cls.forward(input);
	  //std::cout << "\n\nBlood Classification\n"<< out_blood_cls << "\n";

	  //torch::IValue out_blood_det = tmodel_blood_det.forward(input);
	  //std::cout << "\nBlood detection\n" << out_blood_det << "\n";
		
	  auto start = std::chrono::high_resolution_clock::now();
	  torch::IValue out_face_det = tmodel_face_det.forward(input);
	  std::cout << "\nFace detection\n" << out_face_det << "\n";

	  bool fever = detectFever(out_face_det, img_raw, cam337);

	  auto end = std::chrono::high_resolution_clock::now();
	  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	  std::cout << "\nruntime " << duration.count() << "\n";

	  // display results
	  cv::imshow("Thermal Camera", img_raw);
	  if ((char)cv::waitKey(5) >0) break;

	  i += 1;
  }

  return 0;
}
