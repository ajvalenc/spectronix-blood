#include <string>
#include <chrono>
#include <opencv2/opencv.hpp>

#include <torch/torch.h>
#include <torch/script.h>

#include <torchvision/ops/nms.h>
#include <torchvision/vision.h>

torch::Device device(torch::kCPU);

torch::jit::script::Module get_module(const char *file_path) {
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
  
  if (argc != 2) {
		std::cerr << "Set an usage mode (0: offline, 1: online)";
		return -1;
	}
  int input_mode = std::stoi(argv[1]);
    
  // read input
  cv::Mat img_raw;
  cv::VideoCapture cap;

  if (input_mode) {
      cv::namedWindow("Thermal Camera");
      cap = cv::VideoCapture(0);  
  }
  else {
  	  std::string filename{"/home/ajvalenc/Datasets/spectronix/thermal/BinClass_Test/Pos/T_IM_%d.png"};
	  cap = cv::VideoCapture(filename);
  }

  if (!cap.isOpened()){
	  std::cerr << "failed to open image sequence";
  }


  // load models
  torch::jit::script::Module model_bcls =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_bcls-cpu.pt");
  torch::jit::script::Module model_bdet =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_bdet-cpu.pt");
  torch::jit::script::Module model_fdet =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/thermal/Cpp_Codes/weights/traced_fdet-cpu.pt");

  while (true){

	  // scale image
	  cap >> img_raw;
	  cv::Mat img = img_raw;
	  img.convertTo(img, CV_32F, 1.0 / 255.0, 0);

	  // convert image to torch compatible input
	  auto tensor_img =
		  torch::from_blob(img.data, {img.rows, img.cols, img.channels()})
			  .to(device);
	  tensor_img = tensor_img.permute({2, 0, 1});
	  tensor_img.unsqueeze_(0);

	  std::vector<torch::jit::IValue> inputs;
	  inputs.push_back(tensor_img);

	  // dry run 
	  torch::NoGradGuard no_grad; // ensure autograd is off
	  for (size_t i = 0; i < 2; ++i){
		  model_bcls.forward(inputs);
		  model_bdet.forward(inputs);
		  model_fdet.forward(inputs);
	  }

	  // inference
	  auto start = std::chrono::high_resolution_clock::now();
	  at::Tensor out_bcls = model_bcls.forward(inputs).toTensor();
	  auto end = std::chrono::high_resolution_clock::now();
	  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	  std::cout << "\n\nBlood classification\n"
				<< "runtime " << duration.count() << "\n"
				<< "classes " << out_bcls << "\n";

	  start = std::chrono::high_resolution_clock::now();
	  auto out_bdet = model_bdet.forward(inputs)
						  .toTuple()
						  ->elements()
						  .at(1)
						  .toList()
						  .get(0)
						  .toGenericDict();
	  end = std::chrono::high_resolution_clock::now();
	  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	  std::cout << "\nBlood detection\n"
				<< "runtime " << duration.count() << "\n"
				<< out_bdet << "\n";


	  start = std::chrono::high_resolution_clock::now();
	  auto out_fdet = model_fdet.forward(inputs)
						  .toTuple()
						  ->elements()
						  .at(1)
						  .toList()
						  .get(0)
						  .toGenericDict();
	  end = std::chrono::high_resolution_clock::now();
	  duration = std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
	  std::cout << "\nFace detection\n"
				<< "runtime " << duration.count() << "\n"
				<< out_fdet << "\n";

	  // display results
	  cv::imshow("Thermal Camera", img_raw);
	  if ((char)cv::waitKey(5) >0) break;
  }

  return 0;
}
