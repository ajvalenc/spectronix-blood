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

cv::Mat sixteen_bits2eight_bits(cv::Mat &image) {
	double max_value = 30100.0;
	double min_value;

	/* 
	// more precise but more expensive to compute
	image.convertTo(image, CV_8UC1, 255.0 / (max_value - min_value), - 255.0 * min_value / (max_value - min_value));
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);
	imag.convertTo(image, CV_32FC3, 1.0 / 255.0, 0);
	*/

    cv::minMaxIdx(image, &min_value);
 	image.convertTo(image, CV_32FC1, 1.0 / (max_value - min_value), - min_value / (max_value - min_value));
	cv::cvtColor(image, image, cv::COLOR_GRAY2BGR);

	return image;
}

	
int main(int argc, char **argv) {
  
  // load models
  torch::jit::script::Module model_bcls =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_bcls-cpu.pt");
  torch::jit::script::Module model_bdet =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_bdet-cpu.pt");
  torch::jit::script::Module model_fdet =
      get_module("/home/ajvalenc/OneDrive - University of Ottawa/Projects/spectronix/detection_models/blood/weights/torchscript/traced_fdet-cpu.pt");

  // read input
  std::string directory{"/home/ajvalenc/Datasets/spectronix/thermal/blood/16bit/s01_thermal_cloth_01_MicroCalibir_M0000334/"};
  std::vector<cv::String> filenames;
  cv::glob(directory, filenames, false);


  int i = 0;
  while (i < filenames.size()) {

	  // read image (16-bit)
	  cv::Mat img = cv::imread(filenames[i], cv::IMREAD_ANYDEPTH);
	  cv::Mat img_raw = img.clone();
		
	  // scale image
	  img = sixteen_bits2eight_bits(img);

	  // convert image to torch compatible input
	  auto tensor_img = torch::from_blob(img.data, {img.rows, img.cols, img.channels()}).to(device);
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

	  i += 1;
  }

  return 0;
}
