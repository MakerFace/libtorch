#include <vector>
#include <string>

#include <torch/torch.h>
#include <torch/script.h>

#include <opencv2/opencv.hpp>

#include <glog/logging.h>

#ifndef PROJECT_PATH
#define PROJECT_PATH ""
#endif

// Define a new Module.
struct Net : torch::nn::Module
{
    Net()
    {
        // Construct and register two Linear submodules.
        fc1 = register_module("fc1", torch::nn::Linear(784, 64));
        fc2 = register_module("fc2", torch::nn::Linear(64, 32));
        fc3 = register_module("fc3", torch::nn::Linear(32, 10));
    }

    // Implement the Net's algorithm.
    torch::Tensor forward(torch::Tensor x)
    {
        // Use one of many tensor manipulation functions.
        x = torch::relu(fc1->forward(x.reshape({x.size(0), 784})));
        x = torch::dropout(x, /*p=*/0.5, /*train=*/is_training());
        x = torch::relu(fc2->forward(x));
        x = torch::log_softmax(fc3->forward(x), /*dim=*/1);
        return x;
    }

    // Use one of many "standard library" modules.
    torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
};

int main(int argc, char *argv[])
{
    FLAGS_logtostderr = true;
    google::InitGoogleLogging(argv[0]);
    if (argc < 2)
    {
        LOG(ERROR) << "usage: predict model.pth";
        exit(1);
    }
    
    LOG(INFO) << "Start predict";
    torch::DeviceType device_type;
    if (torch::cuda::cudnn_is_available())
    {
        LOG(INFO) << "CUDA available! Predicting on GPU";
        device_type = torch::kCUDA;
    }
    else
    {
        LOG(ERROR) << "CUDA unavilable";
        exit(1);
    }
    torch::Device device(device_type);
    std::string model_pt = PROJECT_PATH;
    model_pt.append("/").append(argv[1]);
    LOG(INFO) << "load pytorch weight from " << model_pt;
    auto model = torch::jit::load(model_pt);
    model.to(at::kCUDA);

    auto data_loader = torch::data::make_data_loader(
        torch::data::datasets::MNIST("./data/raw/", torch::data::datasets::MNIST::Mode::kTest).map(
            torch::data::transforms::Stack<>()),
        64);

    for (auto &&img : *data_loader)
    {
        auto input = img.data.to(at::kCUDA);
        auto prediction = model.forward({input}).toTensor();
        LOG(INFO) << prediction;
    }
}