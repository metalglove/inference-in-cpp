#define UNICODE
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
#include <algorithm>
#include <iostream>

template <typename T>
static void softmax(T &input)
{
    float rowmax = *std::max_element(input.begin(), input.end());
    std::vector<float> y(input.size());
    float sum = 0.0f;
    for (size_t i = 0; i != input.size(); ++i)
    {
        sum += y[i] = std::exp(input[i] - rowmax);
    }
    for (size_t i = 0; i != input.size(); ++i)
    {
        input[i] = y[i] / sum;
    }
}

struct MNIST
{
    MNIST()
    {
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
        output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
    }

    std::ptrdiff_t Run()
    {
        const char *input_names[] = {"mnist_input"};
        const char *output_names[] = {"mnist_output"};
        session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

        softmax(results_);

        result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
        return result_;
    }

    static constexpr const int width_ = 28;
    static constexpr const int height_ = 28;

    std::array<float, width_ * height_> input_image_{};
    std::array<float, 10> results_{};
    int64_t result_{0};

private:
    Ort::Env env;
    Ort::Session session_{env, L"../mnist_model.onnx", Ort::SessionOptions{nullptr}};

    Ort::Value input_tensor_{nullptr};
    std::array<int64_t, 3> input_shape_{1, width_, height_};

    Ort::Value output_tensor_{nullptr};
    std::array<int64_t, 2> output_shape_{1, 10};
};

std::unique_ptr<MNIST> mnist_;

int main(int, char **)
{
    try
    {
        mnist_ = std::make_unique<MNIST>();
    }
    catch (const Ort::Exception &exception)
    {
        std::cout << "Huh";
        std::cout << exception.what();
        return 0;
    }

    float *output = mnist_->input_image_.data();

    std::fill(mnist_->input_image_.begin(), mnist_->input_image_.end(), 0.f);

    for (unsigned y = 4; y < MNIST::height_ - 4; y++)
    {
        output[(y * MNIST::width_) + 11] += 144.0f;
        output[(y * MNIST::width_) + 12] += 255.0f;
        output[(y * MNIST::width_) + 13] += 255.0f;
        output[(y * MNIST::width_) + 14] += 155.0f;
    }

    for (unsigned y = 0; y < MNIST::height_; y++)
    {
        printf("Row: %2d ", y);
        for (unsigned x = 0; x < MNIST::width_; x++)
        {
            if (output[(y * MNIST::width_) + x] != 0.0f)
            {
                std::cout << "1";
            }
            else
            {
                std::cout << "0";
            }
        }
        std::cout << "\n";
    }

    try
    {
        mnist_->Run();
    }
    catch (const Ort::Exception &exception)
    {
        std::cout << "Huh";
        std::cout << exception.what();
        return 0;
    }

    for (int i = 0; i < 10; i++)
    {
        std::cout << i << ": " << mnist_->results_[i] << "\n";
    }

    std::cout << "The result: " << mnist_->result_;

    return 0;
}