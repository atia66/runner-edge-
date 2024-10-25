
#ifndef CONV_H
#define CONV_H
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Structs.h"

class Layer
{
public:
    Weights w_b;
    bool toggle = true;
    Layer() : w_b() {};
    Layer(int input_size, int output_size, int Ksize)
        : w_b(input_size, output_size, Ksize)
    {
        initialize_weights_and_biases(input_size, output_size, Ksize);
    }

private:
    void initialize_weights_and_biases(int input_size, int output_size, int Ksize)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(-0.1, 0.1);
#pragma omp parallel for
        for (int i = 0; i < input_size; i++)
        {
            for (int j = 0; j < output_size; j++)
            {
                for (int ki = 0; ki < Ksize; ++ki)
                {
                    for (int kj = 0; kj < Ksize; ++kj)
                    {
                        w_b.weight[i][j][ki][kj] = distr(eng);
                    }
                }
            }
        }
#pragma omp parallel for
        for (int j = 0; j < output_size; j++)
        {
            w_b.bais[j] = distr(eng);
        }
    }
};

class Linear
{
    int output_size;

public:
    Fcweight w_b;
    Linear() : w_b() {};
    Linear(int input_size, int outputsize) : w_b(input_size, outputsize)
    {
        output_size = outputsize;
        initialize_weights_and_biases(input_size, output_size);
    }
    vector<float> Fclayer(Tensor img)
    {
        vector<float> flatten = apply_flatten(img);
        vector<float> output(output_size);
#pragma omp parallel for
        for (int out = 0; out < output_size; ++out)
        {
            for (int idx = 0; idx < flatten.size(); ++idx)
            {
                output[out] += flatten[idx] * w_b.weight[out][idx];
            }
            output[out] += w_b.bais[out];
        }
        return output;
    }
    vector<float> Fclayer(vector<float> img)
    {
        vector<float> output(output_size);
#pragma omp parallel for
        for (int out = 0; out < output_size; ++out)
        {
            for (int idx = 0; idx < img.size(); ++idx)
            {
                output[out] += img[idx] * w_b.weight[out][idx];
            }
            output[out] += w_b.bais[out];
        }
        return output;
    }
    void backward(vector<float> gradlayer, vector<float> input, float learning_rate)
    {
        int out_layers = w_b.weight.size();
        int in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int out = 0; out < out_layers; ++out)
        {
            for (int in = 0; in < in_layers; ++in)
            {
                w_b.weight[out][in] -= learning_rate * gradlayer[out] * input[in];
            }
            w_b.bais[out] -= learning_rate * gradlayer[out];
        }
    }
    void backward(vector<float> gradlayer, Tensor Input, float learning_rate)
    {
        vector<float> input = apply_flatten(Input);
        int out_layers = w_b.weight.size();
        int in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int out = 0; out < out_layers; ++out)
        {
            for (int in = 0; in < in_layers; ++in)
            {
                w_b.weight[out][in] -= learning_rate * gradlayer[out] * input[in];
            }
            w_b.bais[out] -= learning_rate * gradlayer[out];
        }
    }
    vector<float> grad(vector<float> gradlayer)
    {
        vector<float> grad_input(w_b.weight[0].size(), 0.0f);
        int out_layers = w_b.weight.size();
        int in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int j = 0; j < in_layers; j++)
        {
            for (int i = 0; i < out_layers; i++)
            {
                grad_input[j] += w_b.weight[i][j] * gradlayer[i];
            }
        }
        return grad_input;
    }

private:
    void initialize_weights_and_biases(int input_size, int output_size)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(-0.1, 0.1);
#pragma omp parallel for
        for (int i = 0; i < output_size; i++)
        {
            for (int j = 0; j < input_size; j++)
            {
                w_b.weight[i][j] = distr(eng);
            }
            w_b.bais[i] = distr(eng);
        }
    }

    vector<float> apply_flatten(Tensor img)
    {
        int channel = img.image.size();
        int height = img.image[0].size();
        int width = img.image[0][0].size();
        vector<float> flatten(channel * height * width);
#pragma omp parallel for
        for (int ch = 0; ch < channel; ++ch)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    flatten[w + (h * width) + (ch * height * width)] = img.image[ch][h][w];
                }
            }
        }
        return flatten;
    }
};
class convlve2d
{
    int stride;
    int padding;
    int Ksize;
    int lay_input_channel;

public:
    Layer lay;
    convlve2d() : lay() {}
    convlve2d(int input_size, int output_size, int Kernel_size)
        : lay(input_size, output_size, Kernel_size)
    {
        Ksize = Kernel_size;
        lay_input_channel = input_size;
    }

    Tensor conv(Tensor img, int Stride = 1, int Padding = 0)

    {
        stride = Stride;
        padding = Padding;
        int img_channel = img.image.size();
        int img_height = img.image[0].size();
        int img_width = img.image[0][0].size();

        // int input_channel = lay.w_b.weight.size();
        int output_channel = lay.w_b.weight[0].size();

        int output_height = (img_height + 2 * padding - Ksize) / stride + 1;
        int output_width = (img_width + 2 * padding - Ksize) / stride + 1;

        Tensor padding_img(img_channel, img_height + 2 * padding, img_width + 2 * padding);
        Tensor output(output_channel, output_height, output_width);

        Tensor pad_img = pade_img(padding, padding_img, img);

        output = perform_convolution(pad_img, output);
        return output;
    }
    void backward_conv(Tensor grad_layer, Tensor brev_idden_layer, float learning_rate)
    {
        int input_channel = lay.w_b.weight.size();       // Number of input channels
        int output_channel = lay.w_b.weight[0].size();   // Number of output channels
        int height = brev_idden_layer.image[0].size();   // Height of input image
        int width = brev_idden_layer.image[0][0].size(); // Width of input image
        int grad_height = grad_layer.image[0].size();    // Height of gradient (output gradient)
        int grad_width = grad_layer.image[0][0].size();  // Width of gradient (output gradient)
        int Kernel_size = lay.w_b.weight[0][0].size();   // Kernel size
#pragma omp parallel for

        // Loop over input channels
        for (int ch = 0; ch < input_channel; ++ch)
        {
            // Loop over output channels
            for (int out = 0; out < output_channel; ++out)
            {
                // Loop over the input feature map with correct bounds
                for (int h = 0; h <= height - Kernel_size; ++h)
                {
                    for (int w = 0; w <= width - Kernel_size; ++w)
                    {
                        // Update weights using the gradients
                        for (int kh = 0; kh < Kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < Kernel_size; ++kw)
                            {
                                lay.w_b.weight[ch][out][kh][kw] -= learning_rate *
                                                                   grad_layer.image[out][h][w] * brev_idden_layer.image[ch][h + kh][w + kw];
                            }
                        }
                    }
                }

                // Bias update
                float bias_grad = 0;
                for (int h = 0; h < grad_height; ++h)
                {
                    for (int w = 0; w < grad_width; ++w)
                    {
                        // Accumulate the gradient for bias
                        bias_grad += grad_layer.image[out][h][w];
                    }
                }
                // Apply learning rate to bias update
                lay.w_b.bais[out] -= learning_rate * bias_grad;
            }
        }
    }

    Tensor grid_conv(Tensor grad_layer)
    {
        int input_size = lay.w_b.weight.size();
        int output_size = lay.w_b.weight[0].size();
        int Kernel_size = lay.w_b.weight[0][0].size();
        int channel = grad_layer.image.size();
        int height = grad_layer.image[0].size();
        int width = grad_layer.image[0][0].size();

        Tensor padding(channel, height + 2 * (Kernel_size - 1), width + 2 * (Kernel_size - 1));
        Tensor pad_img = pade_img(Kernel_size - 1, padding, grad_layer);
        Weights flip_kernel(input_size, output_size, Kernel_size);
        Tensor output(input_size, height + Kernel_size - 1, width + Kernel_size - 1);

        for (int in = 0; in < input_size; ++in)
        {
            for (int out = 0; out < output_size; ++out)
            {
                for (int kh = 0; kh < Kernel_size; ++kh)
                {
                    for (int kw = 0; kw < Kernel_size; ++kw)
                    {
                        flip_kernel.weight[in][out][Kernel_size - kh - 1][Kernel_size - kw - 1] = lay.w_b.weight[in][out][kh][kw];
                    }
                }
            }
        }

        for (int out = 0; out < output_size; ++out)
        {

            for (int in = 0; in < input_size; ++in)
            {
                for (int h = 0; h < height + 2 * (Kernel_size - 1); ++h)
                {
                    for (int w = 0; w < width + 2 * (Kernel_size - 1); ++w)
                    {
                        float sum = 0;
                        for (int kh = 0; kh < Kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < Kernel_size; ++kw)
                            {
                                if ((h + kh) < pad_img.image[out].size() && (w + kw) < pad_img.image[out][0].size())
                                {
                                    sum += pad_img.image[out][h + kh][w + kw] * flip_kernel.weight[in][out][kh][kw];
                                }
                            }
                        }
                        if (h < output.image[in].size() && w < output.image[in][0].size())
                        {
                            output.image[in][h][w] += sum;
                        }
                    }
                }
            }
        }
        return output;
    }

private:
    // Apply padding to the image
    Tensor pade_img(int padding, Tensor padding_img, Tensor img)
    {
        int img_channel = img.image.size();
        int img_height = img.image[0].size();
        int img_width = img.image[0][0].size();

#pragma omp parallel for
        for (int ch = 0; ch < img_channel; ++ch)
        {
            for (int i = 0; i < img_height; ++i)
            {
                for (int j = 0; j < img_width; ++j)
                {
                    padding_img.image[ch][i + padding][j + padding] = img.image[ch][i][j];
                }
            }
        }
        return padding_img;
    }

    // Perform convolution
    Tensor perform_convolution(Tensor padding_img, Tensor output)
    {
        int channel = padding_img.image.size();
        int height = padding_img.image[0].size();
        int width = padding_img.image[0][0].size();
        try
        {
            // Ensure input channels match
            if (lay_input_channel != channel)
            {
                throw std::logic_error("Cannot convolve with different layer sizes.");
            }
        }
        catch (const std::logic_error &e)
        {
            std::cerr << e.what() << '\n';
        }

#pragma omp parallel for
        for (int ch = 0; ch < lay.w_b.weight.size(); ++ch)
        {
            for (int out = 0; out < lay.w_b.weight[0].size(); ++out)
            {
                for (int h = 0; h < height - Ksize + 1; h += stride)
                {
                    for (int w = 0; w < width - Ksize + 1; w += stride)
                    {
                        float output_pixel = 0;
                        for (int hk = 0; hk < Ksize; ++hk)
                        {
                            for (int wk = 0; wk < Ksize; ++wk)
                            {
                                output_pixel += lay.w_b.weight[ch][out][hk][wk] * padding_img.image[ch][h + hk][w + wk];
                            }
                        }
                        output.image[out][h / stride][w / stride] += output_pixel + lay.w_b.bais[out];
                    }
                }
            }
        }
        return output;
    }
};

class maxpool
{
public:
    maxpool()
    {
    }
    Tensor maxipool(Tensor image, int stride = 2, int kernel_size = 2)
    {
        int channel = image.image.size();
        int height = image.image[0].size();
        int width = image.image[0][0].size();
        int out_height = height / stride; // Truncate the extra row
        int out_width = width / stride;   // Truncate the extra column
        Tensor output(channel, out_height, out_width);
        try
        {
            if (kernel_size != stride)
            {
                throw logic_error("cant maxipool with that parameter ");
            }
        }
        catch (const std::logic_error &e)
        {
            std::cerr << "Error: " << e.what() << std::endl;
        }
#pragma omp parallel for collapse(2) schedule(static) // Collapsing and static scheduling

        for (int ch = 0; ch < channel; ++ch)
        {
            for (int h = 0; h < out_height * stride; h += stride)
            {
                for (int w = 0; w < out_width * stride; w += stride)
                {
                    float max_value = image.image[ch][h][w];
                    for (int hk = 0; hk < kernel_size; ++hk)
                    {
                        for (int wk = 0; wk < kernel_size; ++wk)
                        {
                            if (h + hk < height && w + wk < width)
                            {
                                max_value = max(max_value, image.image[ch][h + hk][w + wk]);
                            }
                        }
                    }
                    output.image[ch][h / stride][w / stride] = max_value;
                }
            }
        }
        return output;
    }
    Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, Tensor graident)
    {
        int channel = before_maxpool.image.size();
        int height = before_maxpool.image[0].size();
        int width = before_maxpool.image[0][0].size();
        Tensor output(channel, height, width);
        int Ksize = height / maxpool.image[0].size();
#pragma omp parallel for collapse(2)
        for (int ch = 0; ch < channel; ++ch)
        {
            for (int h; h < height; h += Ksize)
            {
                for (int w; w < width; w += Ksize)
                {
                    int position;
                    int num = maxpool.image[ch][h / Ksize][w / Ksize];
                    for (int hk = 0; hk < Ksize; ++hk)
                    {
                        for (int wk = 0; wk < Ksize; ++wk)
                        {
                            if (num == before_maxpool.image[ch][h + hk][w + wk])
                            {
                                output.image[ch][h + hk][w + wk] = graident.image[ch][h / Ksize][w / Ksize];
                            }
                            else
                            {
                                output.image[ch][h + hk][w + wk] = 0;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }
    Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, vector<float> grid)
    {
        Tensor graident = reversed_flatten(grid, maxpool);
        int channel = before_maxpool.image.size();
        int height = before_maxpool.image[0].size();
        int width = before_maxpool.image[0][0].size();
        Tensor output(channel, height, width);
        int Ksize = height / maxpool.image[0].size();
#pragma omp parallel for collapse(2)
        for (int ch = 0; ch < channel; ++ch)
        {
            for (int h; h < height; h += Ksize)
            {
                for (int w; w < width; w += Ksize)
                {
                    int position;
                    int num = maxpool.image[ch][h / Ksize][w / Ksize];
                    for (int hk = 0; hk < Ksize; ++hk)
                    {
                        for (int wk = 0; wk < Ksize; ++wk)
                        {
                            if (num == before_maxpool.image[ch][h + hk][w + wk])
                            {
                                output.image[ch][h + hk][w + wk] = graident.image[ch][h / Ksize][w / Ksize];
                            }
                            else
                            {
                                output.image[ch][h + hk][w + wk] = 0;
                            }
                        }
                    }
                }
            }
        }
        return output;
    }

private:
    Tensor reversed_flatten(vector<float> grad, Tensor maxpool)
    {

        int channel = maxpool.image.size();
        int height = maxpool.image[0].size();
        int width = maxpool.image[0][0].size();
        Tensor output(channel, height, width);
        for (int ch = 0; ch < channel; ++ch)
        {
            for (int h = 0; h < height; ++h)
            {
                for (int w = 0; w < width; ++w)
                {
                    output.image[ch][h][w] = grad[w + h * (width) + ch * (width * height)];
                }
            }
        }
        return output;
    }
};
#endif // conv