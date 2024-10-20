#include <iostream>
# include <vector>
# include <random>
#include <chrono>
# include <cmath>
# include <omp.h>

using namespace std;

struct Image{
    vector<vector<int16_t>> img;
    Image()
    {
    }
    Image(int16_t height, int16_t width)
    {
        img= vector<vector<int16_t>>(height, vector<int16_t>(width));
    }
};

struct Fcweight{

    vector<vector<float>> weight;
    vector<float> bais;
    Fcweight(){
        
    }
    Fcweight(int16_t input_size, int16_t output_size){
        weight = vector<vector<float>>(output_size,vector<float>(input_size));
        bais = vector<float>(output_size);
    }
};

struct Weights
{
    vector<vector<vector<vector<float>>>> weight;
    vector < float > bais;
    Weights(){
    }
    Weights(int16_t input_size, int16_t output_size, int16_t Ksize)
    {
        weight = vector<vector<vector<vector<float>>>>(input_size,
        vector<vector<vector<float>>>(output_size,
        vector<vector<float>>(Ksize,
        vector<float>(Ksize))));
        bais = vector < float > (output_size);
    }
};

struct Tensor
{
    vector<vector<vector<int16_t>>> image;
    Tensor(){};
    Tensor(int16_t channels, int16_t height, int16_t width)
    {
        image = vector<vector<vector<int16_t>>>(channels, vector<vector<int16_t>>(height, vector<int16_t>(width)));
    };
};

// ######################################################################################

class Activation_Function
{
    public:
    Activation_Function()
    {

    }
    Tensor  RelU(Tensor image)
    {
        int16_t channel = image.image.size();
        int16_t height = image.image[0].size();
        int16_t width = image.image[0][0].size();
        Tensor output (channel, height, width);
        #pragma omp parallel for collapse(3)
        for (int16_t ch = 0; ch < channel; ++ch)
        {
            for (int16_t h = 0; h < height; ++h)
            {
                for (int16_t w = 0; w < width; ++w)
                {
                    output.image[ch][h][w] = (image.image[ch][h][w] >= 0) ? image.image[ch][h][w] : 0;
                };
            }
        }
        return output;
    }
    vector<float> softmax(vector<float> layer)
    {
        int16_t max = layer[0];
        float sum = 0;
        #pragma omp parallel for
        for (int16_t i = 1; i < layer.size();++i){
            if (max < layer[i]){
                max = layer[i];
            }
        }
        #pragma omp parallel for
        for (int16_t i = 0; i < layer.size(); ++i)
        {
            layer[i] -= max;
            sum += layer[i];
        }
        #pragma omp parallel for
        for (int16_t i = 0; i < layer.size(); ++i)
        {
            layer[i] /= sum;
        }
        return layer;
    }
    vector<float> backward_softmax(vector<float> predict, vector<float> y)
    {
        vector<float> output(predict.size());
        for (int16_t i = 0; i < predict.size(); ++i)
        {
            output[i] = y[i] - predict[i];
        }
        return output;
    }

};

class Layer{
    public :
    Weights w_b;
    bool toggle = true;
    Layer() : w_b() {};
    Layer(int16_t input_size, int16_t output_size, int16_t Ksize)
        : w_b(input_size, output_size, Ksize)
    {
        initialize_weights_and_biases(input_size, output_size, Ksize);
    }
    private:
        void initialize_weights_and_biases(int16_t input_size, int16_t output_size, int16_t Ksize)
        {
            std::random_device rd;
            std::mt19937 eng(rd());
            std::uniform_real_distribution<float> distr(-0.1, 0.1);
            #pragma omp parallel for
            for (int16_t i = 0; i < input_size; i++)
            {
                for (int16_t j = 0; j < output_size; j++)
                {
                    for (int16_t ki = 0; ki < Ksize; ++ki)
                    {
                        for (int16_t kj = 0; kj < Ksize; ++kj)
                        {
                            w_b.weight[i][j][ki][kj] = distr(eng);
                        }
                    }
                }
            }
            #pragma omp parallel for
            for (int16_t j = 0; j < output_size; j++)
            {
                w_b.bais[j] = distr(eng);
            }
        }
};

class Linear
{
int16_t output_size;
public:
    Fcweight w_b;
    Linear(): w_b() {
    };
    Linear(int16_t input_size, int16_t outputsize) : w_b(input_size, outputsize) 
    {
        output_size = outputsize;
        initialize_weights_and_biases(input_size, output_size);
    }
    vector<float> Fclayer(Tensor img)
    {
        vector<float>flatten =apply_flatten(img);
        vector < float>output(output_size);
        #pragma omp parallel for
        for (int16_t out = 0; out < output_size;++out){
            for (int16_t idx = 0; idx < flatten.size();++idx){
                output[out] += flatten[idx] * w_b.weight[out][idx];
            }
            output[out] += w_b.bais[out]; 
        }
        return output;
    }
    vector<float> Fclayer(vector<float> img)
    {
        vector<float> output  (output_size);
#pragma omp parallel for
        for (int16_t out = 0; out < output_size; ++out)
        {
            for (int16_t idx = 0; idx < img.size(); ++idx)
            {
                output[out] += img[idx] * w_b.weight[out][idx] ;
            }
            output[out] += w_b.bais[out];
        }
        return output;
    }
    void backward(vector<float> gradlayer, vector<float> input, float learning_rate)
    {
        int16_t out_layers = w_b.weight.size();
        int16_t in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int16_t out = 0; out < out_layers; ++out)
        {
            for (int16_t in = 0; in < in_layers; ++in)
            {
                w_b.weight[out][in] -= learning_rate * gradlayer[out]*input[in];
            }
                w_b.bais[out] -= learning_rate * gradlayer[out];
        }
    }
    void backward(vector<float> gradlayer, Tensor Input, float learning_rate)
    {
        vector<float> input =apply_flatten(Input);
        int16_t out_layers = w_b.weight.size();
        int16_t in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int16_t out = 0; out < out_layers; ++out)
        {
            for (int16_t in = 0; in < in_layers; ++in)
            {
                w_b.weight[out][in] -= learning_rate * gradlayer[out] * input[in];
            }
            w_b.bais[out] -= learning_rate * gradlayer[out];
        }
    }
    vector<float> grad(vector<float> gradlayer)
    {
        vector<float> grad_input(w_b.weight[0].size(), 0.0f);
        int16_t out_layers = w_b.weight.size();
        int16_t in_layers = w_b.weight[0].size();
#pragma omp parallel for

        for (int16_t j = 0; j < in_layers; j++)
        {
            for (int16_t i = 0; i < out_layers; i++)
            {
                grad_input[j] += w_b.weight[i][j] * gradlayer[i];
            }
        }
        return grad_input;
    }

    private:

    void initialize_weights_and_biases(int16_t input_size, int16_t output_size)
    {
        std::random_device rd;
        std::mt19937 eng(rd());
        std::uniform_real_distribution<float> distr(-0.1, 0.1);
        #pragma omp parallel for
        for (int16_t i = 0; i < output_size; i++)
        {
            for (int16_t j = 0; j < input_size; j++)
            {
                        w_b.weight[i][j] = distr(eng);
                    }
                    w_b.bais[i] = distr(eng);
                }
    }

    vector<float> apply_flatten(Tensor img)
    {
        int16_t channel = img.image.size();
        int16_t height = img.image[0].size();
        int16_t width = img.image[0][0].size();
        vector<float> flatten(channel * height * width);
        #pragma omp parallel for
        for (int16_t ch = 0; ch < channel; ++ch)
        {
            for (int16_t h = 0; h < height; ++h)
            {
                for (int16_t w = 0; w < width; ++w)
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
    int16_t stride;
    int16_t padding;
    int16_t Ksize;
    int16_t lay_input_channel;

public:
    Layer lay;
    convlve2d():lay(){}
    convlve2d(int16_t input_size, int16_t output_size, int16_t Kernel_size)
        : lay(input_size, output_size, Kernel_size)
    {
        Ksize = Kernel_size;
        lay_input_channel=input_size;
    }

    Tensor conv(Tensor img, int16_t Stride = 1, int16_t Padding = 0)

    {
        stride = Stride;
        padding = Padding;
        int16_t img_channel = img.image.size();
        int16_t img_height = img.image[0].size();
        int16_t img_width = img.image[0][0].size();

        // int16_t input_channel = lay.w_b.weight.size();
        int16_t output_channel = lay.w_b.weight[0].size();

        int16_t output_height = (img_height + 2 * padding - Ksize) / stride + 1;
        int16_t output_width = (img_width + 2 * padding - Ksize) / stride + 1;

        Tensor padding_img(img_channel, img_height + 2 * padding, img_width + 2 * padding);
        Tensor output(output_channel, output_height, output_width);

        Tensor pad_img = pade_img(padding, padding_img, img);

        output=perform_convolution(pad_img, output);
        return output;
    }
    void backward_conv(Tensor grad_layer, Tensor brev_idden_layer, float learning_rate)
    {
        int16_t input_channel = lay.w_b.weight.size();       // Number of input channels
        int16_t output_channel = lay.w_b.weight[0].size();   // Number of output channels
        int16_t height = brev_idden_layer.image[0].size();   // Height of input image
        int16_t width = brev_idden_layer.image[0][0].size(); // Width of input image
        int16_t grad_height = grad_layer.image[0].size();    // Height of gradient (output gradient)
        int16_t grad_width = grad_layer.image[0][0].size();  // Width of gradient (output gradient)
        int16_t Kernel_size = lay.w_b.weight[0][0].size();   // Kernel size
#pragma omp parallel for

        // Loop over input channels
        for (int16_t ch = 0; ch < input_channel; ++ch)
        {
            // Loop over output channels
            for (int16_t out = 0; out < output_channel; ++out)
            {
                // Loop over the input feature map with correct bounds
                for (int16_t h = 0; h <= height - Kernel_size; ++h)
                {
                    for (int16_t w = 0; w <= width - Kernel_size; ++w)
                    {
                        // Update weights using the gradients
                        for (int16_t kh = 0; kh < Kernel_size; ++kh)
                        {
                            for (int16_t kw = 0; kw < Kernel_size; ++kw)
                            {
                                lay.w_b.weight[ch][out][kh][kw] -= learning_rate *
                                                                   grad_layer.image[out][h][w] * brev_idden_layer.image[ch][h + kh][w + kw];
                            }
                        }
                    }
                }

                // Bias update
                float bias_grad = 0;
                for (int16_t h = 0; h < grad_height; ++h)
                {
                    for (int16_t w = 0; w < grad_width; ++w)
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

    Tensor grid_conv(Tensor grad_layer){
        int16_t input_size = lay.w_b.weight.size();
        int16_t output_size = lay.w_b.weight[0].size();
        int16_t Kernel_size = lay.w_b.weight[0][0].size();
        int16_t channel = grad_layer.image.size();
        int16_t height = grad_layer.image[0].size();
        int16_t width = grad_layer.image[0][0].size();

        Tensor padding(channel, height + 2 * (Kernel_size - 1), width + 2 * (Kernel_size - 1));
        Tensor pad_img = pade_img(Kernel_size - 1, padding, grad_layer);
        Weights flip_kernel(input_size, output_size, Kernel_size);
        Tensor output(input_size, height + Kernel_size - 1, width + Kernel_size - 1);

        for (int16_t in=0; in<input_size;++in){
            for (int16_t out = 0; out < output_size;++out)
            {
                for (int16_t kh = 0; kh < Kernel_size; ++kh)
                {
                    for (int16_t kw = 0; kw < Kernel_size; ++kw)
                    {
                        flip_kernel.weight[in][out][Kernel_size - kh - 1][Kernel_size - kw - 1] = lay.w_b.weight[in][out][kh][kw];
                    }
                }
            }
        }
        
        for (int16_t out = 0; out < output_size;++out){

            for (int16_t in = 0; in < input_size; ++in)
            {
                for (int16_t h = 0; h < height + 2 * (Kernel_size - 1); ++h)
                {
                    for (int16_t w = 0; w < width + 2 * (Kernel_size - 1); ++w)
                    {
                        float sum=0;
                        for (int16_t kh = 0; kh < Kernel_size; ++kh)
                        {
                            for (int16_t kw = 0; kw < Kernel_size; ++kw)
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
    Tensor pade_img(int16_t padding, Tensor padding_img, Tensor img)
    {
        int16_t img_channel = img.image.size();
        int16_t img_height = img.image[0].size();
        int16_t img_width = img.image[0][0].size();

#pragma omp parallel for
        for (int16_t ch = 0; ch < img_channel; ++ch)
        {
            for (int16_t i = 0; i < img_height; ++i)
            {
                for (int16_t j = 0; j < img_width; ++j)
                {
                    padding_img.image[ch][i + padding][j + padding] = img.image[ch][i][j];
                }
            }
        }
        return padding_img;
    }

    // Perform convolution
    Tensor  perform_convolution(Tensor padding_img, Tensor output)
    {
        int16_t channel = padding_img.image.size();
        int16_t height = padding_img.image[0].size();
        int16_t width = padding_img.image[0][0].size();
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
        for (int16_t ch = 0; ch < lay.w_b.weight.size(); ++ch)
        {
            for (int16_t out = 0; out < lay.w_b.weight[0].size(); ++out)
            {
                for (int16_t h = 0; h < height - Ksize + 1; h += stride)
                {
                    for (int16_t w = 0; w < width - Ksize + 1; w += stride)
                    {
                        float output_pixel = 0;
                        for (int16_t hk = 0; hk < Ksize; ++hk)
                        {
                            for (int16_t wk = 0; wk < Ksize; ++wk)
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

class maxpool{
public:
    maxpool(){

    }
    Tensor maxipool(Tensor image, int16_t stride = 2, int16_t kernel_size = 2)
    {
        int16_t channel = image.image.size();
        int16_t height = image.image[0].size();
        int16_t width = image.image[0][0].size();
        int16_t out_height = height / stride; // Truncate the extra row
        int16_t out_width = width / stride;   // Truncate the extra column
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

        for (int16_t ch = 0; ch < channel; ++ch)
        {
            for (int16_t h = 0; h < out_height*stride; h += stride)
            {
                for (int16_t w = 0; w < out_width*stride; w += stride)
                {
                    int16_t max_value = image.image[ch][h][w]; 
                    for (int16_t hk = 0; hk < kernel_size; ++hk)
                    {
                        for (int16_t wk = 0; wk < kernel_size; ++wk)
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
    Tensor backward_maxpool(Tensor before_maxpool ,Tensor maxpool ,Tensor graident ){
        int16_t channel=before_maxpool.image.size();
        int16_t height = before_maxpool.image[0].size();
        int16_t width = before_maxpool.image[0][0].size();
        Tensor output(channel,height,width);
        int16_t Ksize = height / maxpool.image[0].size();
#pragma omp parallel for collapse(2)
        for (int16_t ch = 0; ch < channel; ++ch)
        {
            for (int16_t h; h < height; h += Ksize)
            {
                for (int16_t w; w < width; w += Ksize)
                {
                    int16_t position;
                    int16_t num = maxpool.image[ch][h / Ksize][w / Ksize];
                    for (int16_t hk = 0; hk < Ksize; ++hk)
                    {
                        for (int16_t wk = 0; wk < Ksize; ++wk)
                        {
                            if(num==before_maxpool.image[ch][h+hk][w+wk]){
                                output.image[ch][h + hk][w + wk] = graident.image[ch][h / Ksize][w / Ksize];
                            }
                            else{
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
        int16_t channel = before_maxpool.image.size();
        int16_t height = before_maxpool.image[0].size();
        int16_t width = before_maxpool.image[0][0].size();
        Tensor output(channel, height, width);
        int16_t Ksize = height / maxpool.image[0].size();
#pragma omp parallel for collapse(2)
        for (int16_t ch = 0; ch < channel; ++ch)
        {
            for (int16_t h; h < height; h += Ksize)
            {
                for (int16_t w; w < width; w += Ksize)
                {
                    int16_t position;
                    int16_t num = maxpool.image[ch][h / Ksize][w / Ksize];
                    for (int16_t hk = 0; hk < Ksize; ++hk)
                    {
                        for (int16_t wk = 0; wk < Ksize; ++wk)
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
        Tensor reversed_flatten(vector<float> grad, Tensor maxpool){

            int16_t channel=maxpool.image.size();
            int16_t height = maxpool.image[0].size();
            int16_t width = maxpool.image[0][0].size();
            Tensor output(channel, height, width);
            for (int ch = 0; ch < channel;++ch)
            {
                for (int h = 0; h < height;++h)
                {
                    for (int w = 0; w < width;++w)
                    {
                        output.image[ch][h][w] = grad[w  +h * (width) + ch * (width * height)];
                    }
                }
            }
            return output;
        }
};

class LossFunction{
    public:
    float CrossEntropy(vector<float> predicted, vector<float>y) {
        float sum = 0.0;

        for (int i = 0; i < predicted.size();++i){
            sum += -y[i] * log2(predicted[i]);
        }
        return sum;
    };
};

// int main()
// {
//     auto start = chrono::high_resolution_clock::now();

//     // Forward pass
//     // ##############################################################################
//     Tensor img(3, 64, 64); // Input image with 3 channels (RGB) and 128x128 size
//     maxpool maxi;
//     Activation_Function RelU;
//     cout << "img  " << img.image.size() << " " << img.image[0].size() << " " << img.image[0][0].size() << endl;
//     convlve2d co(3, 10, 3);
//     Tensor conv1 = co.conv(img);
//     Tensor max1 = maxi.maxipool(conv1);

//     convlve2d co2(10, 10, 3);
//     Tensor conv2 = co2.conv(max1);
//     Tensor max2 = maxi.maxipool(conv2);

//     convlve2d co3(10, 10, 3);
//     Tensor conv3 = co3.conv(max2);
//     Tensor max3 = maxi.maxipool(conv3);
//     // Fully connected layer 1 (FC1)
//     Linear fclay1(max3.image.size() * max3.image[0].size() * max3.image[0][0].size(), 50); // Input: flattened max3, Output: 50 units
//     vector<float> fcl1 = fclay1.Fclayer(max3); // Forward pass through FC1
//     // cout << "fcl1  " << fcl1.size() << " " << " " << endl;

//     // // Fully connected layer 2 (FC2)
//     Linear fclay2(50, 10);                     // Input: 50 units from FC1, Output: 10 (for classification)
//     vector<float> fcl2 = fclay2.Fclayer(fcl1); // Forward pass through FC2
//     // cout << "fcl2  " << fcl2.size() << " " << " " << endl;

//     vector<float> y(10); // Ground truth labels (one-hot encoded for 10 classes)
//     y[1] = 1;            // Let's assume the correct class is class 1
//     Activation_Function softmax;
//     vector<float> final1 = softmax.softmax(fcl2); // Apply softmax to the output of FC2

//     LossFunction loss;
//     float losses = loss.CrossEntropy(final1, y); // Calculate cross-entropy loss

//     vector<float> error_layer = softmax.backward_softmax(final1, y);
//     // cout << "error_layer size: " << error_layer.size() << endl;

//     fclay2.backward(error_layer, fcl1, 0.01);

//     vector<float> back2 = fclay2.grad(error_layer);
//     // cout << "fcl2_backward size: " << back2.size() << endl;

//     fclay1.backward(back2, max2, 0.01);

//     vector<float> back1 = fclay1.grad(back2);
//     // cout << "fcl1_backward size: " << back1.size() << endl;

//     Tensor backpool1 = maxi.backward_maxpool(conv3, max3,back1);

//     co3.backward_conv(backpool1, max3,0.01);
//     Tensor backconv1 = co3.grid_conv(backpool1);

//     Tensor backpool2 = maxi.backward_maxpool(conv2,max2, backconv1);

//     co2.backward_conv(backpool2, max2, 0.01);
//     Tensor backconv2 = co3.grid_conv(backpool2);

//     Tensor backpool3 = maxi.backward_maxpool(conv1, max1, backconv2);

//     co.backward_conv(backpool3,img, 0.01);
    
//     auto end = chrono::high_resolution_clock::now();
//     auto duration = chrono::duration_cast<chrono::milliseconds>(end - start);
//     std::cout << "Elapsed time for image size: " << duration.count() << " milliseconds\n";
// }