#ifndef MY_ACTIVATION_FUNCTION_H
#define MY_ACTIVATION_FUNCTION_H
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
#include "Structs.h"
using namespace std;
class Activation_function{
    public:
    Activation_function(){}

        Tensor RelU(Tensor image)
    {
        int channel = image.image.size();
        int height = image.image[0].size();
        int width = image.image[0][0].size();
        Tensor output(channel, height, width);
#pragma omp parallel for collapse(3)
    for (int ch = 0; ch < channel; ++ch)
    {
        for (int h = 0; h < height; ++h)
        {
            for (int w = 0; w < width; ++w)
            {
                output.image[ch][h][w] = (image.image[ch][h][w] >= 0) ? image.image[ch][h][w] : 0;
            };
        }
    }
    return output;
};
vector<float> softmax(vector<float> layer)
{
    int max = layer[0];
    float sum = 0;
#pragma omp parallel for
    for (int i = 1; i < layer.size(); ++i)
    {
        if (max < layer[i])
        {
            max = layer[i];
        }
    }
#pragma omp parallel for
    for (int i = 0; i < layer.size(); ++i)
    {
        layer[i] -= max;
        sum += layer[i];
    }
#pragma omp parallel for
    for (int i = 0; i < layer.size(); ++i)
    {
        layer[i] /= sum;
    }
    return layer;
};
vector<float> backward_softmax(vector<float> predict, vector<float> y)
{
    vector<float> output(predict.size());
    for (int i = 0; i < predict.size(); ++i)
    {
        output[i] = y[i] - predict[i];
    }
    return output;
};
};

#endif // MY_ACTIVATION_FUNCTION_Huj