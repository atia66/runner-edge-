#ifndef MY_STRUCTS_H
#define MY_STRUCTS_H
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <cmath>
#include <omp.h>
using namespace std;

struct Tensor
{
    vector<vector<vector<float>>> image;
    Tensor() {};

    Tensor(int channels, int height, int width)
    {
        image.resize(channels, vector<vector<float>>(height, vector<float>(width)));
    };

    void set_pixels(int channel_index, int height_index, int width_index, float value)
    {
        image[channel_index][height_index][width_index] = value;
    }
    float get_pixel(int channel, int height, int width)
    {
        return image[channel][height][width]; // Adjust based on your data structure
    }
    int get_chaanels()
    {
        return image.size(); // Adjust based on your data structure
    }
    int get_height()
    {
        return  image[0].size(); // Adjust based on your data structure
    }
    int get_width()
    {
        return  image[0][0].size(); // Adjust based on your data structure
    }
    void show_tensor()
    {
        std::cout << "Tensor image data: " << std::endl;
        for (int channel = 0; channel < image.size(); ++channel)
        {
            for (int height = 0; height < image[0].size(); ++height)
            {
                for (int width = 0; width < image[0][0].size(); ++width)
                {
                    std::cout << image[channel][height][width] << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
    }
};

struct Fcweight
{

    vector<vector<float>> weight;
    vector<float> bais;
    Fcweight()
    {
    }
    Fcweight(int input_size, int output_size)
    {
        weight = vector<vector<float>>(output_size, vector<float>(input_size));
        bais = vector<float>(output_size);
    }
    void set_weight(int index_input, int index_output, float value)
    {
        weight[index_output][index_input] = value; // Corrected indexing
    }
    void set_bias(int index_output, float value)
    {
        bais[index_output] = value; // Setting bias value
    }

};

struct Image
{
    vector<vector<float>> img;
    Image()
    {
    }
    Image(int height, int width)
    {
        img = vector<vector<float>>(height, vector<float>(width));
    }
    void set_pixel(int height, int width, float value)
    {
        img[height][width] = value;
    }
    void show_image(){

    }
};

struct Weights
{
    vector<vector<vector<vector<float>>>> weight;
    vector<float> bais;
    Weights()
    {
    }
    Weights(int input_size, int output_size, int Ksize)
    {
        weight = vector<vector<vector<vector<float>>>>(input_size,
vector<vector<vector<float>>>(output_size,
vector<vector<float>>(Ksize,
vector<float>(Ksize))));
        bais = vector<float>(output_size);
    }
    void set_weight(int index_input, int index_output, int kernel_height_index, int kernel_width_index, float value)
    {
        weight[index_output][index_input][kernel_height_index][kernel_width_index] = value; // Corrected indexing
    }
    void set_bias(int index_output, float value)
    {
        bais[index_output] = value; // Setting bias value
    }
};

#endif // MY_STRUCTS_H
