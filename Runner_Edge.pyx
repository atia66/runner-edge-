import numpy as np
cimport numpy as np  

from libc.stdint cimport int  
from libcpp.vector cimport vector

cdef extern from "Structs.h":

    cdef cppclass Tensor:
        Tensor()
        Tensor(int channels, int height, int width)
        void set_pixels(int channel_index, int height_index, int width_index, float value)
        float get_pixel(int channel_index, int height_index, int width_index)  # Declare get_pixel
        int get_chaanels()
        int get_height()
        int get_width()
        void show_tensor()

    cdef Tensor to_tensor(np.ndarray[np.float32_t, ndim=3] img):
        #mg = img.transpose(2, 0, 1)  # Change order from HWC to CHW
        cdef int channels = img.shape[0]
        cdef int height = img.shape[1]
        cdef int width = img.shape[2]
        cdef Tensor new_img = Tensor(channels, height, width)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    new_img.set_pixels(c, h, w, img[c, h, w])  
        return new_img


    cdef np.ndarray[np.float32_t, ndim=3] to_numpy(Tensor img):
        cdef channels =img.get_chaanels()
        cdef height =img.get_height()
        cdef width =img.get_width()
        cdef np.ndarray[np.float32_t, ndim=3] np_array = np.empty((channels, height, width), dtype=np.float32)
        for c in range(channels):
            for h in range(height):
                for w in range(width):
                    np_array[c, h, w] = img.get_pixel(c, h, w)  # Assuming get_pixel method exists
        return np_array


    cdef cppclass Fcweight:
        Fcweight()
        Fcweight(int input_size,int output_size)
        void set_weight(int index_input, int index_output, float value)
        void set_bias( int index_output, float value)

    cdef cppclass Weights:
        Weights()
        Weights(int input_size, int output_size, int Ksize)
        void set_weight(int index_input, int index_output, int Kh, int Kw, float value)
        void set_bias(int index_output, float value)

    cdef cppclass Image:
        Image()
        Image(int height, int width)
        void set_pixel(int height_index, int width_index, float value)


cdef extern from "conv.h":

    cdef cppclass Layer:
        Layer()
        Layer(int input_size,int output_size ,int Ksize)



    cdef cppclass Linear:
        Linear()
        Linear(int input_size,int output_size)
        vector[float] Fclayer(Tensor img)
        vector[float] Fclayer(vector[float] img)
        Fcweight get_weight_and_bias()
        void backward(vector[float] gradlayer,vector[float] input_layer, float learning_rate)
        void backward(vector[float] gradlayer,Tensor input_layer, float learning_rate)
        vector[float] grad( vector[float] gradlayer)
        void show_linear_weight()

    cdef cppclass convlve2d:
        convlve2d()
        convlve2d(int input_size, int output_size, int Kernel_size)
        Tensor conv(Tensor img, int Stride , int Padding )
        void backward_conv(Tensor gradlayer ,Tensor brev_hidden_layer,float learning_rate)
        Tensor grid_conv(Tensor gradlayer)

    cdef cppclass maxpool:
        maxpool()
        Tensor maxipool(Tensor image,int stride , int kernel_size )
        Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, vector[float] grid)
        Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, Tensor graident)


cdef extern from "Activation.h":
    cdef cppclass Activation_function:
        Activation_function()
        Tensor RelU(Tensor img)
        vector[float] softmax(vector[float] layer)
        vector[float] backward_softmax(vector[float] predict, vector[float] y)

cdef class PyLayer:

    cdef Layer lay_instance  
    def __cinit__(self, int input_size=1, int output_size=1, int Ksize=1):
        self.lay_instance = Layer(input_size, output_size, Ksize)

cdef class PyLinear:

    cdef Linear linear_instance
    def __cinit__(self, int input_size=1, int output_size=1):
        self.linear_instance = Linear(input_size, output_size)
    
    def fc_layer(self, np.ndarray[np.float32_t, ndim=1]  img_1d=None, np.ndarray[np.float32_t, ndim=3]  img_3d=None):
        cdef Tensor new_img
        cdef vector[float] result_vector
        cdef vector[float] vector_img
        cdef int size
        if img_3d is not None and img_3d.size > 0:
            # Handle 3D input
            new_img = to_tensor(img_3d)
            result_vector = self.linear_instance.Fclayer(new_img)  
            return np.array(result_vector, dtype=np.float32)
        
        elif img_1d is not None and img_1d.size > 0:
            # Handle 1D input
            size = img_1d.shape[0]
            vector_img.resize(size)
            for i in range(size):
                vector_img[i] = img_1d[i]
            result_vector = self.linear_instance.Fclayer(vector_img)  # Assuming Fclayer1d exists
            return np.array(result_vector, dtype=np.float32)
    ## REWRITE IT
    def Backward(self, np.ndarray[np.float32_t, ndim=1] grad_layer, np.ndarray[np.float32_t, ndim=1] input_layer_1d=None, np.ndarray[np.float32_t, ndim=3] input_layer_3d=None, float learning_rate=0.01):
        cdef int grad_size = grad_layer.shape[0]  
        cdef int input_size 
        cdef vector[float] Grad_layer
        Grad_layer.resize(grad_size)
        cdef vector[float] Input_vector
        cdef Tensor Input_Tensor

        for i in range(grad_size):
            Grad_layer[i] = float(grad_layer[i])
        
        if input_layer_1d is not None and input_layer_1d.size > 0:
            input_size = input_layer_1d.shape[0] 

            Input_vector.resize(input_size)

            for i in range(input_size):
                Input_vector[i] = float(input_layer_1d[i])  
            
            self.linear_instance.backward(Grad_layer, Input_vector, learning_rate)

        elif  input_layer_3d is not None and input_layer_3d.size > 0:
            Input_Tensor=to_tensor(input_layer_3d)
            self.linear_instance.backward(Grad_layer, Input_Tensor, learning_rate)

    def grad(self, np.ndarray[np.float32_t, ndim=1] grad_layer):
        
        cdef int grad_size = grad_layer.shape[0]
        cdef vector[float] Grad_layer
        Grad_layer.resize(grad_size)

        for i in range(grad_size):
            Grad_layer[i] = float(grad_layer[i])

        result_vector = self.linear_instance.grad(Grad_layer)
        return np.array(result_vector, dtype=np.float32)


cdef class PyConvlve2d:
    cdef convlve2d conv_instance

    def __cinit__(self, int input_size, int output_size, int Kernel_size):
        self.conv_instance = convlve2d(input_size, output_size, Kernel_size)

    def Conv(self, np.ndarray[np.float32_t, ndim=3] img, int Stride=1, int Padding=0):
        ##### normalize
        cdef img_channel=img.shape[0]
        cdef float img_max, img_min

        for c in range(img_channel):
            img_max = img[c].max()
            img_min = img[c].min()
            if img_max != img_min:
                img[c] = (img[c] - img_min) / (img_max - img_min)
        cdef Tensor Img=to_tensor(img)
        cdef Tensor output =self.conv_instance.conv(Img,Stride,Padding)
        return to_numpy(output)
    def Backward_conv(self,np.ndarray[np.float32_t, ndim=3] grad_layer ,np.ndarray[np.float32_t, ndim=3] brev_hidden_layer,float learning_rate):
        cdef Tensor Grad_layer=to_tensor(grad_layer)
        cdef Tensor Brev_hidden_layer=to_tensor(brev_hidden_layer)
        self.conv_instance.backward_conv(Grad_layer,Brev_hidden_layer,learning_rate)
    def Grid_conv(self,np.ndarray[np.float32_t, ndim=3] grad_layer):
        cdef Tensor Grad_layer=to_tensor(grad_layer)
        cdef Tensor output=self.conv_instance.grid_conv(Grad_layer)
        return to_numpy(output)

cdef class PyMaxpool:
    cdef maxpool maxpool_instance

    def __cinit__(self):
        maxpool_instance =maxpool()
    def Maxipool(self,np.ndarray[np.float32_t, ndim=3] img, int Stride=2, int Ksize=2):
        cdef Tensor Maxi=to_tensor(img)
        cdef Tensor output=self.maxpool_instance.maxipool(Maxi,Stride,Ksize)
        return to_numpy(output)
    # rewrite
    def Backward_maxpool(self,np.ndarray[np.float32_t, ndim=3] before_maxpool,np.ndarray[np.float32_t, ndim=3] maxpool,np.ndarray[np.float32_t, ndim=1] graident_layer_1d=None,np.ndarray[np.float32_t, ndim=3] graident_layer_3d=None):
        
        cdef Tensor Before_maxpool=to_tensor(before_maxpool)
        cdef Tensor Max=to_tensor(maxpool)
        cdef Tensor Graident_layer
        cdef Tensor output
        cdef int size
        cdef vector[float] Graident_Layer
        if graident_layer_3d is not None and graident_layer_3d.size > 0:
            Graident_layer=to_tensor(graident_layer_3d)
            output =self.maxpool_instance.backward_maxpool(Before_maxpool,Max,Graident_layer)
            return to_numpy(output)
        elif graident_layer_1d is not None and graident_layer_1d.size > 0:
            size=graident_layer_1d.shape[0]
            Graident_Layer.resize(size)
            for c in range(size):
                Graident_Layer[c]=graident_layer_1d[c]
                output =self.maxpool_instance.backward_maxpool(Before_maxpool,Max,Graident_Layer)
                return to_numpy(output)

cdef class PyActivation_Function:
    cdef Activation_function activation_instance

    def __cinit__(self):
        self.activation_instance = Activation_function()

    def RelU(self,np.ndarray[np.float32_t, ndim=3] img):
        cdef Tensor Img=to_tensor(img)
        cdef Tensor result=self.activation_instance.RelU(Img)
        return to_numpy(result)

    def Softmax(self,np.ndarray[np.float32_t, ndim=1] vec):
        cdef int vec_size=vec.shape[0]
        cdef vector[float] Vec
        Vec.resize(vec_size) 
        
        for i in range(vec_size):
            Vec[i]=vec[i]
        return np.array(self.activation_instance.softmax(Vec),dtype=np.float32)
    def backward_softmax(self,np.ndarray[np.float32_t, ndim=1] y_predict,np.ndarray[np.float32_t, ndim=1] y_true):
        cdef int size
        cdef vector[float] predict
        cdef vector[float] Y_true
        if y_predict.shape[0]==y_true.shape[0]:
            size=y_predict.shape[0]
            Y_true.resize(size)
            predict.resize(size)

            for i in range(size):
                predict[i]=y_predict[i]

            for i in range(size):
                Y_true[i]=y_true[i]
            return np.array(self.activation_instance.backward_softmax(predict,Y_true),dtype=np.float32)
        else:
            raise("Error in shape")

