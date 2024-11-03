import numpy as np
cimport numpy as np  

from libc.stdint cimport int  
from libcpp.vector cimport vector

cdef extern from "Structs.h":
    cdef cppclass Batch:
        Batch()
        Batch (int batch_size)
        int size()
        void add_Tensor(Tensor img)
        Tensor get_tensor(int idx)


    cdef cppclass DataLoader:
        DataLoader()
        DataLoader(int batch_size,vector[Tensor] img, vector[vector[float]] labels )
        vector[vector[float]] get_label(int index)
        Batch get_batch(int idx)
        int size()



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
        float get_weight(int index_input, int index_output)
        float get_bias( int index_output)

    cdef cppclass Weights:
        Weights()
        Weights(int input_size, int output_size, int Ksize)
        void set_weight(int index_input, int index_output, int Kh, int Kw, float value)
        void set_bias(int index_output, float value)
        float get_weight(int index_input, int index_output, int Kh, int Kw)
        float get_bias( int index_output)
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
        void backward1d(vector[float] gradlayer,vector[float] input_layer, float learning_rate)
        void backward3d(vector[float] gradlayer,Tensor input_layer, float learning_rate)
        vector[float] grad( vector[float] gradlayer)
        vector[vector[float]] get_weight() 
        vector[float] get_bias()
        bint Training_state()


    cdef cppclass convlve2d:
        convlve2d()
        convlve2d(int input_size, int output_size, int Kernel_size, int Stride , int Padding )
        Tensor conv(Tensor img)
        #Batch conv(Batch images)
        void backward_conv3d(Tensor gradlayer ,Tensor brev_hidden_layer,float learning_rate)
        void backward_conv1d(vector[float] grad_layer, Tensor brev_idden_layer, float learning_rate)

        Tensor grid_conv1d(vector[float] Grad_layer)
        Tensor grid_conv3d(Tensor gradlayer)

        vector[vector[vector[vector[float]]]] get_weight()
        vector[float] get_bias()
        bint Training_state()

    cdef cppclass maxpool:
        maxpool()
        maxpool(int stride , int kernel_size )
        Tensor maxipool(Tensor image)
        Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, vector[float] grid)
        Tensor backward_maxpool(Tensor before_maxpool, Tensor maxpool, Tensor graident)


cdef extern from "Activation.h":

    cdef cppclass RelU:
        RelU()
        Tensor relu3d(Tensor img)
        Tensor Grid_RelU3d(Tensor image)
        vector[vector[float]] relu2d(vector[vector[float]] image)
        vector[vector[float]] Grid_RelU2d(vector[vector[float]] image)

    cdef cppclass Softmax:
        Softmax()
        vector[float] softmax(vector[float] layer)
        vector[float] backward_softmax(vector[float] predict, vector[float] y)




#cdef class PyDataLoader:
 #  cdef Batch batch_instance
  #  cdef Tensor Tensor_instance
   # cdef int num_images
#    #cdef int current_idx
 #   cdef int batch_size
  #  cdef int num_batches  
   # cdef vector[Tensor] Dataset
    #cdef vector[vector[float]] Labels
#    cdef vector[vector[float]] lab
 #   cdef list batches 
  #  cdef np.ndarray numpy_array
   # cdef int channels
#    cdef int height
 #   cdef int width
  #  def __cinit__(self,int batch_size,np.ndarray[np.float32_t, ndim=4] dataset,np.ndarray[np.float32_t, ndim=2] labels):
#
 ##      self.channels,self.height,self.width=dataset.shape[1],dataset.shape[2],dataset.shape[3]
   #     self.current_idx = 0
#
 #       self.Dataset.resize(self.num_images)
  #      self.Labels.resize(self.num_images)
   #     
    #    for i in range(self.num_images):
    #      self.Dataset[i] = to_tensor(reshaped_image)  
     #       self.Labels[i] = labels[i]
      #  
       # self.DataLoader_instance =DataLoader(self.batch_size,self.Dataset,self.Labels)
        #self.num_batches = self.DataLoader_instance.size()
        #self.current_idx = 0

#    def __iter__(self):
 #       self.current_idx = 0
  #      return self
#
#    def __next__(self):
 #       if self.current_idx >= self.num_batches:
  #          raise StopIteration
   #     batch_instance = Batch(self.DataLoader_instance.get_batch(self.current_idx ).size())
    #    batch_instance = self.DataLoader_instance.get_batch(self.current_idx )
     #  lab = self.DataLoader_instance.get_label(self.current_idx )
      #  
#        self.batches = [] 
  #      for i in range(batch_instance.size()):
 # #          tensor_instance = batch_instance.get_tensor(i)
  #          self.numpy_array = to_numpy(tensor_instance)  
   #         self.batches.append(self.numpy_array)  
    #    
     #   self.current_idx += 1
      #  
       # return lab, self.batches 

cdef class PyLayer:

    cdef Layer lay_instance
    def __cinit__(self, int input_size=1, int output_size=1, int Ksize=1):
        self.lay_instance = Layer(input_size, output_size, Ksize)
cdef class PyLinear:
    cdef Fcweight fcweight_instance
    cdef Linear linear_instance
    def __cinit__(self, int input_size=1, int output_size=1):
        self.linear_instance = Linear(input_size, output_size)

    def __call__(self,  img):
        cdef vector[float] result_vector
        cdef vector[float] vector_img

        cdef int size

        if img.ndim == 3 and img.size > 0:
        # Handle 3D input
            new_img = to_tensor(img)


            result_vector = self.linear_instance.Fclayer(new_img)  # Convert to tensor if needed
            return np.array(result_vector, dtype=np.float32)
    
        elif img.ndim == 1 and img.size > 0:


            size = img.shape[0]
            vector_img.resize(size)
            for i in range(size):
                vector_img[i] = img[i]
            result_vector = self.linear_instance.Fclayer(vector_img)  # Assuming Fclayer1d exists
            return np.array(result_vector, dtype=np.float32)


    def Backward1d(self, np.ndarray[np.float32_t, ndim=1] grad_layer, np.ndarray[np.float32_t,ndim=1] input_layer, float learning_rate=0.01): 
        cdef int grad_size = grad_layer.shape[0]  
        cdef int input_size 
        cdef vector[float] Grad_layer
        Grad_layer.resize(grad_size)
        cdef vector[float] Input_vector
        cdef Tensor Input_Tensor

        for i in range(grad_size):
            Grad_layer[i] = float(grad_layer[i])
        
        input_size = input_layer.shape[0] 
        Input_vector.resize(input_size)
        for i in range(input_size):
            Input_vector[i] = float(input_layer[i])  
            
        self.linear_instance.backward1d(Grad_layer, Input_vector, learning_rate)
    def Backward3d(self, np.ndarray[np.float32_t, ndim=1] grad_layer, np.ndarray[np.float32_t,ndim=3] input_layer, float learning_rate=0.01): #####
        cdef int grad_size = grad_layer.shape[0]  
        cdef int input_size 
        cdef vector[float] Grad_layer
        Grad_layer.resize(grad_size)
        cdef vector[float] Input_vector
        cdef Tensor Input_Tensor
        for i in range(grad_size):
            Grad_layer[i] = float(grad_layer[i])
        Input_Tensor=to_tensor(input_layer)
        self.linear_instance.backward3d(Grad_layer, Input_Tensor, learning_rate)


    def grad(self, np.ndarray[np.float32_t, ndim=1] grad_layer):
        
        cdef int grad_size = grad_layer.shape[0]
        cdef vector[float] Grad_layer
        Grad_layer.resize(grad_size)

        for i in range(grad_size):
            Grad_layer[i] = float(grad_layer[i])

        result_vector = self.linear_instance.grad(Grad_layer)
        return np.array(result_vector, dtype=np.float32)

    def get_weight(self):
        cdef vector[vector[float]] cpp_weights = self.linear_instance.get_weight()
        return [[cpp_weights[i][j] for j in range(cpp_weights[i].size())] for i in range(cpp_weights.size())]

    def get_bias(self):
        cdef vector[float] cpp_bias = self.linear_instance.get_bias()
        return [cpp_bias[i] for i in range(cpp_bias.size())]
    def Training_state(self):
        return self.linear_instance.Training_state()


cdef class PyConvlve2d:
    cdef convlve2d conv_instance

    def __cinit__(self, int input_size, int output_size, int Kernel_size,int stride=1,int Padding=0):
        self.conv_instance = convlve2d(input_size, output_size, Kernel_size,stride,Padding)

    def __call__(self,img):
        return self.Conv(img)
    def Conv(self, np.ndarray[np.float32_t, ndim=3] img):
        cdef img_channel=img.shape[0]
        cdef float img_max, img_min
        for c in range(img_channel):
            img_max = img[c].max()
            img_min = img[c].min()
            if img_max != img_min:
                img[c] = (img[c] - img_min) / (img_max - img_min)
        cdef Tensor Img=to_tensor(img)
        cdef Tensor output =self.conv_instance.conv(Img)
        return to_numpy(output)
    
    def Backward_conv1d(self,np.ndarray[np.float32_t,ndim=1] grad_layer ,np.ndarray[np.float32_t, ndim=3] brev_hidden_layer,float learning_rate):
        cdef Tensor Brev_hidden_layer=to_tensor(brev_hidden_layer)
        cdef vector[float] GradLayer
        GradLayer.resize(grad_layer.shape[0])
        for i in range (grad_layer.shape[0]):
            GradLayer[i]=grad_layer[i]
        self.conv_instance.backward_conv1d(GradLayer,Brev_hidden_layer,learning_rate)

    def Backward_conv3d(self,np.ndarray[np.float32_t,ndim=3] grad_layer ,np.ndarray[np.float32_t, ndim=3] brev_hidden_layer,float learning_rate):
        cdef Tensor Brev_hidden_layer=to_tensor(brev_hidden_layer)
        cdef Tensor Grad_layer
        Grad_layer=to_tensor(grad_layer)
        self.conv_instance.backward_conv3d(Grad_layer,Brev_hidden_layer,learning_rate)

    def Grid_conv3d(self,np.ndarray[np.float32_t, ndim=3] grad_layer):
        cdef Tensor Grad_layer=to_tensor(grad_layer)
        cdef Tensor output=self.conv_instance.grid_conv3d(Grad_layer)
        return to_numpy(output)

    def Grid_conv1d(self,np.ndarray[np.float32_t, ndim=1] grad_layer):
        cdef vector[float] GradLayer
        GradLayer.resize(grad_layer.shape[0])
        for i in range (grad_layer.shape[0]):
            GradLayer[i]=grad_layer[i]
        cdef Tensor output=self.conv_instance.grid_conv1d(GradLayer)
        return to_numpy(output)

    def get_weight(self):
        cdef vector[vector[vector[vector[float]]]] cpp_weights = self.conv_instance.get_weight()
        return [[[[cpp_weights[i][j][hk][wk] for wk in range(cpp_weights[i][j][hk].size())]
                for hk in range(cpp_weights[i][j].size())]
                for j in range(cpp_weights[i].size())]
                for i in range(cpp_weights.size())]

    def get_bias(self):
        cdef vector[float] cpp_bias = self.conv_instance.get_bias()
        return [cpp_bias[i] for i in range(cpp_bias.size())]
    def Training_state(self):
        return self.conv_instance.Training_state()






cdef class PyMaxpool:
    cdef maxpool maxpool_instance  # Reference to the C++ class
    def __cinit__(self, int stride=2, int Ksize=2):
        self.maxpool_instance = maxpool(stride, Ksize)
    def __call__(self,img):
        return self.Maxipool(img)
    def Maxipool(self,np.ndarray[np.float32_t, ndim=3] img, int Stride=2, int Ksize=2):
        cdef Tensor Maxi=to_tensor(img)
        cdef Tensor output=self.maxpool_instance.maxipool(Maxi)
        return to_numpy(output)
    # rewrite
    def Grad_maxpool(self,np.ndarray[np.float32_t, ndim=3] before_maxpool,np.ndarray[np.float32_t, ndim=3] maxpool,np.ndarray[np.float32_t] graident_layer):
        
        cdef Tensor Before_maxpool=to_tensor(before_maxpool)
        cdef Tensor Max=to_tensor(maxpool)
        cdef Tensor Graident_layer
        cdef Tensor output
        cdef int size
        cdef vector[float] Graident_Layer
        if graident_layer.ndim==3 and graident_layer.size > 0:
            Graident_layer=to_tensor(graident_layer)
            output =self.maxpool_instance.backward_maxpool(Before_maxpool,Max,Graident_layer)
            return to_numpy(output)
        elif graident_layer.ndim ==1 and graident_layer.size > 0:
            size=graident_layer.shape[0]
            Graident_Layer.resize(size)
            for c in range(size):
                Graident_Layer[c]=graident_layer[c]
                output =self.maxpool_instance.backward_maxpool(Before_maxpool,Max,Graident_Layer)
                return to_numpy(output)

    
    

cdef class PyRelU:
    cdef RelU RelU_instance

    def __cinit__(self):
        self.RelU_instance = RelU()
    def __call__(self,img):
        cdef Tensor Img1
        cdef vector[vector[float]] Img2
        cdef Tensor result2
        cdef int height 
        cdef int width
        if img.ndim==3 and img.size>0:
            Img1=to_tensor(img)
            result2=self.RelU_instance.relu3d(Img1)
            return to_numpy(result2)
        elif img.ndim==2 and img.size>0:
            height , width= img.shape[0],img.shape[1]
            Img2.resize(height)
            for i in range (height):
                Img2[i].resize(width)
                for c in range (width):
                    Img2[i][c]=img[i,c]
            return self.RelU.relu2d(Img2)

    def backward(self,np.ndarray[np.float32_t] img):
            cdef Tensor Img1
            cdef vector[vector[float]] Img2
            cdef vector[vector[float]] result1
            cdef Tensor result2
            cdef int height 
            cdef int width
            if img.ndim==3 and img.size>0:
                Img1=to_tensor(img)
                result2=self.RelU_instance.Grid_RelU3d(Img1)
                return to_numpy(result2)
            elif img.ndim==2 and img.size>0:
                height ,self. width= img.shape[0],img.shape[1]
                self.Img2.resize(height)
                for i in range (height):
                    self.Img2[i].resize(width)
                    for c in range (width):
                        self.Img2[i][c]=img[i,c]
                return self.RelU.Grid_RelU2d(self.Img2)



cdef class PySoftmax:
    cdef Softmax Softmax_instance
    def __cinit__(self):
        self.Softmax_instance =Softmax()
    def __call__(self,vec):
        return self.softmax(vec)
    
    def softmax(self,np.ndarray[np.float32_t, ndim=1] vec):

        cdef int vec_size=vec.shape[0]

        cdef vector[float] Vec

        Vec.resize(vec_size) 
        
        for i in range(vec_size):

            Vec[i]=vec[i]
        return np.array(self.Softmax_instance.softmax(Vec),dtype=np.float32)
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
            return np.array(self.Softmax_instance.backward_softmax(predict,Y_true),dtype=np.float32)


