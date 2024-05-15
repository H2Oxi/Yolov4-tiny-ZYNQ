import numpy as np
import torch
import torch.nn as nn
import os
from functools import reduce
from liner_quantize import conv_quantize,conv_quantize_ex_m
import copy
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from tools.math_tools import manhattan_distance_4d,euclidean_distance,my_plot

class Conv2D(object):
    def __init__(self, weights,shape, output_channels, ksize, stride,bias):
        self.input_shape = shape
        self.output_channels = output_channels
        self.input_channels = shape[1]
        self.batchsize = shape[0]
        self.stride = stride
        self.ksize = ksize
        self.weights=np.transpose(weights,(1,2,3,0))
        #print(self.weights)
        if(ksize==1):
            self.eta = np.zeros((shape[0], (shape[2] + ksize +1) // self.stride, 
                             (shape[3] + ksize +1) // self.stride,self.output_channels))
        else:
            self.eta = np.zeros((shape[0], (shape[2] ) // self.stride, 
                             (shape[3] ) // self.stride ,self.output_channels))
        self.bias=bias if bias is not None else np.zeros((1,output_channels,self.eta.shape[1],self.eta.shape[2]))
        self.out_shape = (self.eta.shape[0],self.eta.shape[3],self.eta.shape[1],self.eta.shape[2])

    def forward(self, x):
        col_weights = self.weights.reshape([-1, self.output_channels])

        x = np.pad(x, (
                (0, 0),(0,0) ,(1, 1), (1, 1)),
                             'constant', constant_values=0)
        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            #print(x.shape)
            
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            #print(self.col_image_i)
            #print(col_weights)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) , self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        conv_out=np.transpose(conv_out, (0,3,1,2))
        conv_out=conv_out+self.bias
        return conv_out

class Conv_2D(object):
    def __init__(self, shape , output_channels, ksize, stride,debug=0):
        self.debug=debug
        self.output_channels = output_channels
        self.input_channels = shape[1]
        
        self.stride = stride
        self.ksize = ksize
        self.input_shape = shape
        self.batchsize = shape[0]
        if(ksize==1):
            self.eta = np.zeros((shape[0], (shape[2]  ) // self.stride, 
                             (shape[3]  ) // self.stride,self.output_channels))
        else:
            self.eta = np.zeros((shape[0], (shape[2] ) // self.stride, 
                             (shape[3] ) // self.stride ,self.output_channels))
        self.out_shape = (self.eta.shape[0],self.eta.shape[3],self.eta.shape[1],self.eta.shape[2])
        

    def data_update(self,weights,bias):

        
        self.weights=np.transpose(weights,(1,2,3,0))
        self.bias=bias if bias is not None else np.zeros((1,output_channels,self.eta.shape[1],self.eta.shape[2]))
        
        

    def forward(self, x):
        if(self.debug):
            self.in_ref=copy.deepcopy(x)

        col_weights = self.weights.reshape([-1, self.output_channels])
        if (self.ksize==3):
            x = np.pad(x, (
                (0, 0),(0,0) ,(1, 1), (1, 1)),
                             'constant', constant_values=0)
        
            
        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            #print(x.shape)
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            #print(self.col_image_i)
            #print(col_weights)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) , self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        conv_out=np.transpose(conv_out, (0,3,1,2))
        conv_out=conv_out+self.bias

        if(self.debug):
            self.out_ref=copy.deepcopy(conv_out)
            conv_quantize_debug=conv_quantize(self.in_ref,self.weights,self.bias,self.out_ref)
            conv_quantize_debug.update_m0()
            self.S_in=conv_quantize_debug.S_x
            self.S_out=conv_quantize_debug.S_a
            
        return conv_out

    def save_quantize(self,name,in_ref,out_ref):
        print(f"save to {name}")
        weights_origin=self.weights.transpose(3,0,1,2)
        conv_quantize0=conv_quantize_ex_m(in_ref,weights_origin,self.bias,out_ref,m_bitshift=16,q_bit=8)
        conv_quantize0.update_m0()
        conv_quantize0.update_weights()
        int_dir=f"my_int_weights_relu0125/int{conv_quantize0.quantize_bit}_M{conv_quantize0.M_bit_shift}/{name}"
        os.makedirs(int_dir, exist_ok=True) 

        print(f"weights range from {np.min(conv_quantize0.weights_new)} to {np.max(conv_quantize0.weights_new)}")
        print(f"bias range from {np.min(conv_quantize0.bias_new)} to {np.max(conv_quantize0.bias_new)}")
        print(f"out_scale: {conv_quantize0.S_a}")
        np.save(f"{int_dir}/S_a",conv_quantize0.S_a)
        np.save(f"{int_dir}/M_0",conv_quantize0.M0)
        conv_quantize0.weights_new.astype(np.int8).tofile(f"{int_dir}/w_q.bin")
        bias_reshape=conv_quantize0.bias_new[0,:,0,0]
        bias_reshape.astype(np.int32).tofile(f"{int_dir}/b_q.bin")
        with open(f"{int_dir}/m_0.bin", 'wb') as f:  
            f.write(conv_quantize0.M0.to_bytes(2, byteorder='little'))
        
        


class Conv_2D_Quantized(object):
    def __init__(self, shape , output_channels, ksize, stride ,debug=0):
        self.debug=debug
        self.output_channels = output_channels
        self.input_channels = shape[1]
        self.m_scale=16
        self.stride = stride
        self.ksize = ksize
        self.input_shape = shape
        self.batchsize = shape[0]
        if(ksize==1):
            self.eta = np.zeros((shape[0], (shape[2]  ) // self.stride, 
                             (shape[3]  ) // self.stride,self.output_channels))
        else:
            self.eta = np.zeros((shape[0], (shape[2] ) // self.stride, 
                             (shape[3] ) // self.stride ,self.output_channels))
        self.out_shape = (self.eta.shape[0],self.eta.shape[3],self.eta.shape[1],self.eta.shape[2])
        

    def data_update(self,weights,bias,M0):
        self.weights=np.transpose(weights.astype(float),(1,2,3,0))
        self.bias=bias.reshape(1,bias.shape[0],1,1)
        self.bias=self.bias.astype(float)
        self.M0=M0

        

    def forward(self, x):
        
        if(self.debug):
            self.in_ref=copy.deepcopy(x)

        col_weights = self.weights.reshape([-1, self.output_channels])
        if (self.ksize==3):
            x = np.pad(x, (
                (0, 0),(0,0) ,(1, 1), (1, 1)),
                             'constant', constant_values=0)
        
            
        self.col_image = []
        conv_out = np.zeros(self.eta.shape)
        for i in range(self.batchsize):
            img_i = x[i][np.newaxis, :]
            #print(x.shape)
            self.col_image_i = im2col(img_i, self.ksize, self.stride)
            #print(self.col_image_i)
            #print(col_weights)
            conv_out[i] = np.reshape(np.dot(self.col_image_i, col_weights) , self.eta[0].shape)
            self.col_image.append(self.col_image_i)
        self.col_image = np.array(self.col_image)
        conv_out=np.transpose(conv_out, (0,3,1,2))
        conv_out=conv_out+self.bias
        conv_out=conv_out* (2**(-(self.m_scale-1)))*self.M0


        if(self.debug):
            self.out_ref=copy.deepcopy(conv_out)
            my_plot(self.out_ref)
        
        return conv_out

    

def im2col(image, ksize, stride):
    # image is a 4d tensor([batchsize, width ,height, channel])
    image_col = []
    #print(image)
    for i in range(0, image.shape[2] - ksize + 1, stride):
        for j in range(0, image.shape[3] - ksize + 1, stride):
            col = image[:, : ,i:i + ksize, j:j + ksize].reshape([-1])
            #print(col)
            image_col.append(col)
    image_col = np.array(image_col)

    return image_col

def manhattan_distance_4d(tensor1, tensor2):  

    # 确保两个张量的形状相同  

    assert tensor1.shape == tensor2.shape, "The tensors must have the same shape"  

      

    # 计算对应元素差的绝对值，然后求和  

    diff = np.abs(tensor1 - tensor2)  

    distance = np.sum(diff)  

    return distance  


if __name__ == "__main__":
    weights=np.load('my_weights/backbone.conv1.conv.weight_weights.npy')
    In=np.load('data_val/hook_in_data.npy')
    Out_tst=np.load('data_val/hook_out_data.npy')
    #In=np.ones((1,3,6,6))
    #In[0,1,:,:]=np.triu(In[0,1,:,:], k=0)
    #In[0,0,:,:]= np.ones((6,6))
    #In[0,2,:,:]= np.zeros((6,6))

    #In = np.random.randint(0, 3, (1,3,6,6))  

    #In=np.random.normal(loc=10, scale=2,size=(1,3,20,20))
    shape=In.shape  
    print(weights.shape)
    output_channels=32
    ksize=3
    stride=2
    #weights = np.random.randint(0, 3, (output_channels,shape[1],ksize,ksize))  
    #weights=np.ones((output_channels,shape[1],ksize,ksize))
    #weights[0,:,:,:]=np.zeros((shape[1],ksize,ksize))


    #weights=np.random.normal(loc=10, scale=2, size=(output_channels,shape[1],ksize,ksize))
    
    conv1=Conv2D(weights,shape, output_channels,ksize,stride)
    conv_layer = nn.Conv2d(in_channels=shape[1], out_channels=output_channels, kernel_size=ksize, stride=stride, padding=1,bias=False)  
    conv_layer.weight.data=torch.from_numpy(weights).float()
    my_out=conv1.forward(In)
    tst_out=conv_layer(torch.from_numpy(In).float())
    #Out_tst=tst_out.detach().numpy()
    #err=my_out-Out_tst

    distance = manhattan_distance_4d(Out_tst, my_out)  

    print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")
    print(np.mean(In))

