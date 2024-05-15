import numpy as np
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from tools.math_tools import manhattan_distance_4d,euclidean_distance,my_plot
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/layer')
from layer.Conv2D import Conv_2D ,Conv2D,Conv_2D_Quantized
from layer.BN2Conv import BN2Conv
from layer.MaxPooling import MaxPooling
from layer.LeakyRelu import LeakyRelu , LeakyReluInt
import torch
import torch.nn as nn
from liner_quantize import conv_quantize
import copy 

'''
class BasicConv(object):
    def __init__(self, conv_weights,conv_bias , in_shape , output_channels, ksize, stride=1 ):
        ## conv2d include bn
        self.conv = Conv2D(conv_weights,in_shape, 
                 output_channels,ksize,stride,conv_bias)      
        self.activation = LeakyRelu(self.conv.out_shape)

    def forward(self, x):
        x = self.conv.forward(x)
        x = self.activation.forward(x)
        return x
'''
    
class Basic_Conv(object):
    def __init__(self , name,shape , output_channels, ksize, stride=1 ,int_transfer=0,quantized_enable=0,debug=0):
        ## conv2d include bn
        self.quantized_enable=quantized_enable
        self.debug=debug
        self.int_transfer=int_transfer
        self.ksize=ksize
        if(self.quantized_enable):
            self.conv = Conv_2D_Quantized(shape,
                 output_channels,ksize,stride,debug=self.debug)      
            self.activation = LeakyReluInt(None,self.conv.out_shape,None,debug=self.debug)
        else:        
            self.conv = Conv_2D(shape,
                 output_channels,ksize,stride,debug=self.debug)      
            self.activation = LeakyRelu(self.conv.out_shape,debug=self.int_transfer)
        self.name=name
        
        

    def data_update(self,file_dir):
        if(self.quantized_enable):
            weights=load_bin(file_dir,"w_q",(self.conv.output_channels,self.conv.input_channels,self.ksize,self.ksize),np.int8)
            bias=load_bin(file_dir,"b_q",(self.conv.output_channels,),np.int32)
            m_0=np.load(f"{file_dir}/M_0.npy")
            self.conv.data_update(weights,bias,m_0)
            self.activation.S_update(file_dir)

        else:
            weights=np.load(f"{file_dir}/fused_weights.npy")
            bias=np.load(f"{file_dir}/fused_bias.npy")
            self.conv.data_update(weights,bias)
            print(f"{file_dir}")
            print(self.conv.input_shape)
            print(self.conv.weights.shape)
            print(self.conv.bias.shape)
            print(self.conv.out_shape)

    def forward(self, x):
        if(self.int_transfer):
            in_ref=copy.deepcopy(x)
        
        x = self.conv.forward(x)

        if(self.int_transfer):
            out_ref=copy.deepcopy(x)
            self.conv.save_quantize(self.name,in_ref,out_ref)
            
        x = self.activation.forward(x)

        if(self.int_transfer):
            self.activation.save_quantize(self.name)

        return x

class Resblock_body(object):
    def __init__(self, name0,in_shape , out_channels,quantized_enable=0,int_transfer=0,debug=0):
        self.quantized_enable=quantized_enable
        self.int_transfer=int_transfer
        self.debug=debug
        self.out_channels = out_channels
        self.quantize_bit=8
        self.fused_data_dir="my_fused_weights_relu0125"

        self.conv1 = Basic_Conv(f"{name0}/conv1",in_shape , out_channels, 3                     ,quantized_enable=self.quantized_enable ,int_transfer=self.int_transfer,debug=self.debug)
        self.conv2 = Basic_Conv(f"{name0}/conv2",self.conv1.conv.out_shape//np.array([1,2,1,1]), out_channels//2, 3  ,quantized_enable=self.quantized_enable ,int_transfer=self.int_transfer,debug=self.debug)
        self.conv3 = Basic_Conv(f"{name0}/conv3",self.conv2.conv.out_shape, out_channels//2, 3  ,quantized_enable=self.quantized_enable ,int_transfer=self.int_transfer,debug=self.debug)
        self.conv4 = Basic_Conv(f"{name0}/conv4",in_shape, out_channels, 1                      ,quantized_enable=self.quantized_enable ,int_transfer=self.int_transfer,debug=self.debug)
        
        self.maxpool = MaxPooling([in_shape[0],2*in_shape[1],in_shape[2],in_shape[3]],2,2)

    def load_fused_data(self,name):
        
        if(self.quantized_enable):
            file_list_conv1 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/backbone/"+ name +"/conv1"
            file_list_conv2 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/backbone/"+ name +"/conv2"
            file_list_conv3 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/backbone/"+ name +"/conv3"
            file_list_conv4 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/backbone/"+ name +"/conv4"
        else:
            file_list_conv1 = f"{self.fused_data_dir}/backbone/"+ name +"/conv1"
            file_list_conv2 = f"{self.fused_data_dir}/backbone/"+ name +"/conv2"
            file_list_conv3 = f"{self.fused_data_dir}/backbone/"+ name +"/conv3"
            file_list_conv4 = f"{self.fused_data_dir}/backbone/"+ name +"/conv4"
        
        
        self.conv1.data_update(file_list_conv1)
        self.conv2.data_update(file_list_conv2)
        self.conv3.data_update(file_list_conv3)
        self.conv4.data_update(file_list_conv4)

    def forward(self,x):
        x=self.conv1.forward(x)
        
        route = x
        #c = self.out_channels
        x= np.split(x,2,axis=1)[1]
        
        x = self.conv2.forward(x)
        route1 = x
        x = self.conv3.forward(x)
        
        x = np.concatenate([x,route1],axis=1)
        
        x= self.conv4.forward(x)
        feat =x
        x = np.concatenate([route,x],axis=1)

        x= self.maxpool.forward(x)
        return x,feat

class CSPDarkNet(object):
    def __init__(self,debug=0,quantized_enable=0,int_transfer=0):
        self.debug=debug
        self.quantized_enable=quantized_enable
        self.int_transfer=int_transfer
        self.quantize_bit=8
        self.fused_data_dir="my_fused_weights_relu0125"

        # 首先利用两次步长为2x2的3x3卷积进行高和宽的压缩
        # 416,416,3 -> 208,208,32 -> 104,104,64
        
        self.conv1 = Basic_Conv("backbone/conv1",[1,3,416,416], 32, ksize=3, stride=2       ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        self.conv2 = Basic_Conv("backbone/conv2",[1,32,208,208], 64, ksize=3, stride=2      ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        # 104,104,64 -> 52,52,128
        self.resblock_body1 =  Resblock_body("backbone/resblock_body1",[1,64,104,104], 64   ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        # 52,52,128 -> 26,26,256
        self.resblock_body2 =  Resblock_body("backbone/resblock_body2",[1,128,52,52], 128   ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        # 26,26,256 -> 13,13,512
        self.resblock_body3 =  Resblock_body("backbone/resblock_body3",[1,256,26,26], 256   ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        # 13,13,512 -> 13,13,512
        self.conv3 = Basic_Conv("backbone/conv3",[1,512,13,13], 512, ksize=3                ,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        
        if(self.quantized_enable):
            self.name = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}"
        else:
            self.name = f"{self.fused_data_dir}"

        
    def load_my_data(self):    
        self.conv1.data_update(f"{self.name}/backbone/conv1")
        self.conv2.data_update(f"{self.name}/backbone/conv2")
        self.conv3.data_update(f"{self.name}/backbone/conv3")
        self.resblock_body1.load_fused_data("resblock_body1")
        self.resblock_body2.load_fused_data("resblock_body2")
        self.resblock_body3.load_fused_data("resblock_body3")


    def forward(self, x):
        # 416,416,3 -> 208,208,32 -> 104,104,64
        x = self.conv1.forward(x)
        x = self.conv2.forward(x)

        # 104,104,64 -> 52,52,128
        x, _    = self.resblock_body1.forward(x)
        # 52,52,128 -> 26,26,256
        x, _    = self.resblock_body2.forward(x)
        # 26,26,256 -> x为13,13,512
        #           -> feat1为26,26,256
        x, feat1    = self.resblock_body3.forward(x)

        # 13,13,512 -> 13,13,512
        x = self.conv3.forward(x)
        feat2 = x
        return feat1,feat2
    
def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr
    



if __name__ == "__main__":
    
    
    In=np.load('data_val/hook_in_data.npy')
    bias=np.load('my_fused_weights/backbone/conv1/fused_bias.npy')
    weights=np.load('my_fused_weights/backbone/conv1/fused_weights.npy')

    output_channels=32
    ksize=3
    stride=2
    conv1=Conv2D(weights,In.shape, 
                     output_channels,ksize,stride,bias)
    Out=conv1.forward(In)
    conv_quantize0=conv_quantize(In,weights,bias,Out)
    conv_quantize0.update_m0()
    conv_quantize0.update_weights()

    In_q=conv_quantize0.update_qx(In)

    my_csp=CSPDarkNet(debug=0)
    my_csp.load_my_data()

    out0,out1 = my_csp.forward(In)

    
    my_csp_quantized=CSPDarkNet(debug=0,quantized_enable=1,int_transfer=0)
    my_csp_quantized.load_my_data()

    out0_q,out1_q = my_csp_quantized.forward(In_q)


    out0_r=my_csp_quantized.resblock_body3.conv4.activation.S_2 * out0_q
    out1_r=my_csp_quantized.conv3.activation.S_2 * out1_q

    my_plot(out0_r)
    my_plot(out0)

    print(np.mean(out0_r-out0))
    print(np.mean(out1_r-out1))
    