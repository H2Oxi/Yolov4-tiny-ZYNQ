import numpy as np
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from tools.math_tools import manhattan_distance_4d,euclidean_distance,my_plot
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/layer')
from layer.Conv2D import Conv_2D ,Conv2D,Conv_2D_Quantized
from layer.BN2Conv import BN2Conv
from layer.MaxPooling import MaxPooling
from layer.LeakyRelu import LeakyRelu
import torch
import torch.nn as nn
from model_data.CSPdarknet53_tiny import CSPDarkNet,Basic_Conv
from layer.upsample import UpSample
import copy

class yolo_head(object):
    def __init__(self, name,in_shape , center_channels,out_channels,int_transfer=0,quantized_enable=0,debug=0):
        self.debug=debug
        self.quantized_enable=quantized_enable
        self.int_transfer=int_transfer
        self.out_channels = out_channels
        self.quantize_bit=8
        self.data_dir="my_fused_weights_relu0125"
        if(self.quantized_enable):
            self.conv1 = Basic_Conv(f"{name}/conv1",in_shape , center_channels, 3,quantized_enable=quantized_enable,debug=self.debug)
            self.conv2 = Conv_2D_Quantized(self.conv1.conv.out_shape,out_channels,1,1,debug=self.debug)
        else:
            self.conv1 = Basic_Conv(f"{name}/conv1",in_shape , center_channels, 3,int_transfer=self.int_transfer,debug=self.debug)
            self.conv2 = Conv_2D(self.conv1.conv.out_shape,out_channels,1,1,debug=self.debug)
        
        
        
        self.name=name
        

    def load_fused_data(self,name):

        if(self.quantized_enable):
            file_list_conv1 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/yolo_head"+ name +"/conv1"
            file_list_conv2 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/yolo_head"+ name +"/conv2"
            weights=load_bin(file_list_conv2,"w_q",(self.conv2.output_channels,self.conv2.input_channels,self.conv2.ksize,self.conv2.ksize),np.int8)
            bias=load_bin(file_list_conv2,"b_q",(self.conv2.output_channels,),np.int32)
            m_0=np.load(f"{file_list_conv2}/M_0.npy")
            self.conv1.data_update(file_list_conv1)
            #print(bias.shape)
            self.conv2.data_update(weights,bias,m_0)

        else:
            file_list_conv1 = f"{self.data_dir}/yolo_head"+ name +"/conv1"
            file_list_conv2 = f"{self.data_dir}/yolo_head"+ name +"/conv2"
            weights=np.load(f"{file_list_conv2}/fused_weights.npy")
            bias=np.load(f"{file_list_conv2}/fused_bias.npy")
            self.conv1.data_update(file_list_conv1)
            bias=bias.reshape(1,bias.shape[0],1,1)
            self.conv2.data_update(weights,bias)
            
            print(f"{name}/conv2")
            print(self.conv2.input_shape)
            print(self.conv2.weights.shape)
            print(self.conv2.bias.shape)
            print(self.conv2.out_shape)

            
        
        
        
        

    def forward(self,x):
        x=self.conv1.forward(x)

        if(self.int_transfer):
            in_ref=copy.deepcopy(x)

        x=self.conv2.forward(x)

        if(self.int_transfer):
            out_ref=copy.deepcopy(x)
            self.conv2.save_quantize(f"{self.name}/conv2",in_ref,out_ref)

        
        return x
    
class up_sample(object):
    def __init__(self, in_shape ,quantized_enable=0,int_transfer=0,debug=0):
        self.debug=debug
        self.quantized_enable=quantized_enable
        self.int_transfer=int_transfer
        self.quantize_bit=8
        self.data_dir="my_fused_weights_relu0125"
        self.conv1 = Basic_Conv("upsample/conv1",in_shape , in_shape[1]//2, 1,quantized_enable=self.quantized_enable,int_transfer=self.int_transfer,debug=self.debug)
        self.upsample1 =  UpSample (self.conv1.conv.out_shape ) 

    def load_fused_data(self):
        if(self.quantized_enable):
            file_list_conv1 = f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv1.conv.m_scale}/upsample/conv1"
        else:
            file_list_conv1 = f"{self.data_dir}/upsample/conv1"
            
        self.conv1.data_update(file_list_conv1)

    def forward(self,x):
        x=self.conv1.forward(x)
        x=self.upsample1.forward(x)
        return x
    
class YoloBody(object):
    def __init__(self,quantized_enable=0,int_transfer=0,debug=0):
        self.debug=debug     
        self.quantized_enable=quantized_enable
        self.int_transfer=int_transfer
        self.quantize_bit=8
        anchors_mask        = [[3,4,5], [1,2,3]]
        num_classes         =20
        self.fused_data_dir="my_fused_weights_relu0125"
        
        self.backbone       = CSPDarkNet(quantized_enable=self.quantized_enable                                                                         ,int_transfer=self.int_transfer ,debug=self.debug)
        self.conv_for_P5    = Basic_Conv("conv_for_P5",[1,512,13,13],256,1,quantized_enable=self.quantized_enable                                       ,int_transfer=self.int_transfer ,debug=self.debug)
        self.yolo_headP5    = yolo_head("yolo_headP5",[1,256,13,13],512,len(anchors_mask[0]) * (5 + num_classes),quantized_enable=self.quantized_enable  ,int_transfer=self.int_transfer,debug=self.debug)
        self.upsample       = up_sample([1,256,13,13],quantized_enable=self.quantized_enable                                                            ,int_transfer=self.int_transfer ,debug=self.debug)
        self.yolo_headP4    = yolo_head("yolo_headP4",[1,384,26,26],256,len(anchors_mask[1]) * (5 + num_classes),quantized_enable=self.quantized_enable  ,int_transfer=self.int_transfer,debug=self.debug)

    def load_my_data(self):
        if(self.quantized_enable):
            data_dir=f"my_int_weights_relu0125/int{self.quantize_bit}_M{self.conv_for_P5.conv.m_scale}/"
        else:
            data_dir=f"{self.fused_data_dir}/"
            
            
        
        
        self.backbone.load_my_data()    
        self.conv_for_P5.data_update(f"{data_dir}conv_for_P5")
        self.yolo_headP4.load_fused_data("P4")
        self.yolo_headP5.load_fused_data("P5")
        self.upsample.conv1.data_update(f"{data_dir}upsample/conv1")

    def forward(self, x):
        
        feat1, feat2 = self.backbone.forward(x)
        # 13,13,512 -> 13,13,256
        P5 = self.conv_for_P5.forward(feat2)
        # 13,13,256 -> 13,13,512 -> 13,13,75
        out0 = self.yolo_headP5.forward(P5) 

        # 13,13,256 -> 13,13,128 -> 26,26,128
        P5_Upsample = self.upsample.forward(P5)
        # 26,26,256 + 26,26,128 -> 26,26,384
        
        P4 = np.concatenate([P5_Upsample,feat1],axis=1)
        

        # 26,26,384 -> 26,26,256 -> 26,26,75
        out1 = self.yolo_headP4.forward(P4)
        
        return out0, out1


def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr
    


if __name__ == "__main__":
    
    In=np.load('data_val/hook_in_data.npy')
    

    my_yolo=YoloBody(quantized_enable=1,int_transfer=0)
    my_yolo.load_my_data()

    out0,out1 = my_yolo.forward(In)
    
    
    Out_tst_out0=np.load('data_val/hook_CSP_out_out0_data.npy')
    Out_tst_out1=np.load('data_val/hook_CSP_out_out1_data.npy')

    err0=out0-Out_tst_out0
    err1=out1-Out_tst_out1
    
    print(np.mean(err0))
    print(np.mean(err1))