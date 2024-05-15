import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/layer')
from layer.Conv2D import Conv2D

def read_bin(name,shape,dtype):

    # 读取文件内容到一个字节序列中  
    with open(name, 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  
    return loaded_arr

dir = "hw_tst/conv1_4_1"
my_out_name=f"{dir}/out.bin"
out_shape = (1,32,208,208)  # 原始数组的形状  
dtype = np.int16  # 原始数组的数据类型 
my_out=read_bin(my_out_name,out_shape,dtype)


weights=read_bin(f"{dir}/weights.bin",(3,32,3,3),dtype)
In=read_bin(f"{dir}/in.bin",(1,3,416,416),dtype)
bias=read_bin(f"{dir}/bias.bin",(32,),dtype)

shape=In.shape  
output_channels=32
ksize=3
stride=2


conv1=Conv2D(weights,shape, output_channels,ksize,stride)
conv_layer = nn.Conv2d(in_channels=shape[1], out_channels=output_channels, kernel_size=ksize, stride=stride, padding=1,bias=False)  
conv_layer.weight.data=torch.from_numpy(weights).float()
out_tst=conv1.forward(In)

err=out_tst-my_out
print(np.mean(err))
