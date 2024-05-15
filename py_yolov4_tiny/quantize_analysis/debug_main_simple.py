from liner_quantize import conv_quantize,conv_quantize_ex_m
import numpy as np  
import os
from layer.Conv2D import Conv2D , Conv_2D_Quantized
from layer.LeakyRelu import LeakyRelu,LeakyReluInt
from tools.math_tools import my_plot
from model_data.CSPdarknet53_tiny import Basic_Conv,CSPDarkNet
import copy


def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr

hw_dir=f"hw_tst/conv_simple/int8"
In = load_bin(hw_dir,"in_q",(1,3,6,6),np.int8)
weights = load_bin(hw_dir,"w_q",(4,3,3,3),np.int8)
bias = load_bin(hw_dir,"b_q",(4,),np.int32)


output_channels=4
ksize=3
stride=2


##------standard---------##

bias_new=copy.deepcopy(bias)
bias_new=bias_new.reshape(1,bias_new.shape[0],1,1)

conv1=Conv2D(weights,In.shape, 
                 output_channels,ksize,stride,bias_new)
Out=conv1.forward(In.astype(float))
#out_conv1_true=copy.deepcopy(Out)



##------save-------------##


'''os.makedirs(hw_dir, exist_ok=True) 
In.astype(np.int8).tofile(f"{hw_dir}/in_q.bin")
weights.astype(np.int8).tofile(f"{hw_dir}/w_q.bin")
bias.astype(np.int32).tofile(f"{hw_dir}/b_q.bin")'''
print(In)
 
print(weights)
print(weights.shape)
print(bias)
print(Out)


'''shape = (1,32,208,208)  # 原始数组的形状  
dtype = np.int32  # 原始数组的数据类型  
# 读取文件内容到一个字节序列中  
with open(f"{hw_dir}/out_q.bin", 'rb') as f:  
    raw_data = f.read()  
# 将字节序列转换为NumPy数组  
out_q_tst = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  '''






