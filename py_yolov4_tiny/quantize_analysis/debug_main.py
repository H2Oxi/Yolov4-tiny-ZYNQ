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


In=np.load('data_val/hook_in_data.npy')
bias=np.load('my_fused_weights_relu0125/backbone/conv1/fused_bias.npy')
weights=np.load('my_fused_weights_relu0125/backbone/conv1/fused_weights.npy')

output_channels=32
ksize=3
stride=2
print(weights.shape)

##------standard---------##

conv1=Conv2D(weights,In.shape, 
                 output_channels,ksize,stride,bias)
Out=conv1.forward(In)
out_conv1_true=copy.deepcopy(Out)

##------quantized init---------##
conv_quantize0=conv_quantize_ex_m(In,weights,bias,Out,z_in_enable=0,m_bitshift=16,q_bit=8)
conv_quantize0.update_m0()
conv_quantize0.update_weights()


##------quantized---------##

In_q=conv_quantize0.update_qx(In)
In_q_reg=copy.deepcopy(In_q)
In_q_reg_2=copy.deepcopy(In_q)
##------save-------------##

hw_dir=f"hw_tst/conv1_relu0125/int8_m16"
os.makedirs(hw_dir, exist_ok=True) 
In_q.astype(np.int8).tofile(f"{hw_dir}/in_q.bin")
conv_quantize0.weights_new.astype(np.int8).tofile(f"{hw_dir}/w_q.bin")

bias_reshape=conv_quantize0.bias_new[0,:,0,0]
bias_reshape.astype(np.int32).tofile(f"{hw_dir}/b_q.bin")



in_hw_tst= load_bin(hw_dir,"in_q",(1,3,416,416),np.int8)
w_hw_tst= load_bin(hw_dir,"w_q",(32,3,3,3),np.int8)
b_hw_tst= load_bin(hw_dir,"b_q",(32,),np.int32)




b_hw_tst_reg=copy.deepcopy(b_hw_tst)
b_hw_tst=b_hw_tst.reshape(1,b_hw_tst.shape[0],1,1)


conv1_quantized=Conv2D(w_hw_tst,In_q.shape, 
             output_channels,ksize,stride,b_hw_tst)
Out_q=conv1_quantized.forward(in_hw_tst.astype(float))
print(np.max(Out_q))
print(np.min(Out_q))
my_plot(Out_q)
print(Out_q[0,0,0,0])
Out_q_temp=copy.deepcopy(Out_q)
Out_q=(2**(-(conv_quantize0.M_bit_shift-1)))*Out_q *conv_quantize0.M0
Out_q_true=copy.deepcopy(np.floor(Out_q))
Out_q_true_2=copy.deepcopy(np.floor(Out_q))
my_plot(Out_q_true)

reg_out_q=copy.deepcopy(Out_q)
reg_out_q = np.round(reg_out_q)


out_conv1_tst=conv_quantize0.S_a*reg_out_q

print(np.mean(out_conv1_tst-out_conv1_true))



acc1=LeakyRelu(out_conv1_tst.shape,debug=1)
out_acc1_true=acc1.forward(out_conv1_true)

acc1_q=LeakyReluInt(None,Out_q_true.shape,None)
acc1_q.data_update(conv_quantize0.S_a,acc1.out_ref)
out_acc1_tst=acc1_q.forward(Out_q_true)
out_acc1_tst=acc1_q.S_2*out_acc1_tst

my_plot(out_acc1_tst)
my_plot(out_acc1_true)
print(np.mean(out_acc1_tst-out_acc1_true))

conv_q_tst=Conv_2D_Quantized(In_q_reg.shape,output_channels,ksize,stride,debug=1)


file_dir = "my_int_weights_relu0125/int8_M16/backbone/conv1"

conv_q_tst.data_update(w_hw_tst.astype(np.float),b_hw_tst_reg.astype(np.float),conv_quantize0.M0)


out_conv1_q_tst2=conv_q_tst.forward(In_q_reg)
my_plot(out_conv1_q_tst2)
my_plot(Out_q_true_2)
print(np.mean(out_conv1_q_tst2-Out_q_true_2))
print(np.mean(out_conv1_q_tst2/Out_q_true_2))
out_conv1_r_tst2=out_conv1_q_tst2*conv_quantize0.S_a##*conv_quantize0.M0

print(np.mean(out_conv1_r_tst2-out_conv1_tst))


conv_q_tst_basic_tst=Basic_Conv("backbone/conv1",[1,3,416,416], 32, ksize=3, stride=2 ,quantized_enable=1)
conv_q_tst_basic_tst.data_update(f"{file_dir}")

out_basic_tst=conv_q_tst_basic_tst.forward(In_q_reg_2.astype(float))
out_basic_tst=conv_q_tst_basic_tst.activation.S_2*out_basic_tst
print(np.mean(out_basic_tst-out_acc1_true))

shape = (1,32,208,208)  # 原始数组的形状  
dtype = np.int8  # 原始数组的数据类型  
# 读取文件内容到一个字节序列中  
with open(f"{hw_dir}/acc_Out_q.bin", 'rb') as f:  
    raw_data = f.read()  
# 将字节序列转换为NumPy数组  
out_q_tst = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  

out_q_tst = out_q_tst*conv_q_tst_basic_tst.activation.S_2
my_plot(out_q_tst)
print(acc1_q.S_2)
print(conv_q_tst_basic_tst.activation.S_2)
print(np.mean(out_q_tst-out_acc1_tst))

dtype = np.int8  # 原始数组的数据类型  
# 读取文件内容到一个字节序列中  
with open(f"{hw_dir}/conv_Out_q8.bin", 'rb') as f:  
    raw_data = f.read()  
# 将字节序列转换为NumPy数组  
out_q_tst = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  
my_plot(out_q_tst)
print(np.mean(out_q_tst.astype(float)-Out_q_true_2))


dtype = np.int32  # 原始数组的数据类型  
# 读取文件内容到一个字节序列中  
with open(f"{hw_dir}/conv_Out_q32.bin", 'rb') as f:  
    raw_data = f.read()  
# 将字节序列转换为NumPy数组  
out_q_tst = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  
my_plot(out_q_tst)





