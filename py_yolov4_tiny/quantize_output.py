from liner_quantize import conv_quantize
import numpy as np  
import os
from layer.Conv2D import Conv2D , Conv_2D_Quantized
from layer.LeakyRelu import LeakyRelu,LeakyReluInt
from tools.math_tools import my_plot
from model_data.CSPdarknet53_tiny import Basic_Conv,CSPDarkNet
import copy

In=np.load('data_val/hook_in_data.npy')
bias=np.load('my_fused_weights/backbone/conv1/fused_bias.npy')
weights=np.load('my_fused_weights/backbone/conv1/fused_weights.npy')

output_channels=32
ksize=3
stride=2
conv1=Conv2D(weights,In.shape, 
                 output_channels,ksize,stride,bias)
Out=conv1.forward(In)

conv_quantize0=conv_quantize(In,weights,bias,Out,z_in_enable=0)
conv_quantize0.update_m0()
conv_quantize0.update_weights()
acc1=LeakyRelu(conv1.out_shape,debug=1)
Out=acc1.forward(Out)

In_q=conv_quantize0.update_qx(In)
print(np.max(In_q))
print(np.min(In_q))

#conv1_quantized=Conv_2D_Quantized(In.shape,output_channels,ksize,stride,debug=1)
#conv1_quantized.data_update(conv_quantize0.weights_new,conv_quantize0.bias_new)
conv1_quantized=Conv2D(conv_quantize0.weights_new,In_q.shape, 
             output_channels,ksize,stride,conv_quantize0.bias_new)
print(conv_quantize0.Z_x)
Out_q=conv1_quantized.forward(In_q-conv_quantize0.Z_x)

Out_q=(2**(-(conv_quantize0.M_bit_shift-1)))*Out_q #*conv_quantize0.M0

reg_out_q=copy.deepcopy(Out_q)

acc_quantized=LeakyReluInt(acc1.in_ref,conv1_quantized.out_shape,acc1.out_ref,debug=1)
Out_q=acc_quantized.forward(Out_q)
Out_r=Out_q*acc_quantized.S_2
#Out_r=Out_q*conv_quantize0.S_a*conv_quantize0.M0*(2**(-(conv_quantize0.M_bit_shift-1)))
my_plot(Out)
my_plot(Out_r)

print(f"Out-Out_r:{np.mean(Out-Out_r)}")

my_conv_in_csp=Basic_Conv("backbone/conv1",[1,3,416,416], 32, ksize=3, stride=2       ,quantized_enable=1,debug=1)
my_conv_in_csp.data_update("my_int_weights/int8_M15/backbone/conv1")
Out_q_2=my_conv_in_csp.forward(In_q)

Out_r_2=Out_q_2*acc_quantized.S_2
my_plot(Out_r_2)
print(acc_quantized.S_2)
print(my_conv_in_csp.activation.S_2)

print(f"Out-Out_r_2:{np.mean(Out-Out_r_2)}")
print(np.mean(Out_r-Out_r_2))
print(np.mean(reg_out_q-my_conv_in_csp.activation.in_ref))

bias_q_true = conv1_quantized.bias
bias_q_tst = my_conv_in_csp.conv.bias

weights_q_true = conv1_quantized.weights
weights_q_tst = my_conv_in_csp.conv.weights

print(np.mean(bias_q_tst-bias_q_true))
print(np.mean(np.transpose(weights_q_true,(1,2,3,0))-weights_q_tst))










'''my_csp_quantized=CSPDarkNet(debug=1,quantized_enable=1)
my_csp_quantized.load_my_data()
out0_q,out1_q = my_csp_quantized.forward(In_q)

out_q_3 = my_csp_quantized.conv1.conv.out_ref 

print(np.mean(Out_q-out_q_3))'''


'''#In = np.random.randint(0, 3, (1,3,4,4))  
In = np.ones((1,1,2,2)) 

print(In)
shape=In.shape  
output_channels=2
ksize=1
stride=1

#weights = np.random.randint(0, 3, (output_channels,shape[1],ksize,ksize))  
#bias = np.random.randint(0, 3, (output_channels))
bias = np.ones((output_channels))
weights = np.ones((output_channels,shape[1],ksize,ksize))  
bias = bias.reshape(1,bias.shape[0],1,1)
print(weights)
print(bias)

conv_tst=Conv2D(weights,shape, 
             output_channels,ksize,stride,bias)
Out_tst=conv_tst.forward(In)

conv_quantize0=conv_quantize(In,weights,bias,Out_tst)
conv_quantize0.update_m0()
conv_quantize0.update_weights()

print(conv_quantize0.qw)
print(conv_quantize0.qb)

conv1=Conv2D(conv_quantize0.weights_new,shape, 
             output_channels,ksize,stride,conv_quantize0.bias_new)

in_q=conv_quantize0.update_qx(In)
print(in_q)
print(conv_quantize0.weights_new)
print(conv_quantize0.bias_new)

q_a_temp=conv1.forward(in_q)
print(q_a_temp)

q_a=  q_a_temp * conv_quantize0.M#* (2**(-(15)))
r_a=conv_quantize0.S_a * ( q_a - conv_quantize0.Z_a)
err=r_a-Out_tst

print(f"S_x:{conv_quantize0.S_x}")
print(f"S_w:{conv_quantize0.S_w}")
print(f"S_b:{conv_quantize0.S_b}")
print(f"S_a:{conv_quantize0.S_a}")

print(conv_quantize0.S_a )
print(Out_tst)
print(q_a)
print(r_a)

print(np.mean(err))'''

