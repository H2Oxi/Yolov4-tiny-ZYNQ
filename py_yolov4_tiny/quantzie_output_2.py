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

reg_out_true=copy.deepcopy(Out)

conv_quantize0=conv_quantize(In,weights,bias,Out,z_in_enable=1)
conv_quantize0.update_m0()
conv_quantize0.update_weights()

dir=f"hw_tst/conv1_4_1/"
os.makedirs(dir, exist_ok=True) 

save_weights=conv_quantize0.weights_new.astype(np.int8)
save_weights.tofile(f"{dir}weights.bin")
save_bias=conv_quantize0.bias_new.astype(np.int8)
save_bias.tofile(f"{dir}bias.bin")

acc1=LeakyRelu(conv1.out_shape,debug=1)
Out=acc1.forward(Out)

In_q=conv_quantize0.update_qx(In)
save_in = In_q.astype(np.int8)
save_in.tofile(f"{dir}in.bin")

print(np.max(In_q))
print(np.min(In_q))

conv1_quantized=Conv2D(conv_quantize0.weights_new,In_q.shape, 
             output_channels,ksize,stride,conv_quantize0.bias_new)


print(conv_quantize0.Z_x)
Out_q=conv1_quantized.forward(In_q-conv_quantize0.Z_x)

Out_q=(2**(-(conv_quantize0.M_bit_shift-1)))*Out_q #*conv_quantize0.M0

reg_out_q=copy.deepcopy(Out_q)

acc_quantized=LeakyReluInt(acc1.in_ref,conv1_quantized.out_shape,acc1.out_ref,debug=1)
Out_q=acc_quantized.forward(Out_q)
Out_r=Out_q*acc_quantized.S_2

my_plot(Out)
my_plot(Out_r)

print(f"Out-Out_r:{np.mean(Out-Out_r)}")

reg_out_tst=reg_out_q * conv_quantize0.S_a
print(f"Out_reg_true-Out_reg_tst:{np.mean(reg_out_true-reg_out_tst)}")

Out_tst_2=acc1.forward(reg_out_tst)
print(f"Out-Out_tst_2:{np.mean(Out-Out_tst_2)}")

