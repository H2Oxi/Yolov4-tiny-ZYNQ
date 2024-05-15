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
from model_data.CSPdarknet53_tiny import CSPDarkNet,Basic_Conv
from liner_quantize import conv_quantize
import copy

def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr


if __name__ == "__main__":
    In=np.load('data_val/hook_in_data.npy')
    hw_dir=f"hw_tst/conv1_again/int16_m24"
    in_hw_tst= load_bin(hw_dir,"in_q",(1,3,416,416),np.int16)
    
    my_csp_true=CSPDarkNet(debug=1)
    my_csp_true.load_my_data()
    out0,out1 = my_csp_true.forward(In)

    my_csp_quantized=CSPDarkNet(debug=1,quantized_enable=1)
    my_csp_quantized.load_my_data()
    out0_q,out1_q = my_csp_quantized.forward(in_hw_tst.astype(float))



    out1_r=my_csp_quantized.conv3.activation.S_2 * out1_q
    out0_r=my_csp_quantized.resblock_body3.conv4.activation.S_2 * out0_q

    print(np.mean(out0_r-out0))
    print(np.mean(out1_r-out1))
    my_plot(out0)
    my_plot(out0_r)

    err=np.zeros((30))
    
    err[0]  = np.mean(my_csp_true.conv1.conv.out_ref-my_csp_quantized.conv1.activation.S_1*my_csp_quantized.conv1.conv.out_ref                                                 )
    err[1]  = np.mean(my_csp_true.conv2.conv.out_ref-my_csp_quantized.conv2.activation.S_1*my_csp_quantized.conv2.conv.out_ref)
    err[2]  = np.mean(my_csp_true.resblock_body1.conv1.conv.out_ref-my_csp_quantized.resblock_body1.conv1.activation.S_1*my_csp_quantized.resblock_body1.conv1.conv.out_ref)
    err[3]  = np.mean(my_csp_true.resblock_body1.conv2.conv.out_ref-my_csp_quantized.resblock_body1.conv2.activation.S_1*my_csp_quantized.resblock_body1.conv2.conv.out_ref)
    err[4]  = np.mean(my_csp_true.resblock_body1.conv3.conv.out_ref-my_csp_quantized.resblock_body1.conv3.activation.S_1*my_csp_quantized.resblock_body1.conv3.conv.out_ref)
    err[5]  = np.mean(my_csp_true.resblock_body1.conv4.conv.out_ref-my_csp_quantized.resblock_body1.conv4.activation.S_1*my_csp_quantized.resblock_body1.conv4.conv.out_ref    )  
    err[6]  = np.mean(my_csp_true.resblock_body2.conv1.conv.out_ref-my_csp_quantized.resblock_body2.conv1.activation.S_1*my_csp_quantized.resblock_body2.conv1.conv.out_ref)
    err[7]  = np.mean(my_csp_true.resblock_body2.conv2.conv.out_ref-my_csp_quantized.resblock_body2.conv2.activation.S_1*my_csp_quantized.resblock_body2.conv2.conv.out_ref)
    err[8]  = np.mean(my_csp_true.resblock_body2.conv3.conv.out_ref-my_csp_quantized.resblock_body2.conv3.activation.S_1*my_csp_quantized.resblock_body2.conv3.conv.out_ref)
    err[9]  = np.mean(my_csp_true.resblock_body2.conv4.conv.out_ref-my_csp_quantized.resblock_body2.conv4.activation.S_1*my_csp_quantized.resblock_body2.conv4.conv.out_ref)
    err[10] = np.mean(my_csp_true.resblock_body3.conv1.conv.out_ref-my_csp_quantized.resblock_body3.conv1.activation.S_1*my_csp_quantized.resblock_body3.conv1.conv.out_ref)
    err[11] = np.mean(my_csp_true.resblock_body3.conv2.conv.out_ref-my_csp_quantized.resblock_body3.conv2.activation.S_1*my_csp_quantized.resblock_body3.conv2.conv.out_ref)
    err[12] = np.mean(my_csp_true.resblock_body3.conv3.conv.out_ref-my_csp_quantized.resblock_body3.conv3.activation.S_1*my_csp_quantized.resblock_body3.conv3.conv.out_ref)
    err[13] = np.mean(my_csp_true.resblock_body3.conv4.conv.out_ref-my_csp_quantized.resblock_body3.conv4.activation.S_1*my_csp_quantized.resblock_body3.conv4.conv.out_ref)
    err[14] = np.mean(my_csp_true.conv3.conv.out_ref-my_csp_quantized.conv3.activation.S_1*my_csp_quantized.conv3.conv.out_ref)


    
    print(err)