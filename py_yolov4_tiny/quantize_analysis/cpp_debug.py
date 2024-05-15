from ref.utils_bbox import DecodeBox
from model_data.Yolo4tiny import YoloBody
import numpy as np
from ref.utils import get_anchors,get_classes,cvtColor,preprocess_input,resize_image
import torch
from PIL import Image,ImageDraw,ImageFont
import colorsys
from model_data.CSPdarknet53_tiny import CSPDarkNet
from layer.Conv2D import Conv2D
from liner_quantize import conv_quantize
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/quantize_analysis')

from tools.math_tools import my_plot
import copy


def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr


In=np.load('data_val/hook_in_data.npy')
bias=np.load('my_fused_weights/backbone/conv1/fused_bias.npy')
weights=np.load('my_fused_weights/backbone/conv1/fused_weights.npy')

hw_dir=f"hw_tst/conv1_relu0125/int8_m16"
in_hw_tst= load_bin(hw_dir,"in_q",(1,3,416,416),np.int8)

my_csp=CSPDarkNet(debug=0)
my_csp.load_my_data()
out0,out1 = my_csp.forward(In)

my_csp_quantized=CSPDarkNet(debug=0,quantized_enable=1,int_transfer=0)
my_csp_quantized.load_my_data()
out0_q,out1_q = my_csp_quantized.forward(in_hw_tst.astype(float))
out0_q_true=copy.deepcopy(out0_q)
out1_q_true=copy.deepcopy(out1_q)

out0_r=my_csp_quantized.resblock_body3.conv4.activation.S_2 * out0_q
out1_r=my_csp_quantized.conv3.activation.S_2 * out1_q

my_plot(out0_r)
my_plot(out0)
print(np.mean(out0_r-out0))
print(np.mean(out1_r-out1))

print(out0_q_true.shape)
print(out1_q_true.shape)

csp_feat1_tst= load_bin(hw_dir,"csp_feat1_q8",(1,256,26,26),np.int8)
csp_feat2_tst= load_bin(hw_dir,"csp_feat2_q8",(1,512,13,13),np.int8)

print("cpp_tst")
print(np.mean(csp_feat1_tst-out0_q_true))


my_plot(out0_q_true)
my_plot(csp_feat1_tst)

print(np.mean(csp_feat2_tst-out1_q_true))
my_plot(out1_q_true)
my_plot(csp_feat2_tst)

