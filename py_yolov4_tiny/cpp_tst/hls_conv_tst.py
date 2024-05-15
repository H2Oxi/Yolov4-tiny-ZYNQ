import numpy as np
import torch
import torch.nn as nn
import os
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/layer')
from layer.Conv2D import Conv2D

weights=np.ones((16,3,3,3))
In=np.ones((1,3,32,32))
bias=2* np.ones((1,16,16,16))

shape=In.shape  
output_channels=16
ksize=3
stride=2

conv1=Conv2D(weights,shape, output_channels,ksize,stride,bias)
out_tst=conv1.forward(In)

print(out_tst)