import numpy as  np 
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from math_tools import manhattan_distance_4d,euclidean_distance

import torch
import torch.nn as nn

class MaxPooling(object):
    def __init__(self, shape, ksize, stride):
        self.input_shape = shape
        self.ksize = ksize
        self.stride = stride
        self.output_channels = shape[1]
        self.index = np.zeros(shape)
        self.output_shape = [shape[0],self.output_channels, shape[2] // self.stride, shape[3] // self.stride ]

    def forward(self, x):
        out = np.zeros([x.shape[0], self.output_channels,x.shape[2] // self.stride, x.shape[3] // self.stride ])
        #print(x)
        for b in range(x.shape[0]):
            for c in range(self.output_channels):
                for i in range(0, x.shape[2], self.stride):
                    for j in range(0, x.shape[3], self.stride):
                        out[b, c , i // self.stride, j // self.stride] = np.max(
                            x[b, c, i:i + self.ksize, j:j + self.ksize])
        return out
    

if __name__ == "__main__":
    In = np.random.randint(0, 3, (1,3,6,6))  
    shape=In.shape  
    ksize=2
    stride=2

    maxpool1=MaxPooling(shape,ksize,stride)
    out = maxpool1.forward(In)

    maxpool_tst=nn.MaxPool2d([ksize,ksize],[stride,stride])
    tst_out = maxpool_tst(torch.from_numpy(In).float())
    Out_tst=tst_out.detach().numpy()

    distance = manhattan_distance_4d(Out_tst, out)
    print(Out_tst)
    print(out)


    print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")
    euclidean_dist = euclidean_distance(Out_tst, out) 
    print(f"欧几里德距离: {euclidean_dist}")  
    err=out-Out_tst
    print(np.mean(err))

