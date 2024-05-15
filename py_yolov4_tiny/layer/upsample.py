import torch
import torch.nn as nn
import numpy as  np 
import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from tools.math_tools import manhattan_distance_4d,euclidean_distance

class UpSample(object):
    def __init__(self, shape):
        self.input_shape = shape
        
        self.output_shape = [shape[0],shape[1], 2*shape[2] , 2*shape[3]  ]

    def forward(self, x):
        out = np.zeros(self.output_shape)
        #print(x)
        for b in range(x.shape[0]):
            for c in range(x.shape[1]):
                out[b,c,:,:]=np.kron(x[b,c,:,:],np.ones((2,2)))

        return out
    
if __name__ == "__main__":
    In = np.random.randint(0, 3, (1,3,6,6))  
    shape=In.shape  
    

    maxpool1=UpSample(shape)
    out = maxpool1.forward(In)

    maxpool_tst=nn.Upsample(scale_factor=2, mode='nearest')
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