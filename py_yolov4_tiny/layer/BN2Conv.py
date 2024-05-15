import numpy as np
#from Conv2D import Conv2D
#from BN import BatchNorm

import sys
sys.path.append('/home/derui/work/Py_proj/yolo_tiny_v4_baseline/tools')
from math_tools import manhattan_distance_4d,euclidean_distance

def BN2Conv(out_shape,conv_weights,conv_bias,bn_weights,bn_bias,bn_mean,bn_var):
    eps=0.00001
    conv_bias=conv_bias if conv_bias is not None else np.zeros(out_shape)
    w_bn=bn_weights/np.sqrt(bn_var+eps)
    w_bn_bias=bn_bias-(bn_mean*bn_weights)/np.sqrt(bn_var+eps)
    w_bn=w_bn.reshape(w_bn.shape[0],1,1,1)
    w_bn_bias=w_bn_bias.reshape(1,w_bn_bias.shape[0],1,1)
    weights_new=conv_weights*w_bn
    bias_new=w_bn_bias+conv_bias
    print(bias_new.shape)
    print(weights_new.shape)
    return weights_new,bias_new


if __name__ == "__main__":
    In=np.load('data_val/hook_in_data.npy')
    #Out_tst=np.load('data_val/hook_relu_in_data.npy')
    bias_weights=np.load('my_weights/backbone.conv1.bn.bias_weights.npy')
    weight_weights=np.load('my_weights/backbone.conv1.bn.weight_weights.npy')
    mean_weights=np.load('my_weights/backbone.conv1.bn.running_mean_weights.npy')
    var_weights=np.load('my_weights/backbone.conv1.bn.running_var_weights.npy')
    weights=np.load('my_weights/backbone.conv1.conv.weight_weights.npy')
    #In = np.random.randint(0, 3, (1,3,6,6)) 
    
    shape=In.shape  
    output_channels=32
    ksize=3
    stride=2
    
    #weights=np.random.randint(0, 3, (output_channels,shape[1],ksize,ksize)) 
    #bias_weights=np.random.randint(0, 3,(output_channels))
    #weight_weights=np.random.randint(0, 3,(output_channels))
    #mean_weights=np.random.randint(0, 3,(output_channels))
    #var_weights=np.random.randint(0, 3,(output_channels))
    
    weights_new,bias_new=BN2Conv((1,output_channels,shape[2]//stride,shape[3]//stride),
                weights,None,weight_weights,bias_weights,mean_weights,var_weights)
    conv_co=Conv2D(weights_new,shape, output_channels,ksize,stride,bias_new)
    my_out=conv_co.forward(In)

    

    conv1=Conv2D(weights,shape, output_channels,ksize,stride,None)
    bn1=BatchNorm(bias_weights,weight_weights,mean_weights,var_weights,[1,output_channels,shape[2]//2,shape[3]//2])
    tst_out=conv1.forward(In)
    tst_out=bn1.forward(tst_out)
    Out_tst=tst_out

    distance = manhattan_distance_4d(Out_tst, my_out)  
    print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")
    euclidean_dist = euclidean_distance(Out_tst, my_out) 
    print(f"欧几里德距离: {euclidean_dist}")  

