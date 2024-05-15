import numpy as np
from tools.math_tools import manhattan_distance_4d,euclidean_distance,cosine_similarity
from layer.BN import BatchNorm
from layer.Conv2D import Conv2D

In=np.load('data_val/hook_in_data.npy')
Out_tst=np.load('data_val/hook_relu_in_data.npy')
bias_weights=np.load('my_weights/backbone.conv1.bn.bias_weights.npy')
weight_weights=np.load('my_weights/backbone.conv1.bn.weight_weights.npy')
mean_weights=np.load('my_weights/backbone.conv1.bn.running_mean_weights.npy')
var_weights=np.load('my_weights/backbone.conv1.bn.running_var_weights.npy')
weights=np.load('my_weights/backbone.conv1.conv.weight_weights.npy')

shape=In.shape  
output_channels=32
ksize=3
stride=2

conv1=Conv2D(weights,shape, output_channels,ksize,stride,None)
bn1=BatchNorm(bias_weights,weight_weights,mean_weights,var_weights,[1,output_channels,shape[2]//2,shape[3]//2])

my_out=conv1.forward(In)
my_out=bn1.forward(my_out)

print(my_out.shape)
distance = manhattan_distance_4d(Out_tst, my_out)  
print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")
euclidean_dist = euclidean_distance(Out_tst, my_out) 
print(f"欧几里德距离: {euclidean_dist}")  



