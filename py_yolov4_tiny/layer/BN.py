import numpy as np
#from tools.math_tools import manhattan_distance_4d,euclidean_distance,cosine_similarity

class BatchNorm(object):
    def __init__(self, bias_weights,weight_weights,mean_weights,var_weights,shape):    
        self.input_shape = shape
        self.batchsize = shape[0]
        self.bias_weights=np.expand_dims(bias_weights,axis=(0,2,3))
        self.weight_weights=np.expand_dims(weight_weights,axis=(0,2,3))
        self.mean_weights=np.expand_dims(mean_weights,axis=(0,2,3))
        self.var_weights=np.expand_dims(var_weights,axis=(0,2,3))
        self.epsilon = 0.00001
        print(self.bias_weights.shape)

    def forward(self, x):
        
        self.normed_x = (x - self.mean_weights)/np.sqrt(self.var_weights+self.epsilon)
        return self.normed_x*self.weight_weights+self.bias_weights


 


if __name__ == "__main__":
    In=np.load('data_val/hook_out_data.npy')
    Out_tst=np.load('data_val/hook_relu_in_data.npy')
    bias_weights=np.load('my_weights/backbone.conv1.bn.bias_weights.npy')
    weight_weights=np.load('my_weights/backbone.conv1.bn.weight_weights.npy')
    mean_weights=np.load('my_weights/backbone.conv1.bn.running_mean_weights.npy')
    var_weights=np.load('my_weights/backbone.conv1.bn.running_var_weights.npy')
    
    shape=In.shape
    
    bn1=BatchNorm(bias_weights,weight_weights,mean_weights,var_weights,shape)
    my_out=bn1.forward(In)
    print(my_out.shape)

    distance = manhattan_distance_4d(Out_tst, my_out)  

    print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")

    euclidean_dist = euclidean_distance(Out_tst, my_out) 

    print(f"欧几里德距离: {euclidean_dist}")  

