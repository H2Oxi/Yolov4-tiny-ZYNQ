import numpy as np
from tools.math_tools import manhattan_distance_4d,euclidean_distance
#from layer.Conv2D import Conv2D
from layer.BN2Conv import BN2Conv
import matplotlib.pyplot as plt  
from tools.float_quantize import quantize_float_to_fixed_point
import os
from layer.LeakyRelu import LeakyReluInt
#conv1

class conv_quantize(object):
    def __init__(self, in_array , weights, bias ,y_array,z_in_enable=0):
        #self.in_array=in_array
        self.weights_late=weights 
        self.bias_late=bias

        self.quantize_bit=16
        self.bias_bit=4*self.quantize_bit
        self.M_bit_shift=24

        self.in_max=np.max(in_array) if in_array is not None else 1
        self.in_min=np.min(in_array) if in_array is not None else 0
        if(not z_in_enable):
            self.in_max=get_abs_max(self.in_max,self.in_min)
            self.in_min=-self.in_max
        
            

        self.weights_max=np.max(self.weights_late)
        self.weights_min=np.min(self.weights_late)
        self.weights_max=get_abs_max(self.weights_max,self.weights_min)
        self.weights_min=-self.weights_max

        self.bias_max=np.max(self.bias_late)
        self.bias_min=np.min(self.bias_late)
        self.bias_max=get_abs_max(self.bias_max,self.bias_min)
        self.bias_min=-self.bias_max

        self.y_max=np.max(y_array ) if y_array is not None else 3
        self.y_min=np.min(y_array ) if y_array is not None else -3
        self.y_max=get_abs_max(self.y_max,self.y_min)
        self.y_min=-self.y_max

        q_max=2**(self.quantize_bit-1)-1
        q_min=-(2**(self.quantize_bit-1))

        b_max=2**(self.bias_bit-1)-1
        b_min=-(2**(self.bias_bit-1))

        self.q_max=q_max
        self.q_min=q_min

        self.S_x=(self.in_max-self.in_min)/(q_max-q_min)
        self.S_w=(self.weights_max-self.weights_min)/(q_max-q_min)
        self.S_b=self.S_x*self.S_w
        self.S_a=(self.y_max-self.y_min)/(q_max-q_min)

        self.Z_x=round(q_max-self.in_max/self.S_x)
        self.Z_w=round(q_max-self.weights_max/self.S_w)
        self.Z_b=0
        self.Z_a=round(q_max-self.y_max/self.S_a)
        
        
        
        self.M=self.S_w*self.S_x/self.S_a
        self.M0=59

        self.qw=np.clip(np.round(self.weights_late/self.S_w +self.Z_w),q_min,q_max)
        self.qb=np.clip(np.round(self.bias_late/self.S_b +self.Z_b),b_min,b_max)


        self.weights_new=np.zeros(self.weights_late.shape)
        self.bias_new=np.zeros(self.bias_late.shape)

    def update_m0(self):
        
        quantized_value_r, quantization_error, quantized_value_q= quantize_float_to_fixed_point(self.M, self.M_bit_shift)  
        print(f"Quantized to {self.M_bit_shift} bits: {quantized_value_r:.4f}, Quantization Error: {quantization_error:.6f},Quantized to {quantized_value_q}")
        self.M0=quantized_value_q

    def update_weights(self):
        '''
        self.weights_new=self.qw-self.Z_w
        bias_q_temp=np.ones((self.qw.shape[0]))
        for i in range(self.qw.shape[0]):
            bias_q_temp[i]=np.sum(self.qw[i,:,:,:])
        bias_q_temp=bias_q_temp.reshape(1,bias_q_temp.shape[0],1,1)
        
        self.bias_new=self.Z_x*bias_q_temp+self.qb
        '''
        self.weights_new=self.qw        *self.M0
        self.bias_new=self.qb           *self.M0

    def update_qx(self,x_array):
        qx=np.clip(np.round(x_array/self.S_x +self.Z_x),self.q_min,self.q_max)
        return qx


        
def get_abs_max(max,min):
    abs_max=np.max([abs(max),abs(min)])
    return abs_max

class conv_quantize_ex_m(object):
    def __init__(self, in_array , weights, bias ,y_array,z_in_enable=0,m_bitshift=24,q_bit=16):
        #self.in_array=in_array
        self.weights_late=weights 
        self.bias_late=bias

        self.quantize_bit=q_bit
        self.bias_bit=4*self.quantize_bit
        self.M_bit_shift=m_bitshift

        self.in_max=np.max(in_array) if in_array is not None else 1
        self.in_min=np.min(in_array) if in_array is not None else 0
        if(not z_in_enable):
            self.in_max=get_abs_max(self.in_max,self.in_min)
            self.in_min=-self.in_max
        
            

        self.weights_max=np.max(self.weights_late)
        self.weights_min=np.min(self.weights_late)
        self.weights_max=get_abs_max(self.weights_max,self.weights_min)
        self.weights_min=-self.weights_max

        self.bias_max=np.max(self.bias_late)
        self.bias_min=np.min(self.bias_late)
        self.bias_max=get_abs_max(self.bias_max,self.bias_min)
        self.bias_min=-self.bias_max

        self.y_max=np.max(y_array ) if y_array is not None else 3
        self.y_min=np.min(y_array ) if y_array is not None else -3
        self.y_max=get_abs_max(self.y_max,self.y_min)
        self.y_min=-self.y_max

        q_max=2**(self.quantize_bit-1)-1
        q_min=-(2**(self.quantize_bit-1))

        b_max=2**(self.bias_bit-1)-1
        b_min=-(2**(self.bias_bit-1))

        self.q_max=q_max
        self.q_min=q_min

        self.S_x=(self.in_max-self.in_min)/(q_max-q_min)
        self.S_w=(self.weights_max-self.weights_min)/(q_max-q_min)
        self.S_b=self.S_x*self.S_w
        self.S_a=(self.y_max-self.y_min)/(q_max-q_min)

        self.Z_x=round(q_max-self.in_max/self.S_x)
        self.Z_w=round(q_max-self.weights_max/self.S_w)
        self.Z_b=0
        self.Z_a=round(q_max-self.y_max/self.S_a)
        
        
        
        self.M=self.S_w*self.S_x/self.S_a
        self.M0=1

        self.qw=np.clip(np.round(self.weights_late/self.S_w +self.Z_w),q_min,q_max)
        self.qb=np.clip(np.round(self.bias_late/self.S_b +self.Z_b),b_min,b_max)


        self.weights_new=np.zeros(self.weights_late.shape)
        self.bias_new=np.zeros(self.bias_late.shape)

    def update_m0(self):
        
        quantized_value_r, quantization_error, quantized_value_q= quantize_float_to_fixed_point(self.M, self.M_bit_shift)  
        print(f"Quantized to {self.M_bit_shift} bits: {quantized_value_r:.4f}, Quantization Error: {quantization_error:.6f},Quantized to {quantized_value_q}")
        self.M0=quantized_value_q

    def update_weights(self):
        '''
        self.weights_new=self.qw-self.Z_w
        bias_q_temp=np.ones((self.qw.shape[0]))
        for i in range(self.qw.shape[0]):
            bias_q_temp[i]=np.sum(self.qw[i,:,:,:])
        bias_q_temp=bias_q_temp.reshape(1,bias_q_temp.shape[0],1,1)
        
        self.bias_new=self.Z_x*bias_q_temp+self.qb
        '''
        self.weights_new=self.qw        ##*self.M0
        self.bias_new=self.qb           ##*self.M0

    def update_qx(self,x_array):
        qx=np.clip(np.round(x_array/self.S_x +self.Z_x),self.q_min,self.q_max)
        return qx



        

if __name__ == "__main__":
    
    '''In=np.load('data_val/hook_in_data.npy')
    bias_weights=np.load('my_weights/backbone.conv1.bn.bias_weights.npy')
    weight_weights=np.load('my_weights/backbone.conv1.bn.weight_weights.npy')
    mean_weights=np.load('my_weights/backbone.conv1.bn.running_mean_weights.npy')
    var_weights=np.load('my_weights/backbone.conv1.bn.running_var_weights.npy')
    weights=np.load('my_weights/backbone.conv1.conv.weight_weights.npy')

    Out_tst=np.load('data_val/hook_conv1_relu_out_data.npy')

    shape=In.shape  
    output_channels=32
    ksize=3
    stride=2
    
    weights_new,bias_new=BN2Conv((1,output_channels,shape[2]//stride,shape[3]//stride),
                weights,None,weight_weights,bias_weights,mean_weights,var_weights)
    
    conv_quantize0=conv_quantize(In,weights_new,bias_new,Out_tst)
    conv_quantize0.update_m0()
    conv_quantize0.update_weights()

    conv1_int_dir=f"hw_tst/conv1/int8_M16"
    
    os.makedirs(conv1_int_dir, exist_ok=True) 
    print(np.max(conv_quantize0.weights_new))

    np.save(f"{conv1_int_dir}/conv1_weights",conv_quantize0.weights_new)
    np.save(f"{conv1_int_dir}/conv1_bias",conv_quantize0.bias_new)'''

    '''conv1=Conv2D(conv_quantize0.weights_new,shape, 
                 output_channels,ksize,stride,conv_quantize0.bias_new)

    q_a_temp=conv1.forward(conv_quantize0.update_qx(In))
    q_a= q_a_temp * (2**(-15))
    
    acc1=LeakyReluInt(None,conv1.out_shape,None)
    conv1_acc1_out_int=acc1.forward(q_a)

    conv1_acc1_out=acc1.S_2*(conv1_acc1_out_int-acc1.Z_2)'''
    
    #r_a=conv_quantize0.S_a * ( q_a - conv_quantize0.Z_a)

    #print(conv_quantize0.M0 *(2**(-23)))
    #print(conv_quantize0.M)
    '''
    for num_bits in range(1,33,1):  
        quantized_value_r, quantization_error, quantized_value_q= quantize_float_to_fixed_point(conv_quantize0.M, num_bits)  
        print(f"Quantized to {num_bits} bits: {quantized_value_r:.4f}, Quantization Error: {quantization_error:.6f},Quantized to {quantized_value_q}")
    '''
    
    '''err=conv1_acc1_out-Out_tst
    print(np.mean(err))
    
    data=conv1_acc1_out
    counts, bin_edges = np.histogram(data, bins=100)  

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
    plt.bar(bin_centers, counts, width=np.diff(bin_edges)[0], edgecolor='black')  
    plt.title('Conv fused weights distribution histogram')  
    plt.xlabel('Value')  
    plt.ylabel('Frequency')  
    plt.grid(True)  
    plt.show()'''


    
'''    In = np.random.randint(0, 3, (1,3,40,40))  
    
    shape=In.shape  
    output_channels=20
    ksize=3
    stride=2
    
    weights = np.random.randint(0, 3, (output_channels,shape[1],ksize,ksize))  
    bias = np.random.randint(0, 3, (output_channels))
    bias = bias.reshape(1,bias.shape[0],1,1)

    conv_tst=Conv2D(weights,shape, 
                 output_channels,ksize,stride,bias)
    Out_tst=conv_tst.forward(In)
    
    conv_quantize0=conv_quantize(In,weights,bias,Out_tst)
    conv_quantize0.update_weights()

    conv1=Conv2D(conv_quantize0.weights_new,shape, 
                 output_channels,ksize,stride,conv_quantize0.bias_new)

    q_a_temp=conv1.forward(conv_quantize0.update_qx(In))
    q_a=conv_quantize0.M * q_a_temp+conv_quantize0.Z_a

    r_a=conv_quantize0.S_a * ( q_a - conv_quantize0.Z_a)

    err=r_a-Out_tst
    print(np.mean(err))
    
    distance = manhattan_distance_4d(Out_tst, r_a)  
    print(f"The Manhattan distance between tensor1 and tensor2 is: {distance}")
    euclidean_dist = euclidean_distance(Out_tst, r_a) 
    print(f"欧几里德距离: {euclidean_dist}") ''' 
    

    
    








