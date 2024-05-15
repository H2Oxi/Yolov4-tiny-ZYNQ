import numpy as np
import matplotlib.pyplot as plt  
from tools.math_tools import my_plot
import copy
import os
from tools.float_quantize import quantize_float_to_fixed_point
import math

class LeakyRelu(object):
    def __init__(self, shape,debug=0):
        self.debug=debug
        self.output_shape = shape
        self.x = np. zeros(self.output_shape)
        self.alpha=0.125
        
    def forward(self,x):
        self.x= x 
        self.x[self.x<0]*=self.alpha
        if(self.debug):
            self.out_ref=copy.deepcopy(x)

        return self.x
       
    def save_quantize(self,name):
        self.quantize_bit=8
        q_max=2**(self.quantize_bit-1)-1
        q_min=-(2**(self.quantize_bit-1))

        self.y_max=np.max(self.out_ref ) 
        self.y_min=np.min(self.out_ref ) 
        self.y_max=get_abs_max(self.y_max,self.y_min)
        self.y_min=-self.y_max

        self.S_2=(self.y_max-self.y_min)/(q_max-q_min)
        
        print(f"save to {name}")
        
        int_dir=f"my_int_weights_relu0125/int{self.quantize_bit}_M16/{name}"
        os.makedirs(int_dir, exist_ok=True) 
        np.save(f"{int_dir}/S_2",self.S_2)


class LeakyReluInt(object):
    def __init__(self, in_array,shape,y_array,debug=0,quantize_bit=8,save_enable=1):
        self.debug=debug
        self.output_shape = shape
        self.save_enable=save_enable
        self.x = np. zeros(shape)

        self.quantize_bit=quantize_bit

        self.abs_max=3

        self.in_max=np.max(in_array) if in_array is not None else self.abs_max
        self.in_min=np.min(in_array) if in_array is not None else -self.abs_max
        self.in_max=get_abs_max(self.in_max,self.in_min)
        self.in_min=-self.in_max

        self.y_max=np.max(y_array ) if y_array is not None else self.abs_max
        self.y_min=np.min(y_array ) if y_array is not None else -self.abs_max
        self.y_max=get_abs_max(self.y_max,self.y_min)
        self.y_min=-self.y_max

        q_max=2**(self.quantize_bit-1)-1
        q_min=-(2**(self.quantize_bit-1))
        
        self.S_1=(self.in_max-self.in_min)/(q_max-q_min)
        self.S_2=(self.y_max-self.y_min)/(q_max-q_min)
        self.Z_1=round(q_max-self.in_max//self.S_1)
        self.Z_2=round(q_max-self.y_max//self.S_2)


        

    def forward(self,x):
        #print(self.S_1/self.S_2)
        if(self.debug):
            self.in_ref=copy.deepcopy(x)
        self.x=x
        for i in range(self.x.shape[0]):
            for j in range( self.x.shape[1]):
                for k in range(self.x.shape[2]):
                    x_temp=self.x[i,j,k,:]
                    #####  here S_1=S_2
                    x_temp[x_temp>=self.Z_1] = ((self.S_1/self.S_2)*(x_temp[x_temp>=self.Z_1]-self.Z_1))+self.Z_2
                    x_temp[x_temp<self.Z_1]=(0.125*(self.S_1/self.S_2)*(x_temp[x_temp<self.Z_1]-self.Z_1))+self.Z_2
                    
                    self.x[i,j,k,:]=x_temp
        if(self.debug):
            self.out_ref=copy.deepcopy(self.x)

        return self.x
    def data_update(self,S_1,y_array):
        self.y_max=np.max(y_array ) if y_array is not None else self.abs_max
        self.y_min=np.min(y_array ) if y_array is not None else -self.abs_max
        self.y_max=get_abs_max(self.y_max,self.y_min)
        self.y_min=-self.y_max

        q_max=2**(self.quantize_bit-1)-1
        q_min=-(2**(self.quantize_bit-1))
        
        self.S_1=S_1
        self.S_2=(self.y_max-self.y_min)/(q_max-q_min)
        self.Z_1=0
        self.Z_2=round(q_max-self.y_max//self.S_2)

    def S_update(self,dir):
        int_dir=f"{dir}"
        self.S_1=np.load(f"{int_dir}/S_a.npy")
        self.S_2=np.load(f"{int_dir}/S_2.npy")
        if(self.save_enable):
            S_relu=self.S_1/self.S_2
            quantized_value_r, quantization_error, quantized_value_q= quantize_float_to_fixed_point(S_relu, 15)  
            print(f"Quantized to {15} bits: {quantized_value_r:.4f}, Quantization Error: {quantization_error:.6f},Quantized to {quantized_value_q}")
            S_relu_q=quantized_value_q
            
            with open(f"{int_dir}/S_relu_q.bin", 'wb') as f:  
                f.write(S_relu_q.to_bytes(2, byteorder='little'))
        

def get_abs_max(max,min):
    abs_max=np.max([abs(max),abs(min)])
    return abs_max

if __name__ == "__main__":
    In=np.load('data_val/hook_relu_in_data.npy')
    Out_tst=np.load('data_val/hook_conv1_relu_out_data.npy')

    acc1=LeakyRelu(In.shape)
    my_out=acc1.forward(In)
    

    acc1_int=LeakyReluInt(In,In.shape,Out_tst)

    qx=np.clip(np.round((In/acc1_int.S_1)+acc1_int.Z_1),-128,127)
    #qx=np.round((In/acc1_int.S_1)+acc1_int.Z_1)
    
    #qx=(In/acc1_int.S_1)+acc1_int.Z_1

    

    my_int_out=acc1_int.forward(qx)
    out2=(my_int_out-acc1_int.Z_2)*acc1_int.S_2
    
    err=out2-Out_tst
    print(np.mean(err))
    my_plot(out2)
    my_plot(Out_tst)
    

    