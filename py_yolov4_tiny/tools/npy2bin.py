import numpy as np  
import os

def npy2bin(dir,name):
    buffer=np.load(f"{dir}/{name}.npy")
    buffer.tofile(f"{dir}/{name}.bin")

def bias_flatten(dir , name ):
    buffer=np.load(f"{dir}/{name}.npy")
    new_buffer=np.zeros((buffer.shape[1]))
    for i in range(buffer.shape[1]):
        new_buffer[i]=buffer[0,i,0,0]

    return new_buffer

def load_bin(dir , name , shape , dtype):
    with open(f"{dir}/{name}.bin", 'rb') as f:  
        raw_data = f.read()  
    # 将字节序列转换为NumPy数组  
    loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape) 
    return loaded_arr

hw_dir=f"hw_tst/conv1/int8_m15"

npy2bin(hw_dir,"in_q")
'''npy2bin(hw_dir,"b_q")'''
npy2bin(hw_dir,"w_q")

new_bias=bias_flatten(hw_dir,"b_q").astype(np.int32)
new_bias.tofile(f"{hw_dir}/b_q.bin")


