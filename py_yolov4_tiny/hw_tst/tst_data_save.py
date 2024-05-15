import numpy as np  
import os


'''arr = np.array([1, 2, 3, 4, 5], dtype=np.int16)  
dir=f"hw_tst/simple_array/"
os.makedirs(dir, exist_ok=True) 
arr.tofile(f"{dir}integer_array.bin")
'''

shape = (5,)  # 原始数组的形状  
dtype = np.int16  # 原始数组的数据类型  
# 读取文件内容到一个字节序列中  
with open(f"hw_tst/hw_data/write_array.bin", 'rb') as f:  
    raw_data = f.read()  
# 将字节序列转换为NumPy数组  
loaded_arr = np.frombuffer(raw_data, dtype=dtype).reshape(shape)  
print(loaded_arr)
