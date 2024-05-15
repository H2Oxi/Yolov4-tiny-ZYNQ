import numpy as np
import matplotlib.pyplot as plt 

def manhattan_distance_4d(tensor1, tensor2):  

    # 确保两个张量的形状相同  

    assert tensor1.shape == tensor2.shape, "The tensors must have the same shape"  

      

    # 计算对应元素差的绝对值，然后求和  

    diff = np.abs(tensor1 - tensor2)  

    distance = np.sum(diff)  

    return distance 

def euclidean_distance(vec1, vec2):  

    return np.sqrt(np.sum((vec1 - vec2) ** 2))  

  

# 计算余弦相似度  

def cosine_similarity(vec1, vec2):  

    dot_product = np.dot(vec1, vec2)  

    norm1 = np.linalg.norm(vec1)  

    norm2 = np.linalg.norm(vec2)  

    return dot_product / (norm1 * norm2)  

def my_plot(data)  :
    
    counts, bin_edges = np.histogram(data, bins=100)  

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  
    plt.bar(bin_centers, counts, width=np.diff(bin_edges)[0], edgecolor='black')  
    plt.title('Conv fused weights distribution histogram')  
    plt.xlabel('Value')  
    plt.ylabel('Frequency')  
    plt.grid(True)  
    plt.show()
