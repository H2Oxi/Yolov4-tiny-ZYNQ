import numpy as np  
import matplotlib.pyplot as plt  

backbone_dir="my_fused_weights/"
fused_w_name=f"fused_weights.npy"
fused_b_name=f"fused_bias.npy"

layername0=f"backbone"  
layername1=f"conv1"
layername2=f"conv2"
data_dir=backbone_dir+f"{layername0}/{layername1}"

data=np.load(f"{data_dir}/{fused_b_name}")
  
counts, bin_edges = np.histogram(data, bins=100)  


bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  


plt.bar(bin_centers, counts, width=np.diff(bin_edges)[0], edgecolor='black')  

  

plt.title('Conv fused weights distribution histogram')  
plt.xlabel('Value')  
plt.ylabel('Frequency')  



plt.grid(True)  
plt.show()