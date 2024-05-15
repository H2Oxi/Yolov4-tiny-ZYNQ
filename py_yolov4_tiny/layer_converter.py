import numpy as np
from layer.BN2Conv import BN2Conv
import os

#backbone
origin_dir="my_weights_relu_0125"
backbone_dir="my_fused_weights_relu0125/"
os.makedirs(backbone_dir, exist_ok=True) 

conv_w_name=f".conv.weight_weights.npy"
bn_w_name=f".bn.weight_weights.npy"
bn_b_name=f".bn.bias_weights.npy"
bn_mean_name=f".bn.running_mean_weights.npy"
bn_var_name=f".bn.running_var_weights.npy"
fused_w_name=f"fused_weights"
fused_b_name=f"fused_bias"

b_name=f".bias_weights.npy"
w_name=f".weight_weights.npy"

layername0=f"backbone"


cho_0=np.array([32,64])
shape_0=np.array([[1,3,416,416],[1,32,208,208]])

for i in range(2):
    
    layername1=f"conv{i+1}"
    
    bias_weights=np.load(f"{origin_dir}/{layername0}.{layername1}{bn_b_name}")
    weight_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_w_name}')
    mean_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_mean_name}')
    var_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_var_name}')
    weights=np.load(f'{origin_dir}/{layername0}.{layername1}{conv_w_name}')
    
    
    shape= shape_0[i] 
    print(shape)
    output_channels=cho_0[i]
    ksize=3
    stride=2
    
    weights_new,bias_new=BN2Conv((1,output_channels,shape[2]//stride,shape[3]//stride),
                weights,None,weight_weights,bias_weights,mean_weights,var_weights)
    
    fused_dir=backbone_dir+f"{layername0}/{layername1}"
    
    os.makedirs(fused_dir, exist_ok=True) 

    np.save(f"{fused_dir}/{fused_w_name}",weights_new)
    np.save(f"{fused_dir}/{fused_b_name}",bias_new)


# ksize=1,stride=1,in_shape=out_shape
def res_conv_shape(in_shape):
    res_shape=np.zeros((4,4))
    res_shape[0,:]=in_shape
    res_shape[1,:]=in_shape
    res_shape[1,1]=res_shape[1,1]//2
    res_shape[2,:]=res_shape[1,:]
    res_shape[3,:]=in_shape
    
    return res_shape

res_in_shape=np.array([[1,64,104,104],
                       [1,128,52,52],
                       [1,256,26,26]])

for i in range(3):
    res_shape=res_conv_shape(res_in_shape[i,:])
    layername1=f"resblock_body{i+1}"
    for j in range(4):
        layername2=f"conv{j+1}"
        
        bias_weights=np.load(f"{origin_dir}/{layername0}.{layername1}.{layername2}{bn_b_name}")
        weight_weights=np.load(f'{origin_dir}/{layername0}.{layername1}.{layername2}{bn_w_name}')
        mean_weights=np.load(f'{origin_dir}/{layername0}.{layername1}.{layername2}{bn_mean_name}')
        var_weights=np.load(f'{origin_dir}/{layername0}.{layername1}.{layername2}{bn_var_name}')
        weights=np.load(f'{origin_dir}/{layername0}.{layername1}.{layername2}{conv_w_name}')
        
        weights_new,bias_new=BN2Conv(res_shape[j,:].astype(int),
                weights,None,weight_weights,bias_weights,mean_weights,var_weights)
        
        fused_dir=backbone_dir+f"{layername0}/{layername1}/{layername2}"
    
        os.makedirs(fused_dir, exist_ok=True) 

        np.save(f"{fused_dir}/{fused_w_name}",weights_new)
        np.save(f"{fused_dir}/{fused_b_name}",bias_new)



layername1=f"conv3"

bias_weights=np.load(f"{origin_dir}/{layername0}.{layername1}{bn_b_name}")
weight_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_w_name}')
mean_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_mean_name}')
var_weights=np.load(f'{origin_dir}/{layername0}.{layername1}{bn_var_name}')
weights=np.load(f'{origin_dir}/{layername0}.{layername1}{conv_w_name}')


shape= np.array([1,512,13,13]).astype(int)


weights_new,bias_new=BN2Conv(shape,
            weights,None,weight_weights,bias_weights,mean_weights,var_weights)

fused_dir=backbone_dir+f"{layername0}/{layername1}"

os.makedirs(fused_dir, exist_ok=True) 
np.save(f"{fused_dir}/{fused_w_name}",weights_new)
np.save(f"{fused_dir}/{fused_b_name}",bias_new)

layername0=f"conv_for_P5"


bias_weights=  np.load(f"{origin_dir}/{layername0}{bn_b_name}")
weight_weights=np.load(f'{origin_dir}/{layername0}{bn_w_name}')
mean_weights=  np.load(f'{origin_dir}/{layername0}{bn_mean_name}')
var_weights=   np.load(f'{origin_dir}/{layername0}{bn_var_name}')
weights=       np.load(f'{origin_dir}/{layername0}{conv_w_name}')


shape= np.array([1,256,13,13]).astype(int)


weights_new,bias_new=BN2Conv(shape,
            weights,None,weight_weights,bias_weights,mean_weights,var_weights)

fused_dir=backbone_dir+f"{layername0}"

os.makedirs(fused_dir, exist_ok=True) 
np.save(f"{fused_dir}/{fused_w_name}",weights_new)
np.save(f"{fused_dir}/{fused_b_name}",bias_new)


out_shape_array=np.array([[1,256,26,26],
                       [1,512,13,13]])

for i in range(2):
    out_shape=out_shape_array[i,:]
    layername0=f"yolo_headP{i+4}"
    
    bias_weights=  np.load(f"{origin_dir}/{layername0}.0{bn_b_name}")
    weight_weights=np.load(f'{origin_dir}/{layername0}.0{bn_w_name}')
    mean_weights=  np.load(f'{origin_dir}/{layername0}.0{bn_mean_name}')
    var_weights=   np.load(f'{origin_dir}/{layername0}.0{bn_var_name}')
    weights=       np.load(f'{origin_dir}/{layername0}.0{conv_w_name}')


    


    weights_new,bias_new=BN2Conv(out_shape,
                weights,None,weight_weights,bias_weights,mean_weights,var_weights)

    fused_dir=backbone_dir+f"{layername0}/conv1"

    os.makedirs(fused_dir, exist_ok=True) 
    np.save(f"{fused_dir}/{fused_w_name}",weights_new)
    np.save(f"{fused_dir}/{fused_b_name}",bias_new)




out_shape=np.array([1,128,13,13])
layername0=f"upsample"

bias_weights=  np.load(f"{origin_dir}/{layername0}.upsample.0{bn_b_name}")
weight_weights=np.load(f'{origin_dir}/{layername0}.upsample.0{bn_w_name}')
mean_weights=  np.load(f'{origin_dir}/{layername0}.upsample.0{bn_mean_name}')
var_weights=   np.load(f'{origin_dir}/{layername0}.upsample.0{bn_var_name}')
weights=       np.load(f'{origin_dir}/{layername0}.upsample.0{conv_w_name}')

weights_new,bias_new=BN2Conv(out_shape,
            weights,None,weight_weights,bias_weights,mean_weights,var_weights)
fused_dir=backbone_dir+f"{layername0}/conv1"
os.makedirs(fused_dir, exist_ok=True) 
np.save(f"{fused_dir}/{fused_w_name}",weights_new)
np.save(f"{fused_dir}/{fused_b_name}",bias_new)




for i in range(2):
    
    layername0=f"yolo_headP{i+4}"
    
    bias=  np.load(f"{origin_dir}/{layername0}.1{b_name}")
    weights=       np.load(f'{origin_dir}/{layername0}.1{w_name}')



    fused_dir=backbone_dir+f"{layername0}/conv2"

    os.makedirs(fused_dir, exist_ok=True) 
    np.save(f"{fused_dir}/{fused_w_name}",weights)
    np.save(f"{fused_dir}/{fused_b_name}",bias)