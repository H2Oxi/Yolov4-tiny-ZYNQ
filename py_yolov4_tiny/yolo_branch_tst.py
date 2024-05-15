from ref.utils_bbox import DecodeBox
from model_data.Yolo4tiny import YoloBody
import numpy as np
from ref.utils import get_anchors,get_classes,cvtColor,preprocess_input,resize_image
import torch
from PIL import Image,ImageDraw,ImageFont
import colorsys
from model_data.CSPdarknet53_tiny import CSPDarkNet,Basic_Conv
from layer.Conv2D import Conv2D
from liner_quantize import conv_quantize
from layer.LeakyRelu import LeakyRelu,LeakyReluInt


classes_path      = 'model_data/voc_classes.txt'
anchors_path      = 'model_data/yolo_anchors.txt'
anchors_mask      = [[3,4,5], [1,2,3]]
input_shape         =[416, 416]
nms_iou             =0.3
confidence          =0.5

image = Image.open("img/street.jpg")
image_shape = np.array(np.shape(image)[0:2])
image       = cvtColor(image)

image_data  = resize_image(image, (416,416), False)
image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

class_names, num_classes  = get_classes(classes_path)
print( num_classes)
anchors, num_anchors      = get_anchors(anchors_path)
bbox_util                      = DecodeBox(anchors, num_classes, (input_shape[0], input_shape[1]), anchors_mask)

In=np.load('data_val/hook_in_data.npy')
bias=np.load('my_fused_weights/backbone/conv1/fused_bias.npy')
weights=np.load('my_fused_weights/backbone/conv1/fused_weights.npy')
my_yolo_tst=YoloBody(debug=1)
my_yolo_tst.load_my_data()
out_tst=my_yolo_tst.forward(In)





#-----------pre quantized conv--------------#
my_yolo=YoloBody(quantized_enable=0)
my_yolo.load_my_data()

conv1_tst=np.load (f"data_val/quantize_data/int8_m15/conv1_out.npy")

x = my_yolo.backbone.conv2.forward(conv1_tst)
# 104,104,64 -> 52,52,128
x, _    = my_yolo.backbone.resblock_body1.forward(x)
# 52,52,128 -> 26,26,256
x, _    = my_yolo.backbone.resblock_body2.forward(x)
# 26,26,256 -> x为13,13,512
#           -> feat1为26,26,256
x, feat1    = my_yolo.backbone.resblock_body3.forward(x)
# 13,13,512 -> 13,13,512
x = my_yolo.backbone.conv3.forward(x)
feat2 = x

# 13,13,512 -> 13,13,256
P5 = my_yolo.conv_for_P5.forward(feat2)
# 13,13,256 -> 13,13,512 -> 13,13,75
out0 = my_yolo.yolo_headP5.forward(P5)
# 13,13,256 -> 13,13,128 -> 26,26,128
P5_Upsample = my_yolo.upsample.forward(P5)
# 26,26,256 + 26,26,128 -> 26,26,384

P4 = np.concatenate([P5_Upsample,feat1],axis=1)

# 26,26,384 -> 26,26,256 -> 26,26,75
out1 = my_yolo.yolo_headP4.forward(P4)


#out0,out1 = my_yolo.forward(In_q)

#out0=out0*my_yolo_tst.yolo_headP5.conv2.S_out
#out1=out1*my_yolo_tst.yolo_headP4.conv2.S_out




Out_tst_out0=np.load('data_val/hook_CSP_out_out0_data.npy')
Out_tst_out1=np.load('data_val/hook_CSP_out_out1_data.npy')


print(np.mean(Out_tst_out0-out0))
print(np.mean(Out_tst_out1-out1))

out0=torch.from_numpy(out0)
out1=torch.from_numpy(out1)

outputs=(out0,out1)
outputs = bbox_util.decode_box(outputs)

#---------------------------------------------------------#
#   将预测框进行堆叠，然后进行非极大抑制
#---------------------------------------------------------#
results = bbox_util.non_max_suppression(torch.cat(outputs, 1), num_classes, input_shape, 
            image_shape, False, conf_thres = confidence, nms_thres = nms_iou)
                                        

top_label   = np.array(results[0][:, 6], dtype = 'int32')
top_conf    = results[0][:, 4] * results[0][:, 5]
top_boxes   = results[0][:, :4]


hsv_tuples = [(x / num_classes, 1., 1.) for x in range(num_classes)]
colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
thickness   = int(max((image.size[0] + image.size[1]) // np.mean(input_shape), 1))
#---------------------------------------------------------#
#   图像绘制
#---------------------------------------------------------#
for i, c in list(enumerate(top_label)):
    predicted_class = class_names[int(c)]
    box             = top_boxes[i]
    score           = top_conf[i]
    top, left, bottom, right = box
    top     = max(0, np.floor(top).astype('int32'))
    left    = max(0, np.floor(left).astype('int32'))
    bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
    right   = min(image.size[0], np.floor(right).astype('int32'))
    label = '{} {:.2f}'.format(predicted_class, score)
    draw = ImageDraw.Draw(image)
    label_size = draw.textsize(label, font)
    label = label.encode('utf-8')
    print(label, top, left, bottom, right)
    
    if top - label_size[1] >= 0:
        text_origin = np.array([left, top - label_size[1]])
    else:
        text_origin = np.array([left, top + 1])
    for i in range(thickness):
        draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
    draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
    draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
    del draw

image.show()
