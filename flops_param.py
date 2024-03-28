import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision.models as M
import torch.nn as nn

import timm
from fvcore.nn import FlopCountAnalysis, parameter_count_table


# 列出timm全部网络
model_names = timm.list_models(pretrained=True)
for i in model_names:
    print(i)

#根据上面网络列表，可以选一个，num_classes默认为1000
# model = timm.create_model("resnext50_32x4d", pretrained=False)
model = M.densenet169(pretrained=False)
print(model)

# 384可更改，意为输入的大小
# tensor1 = torch.randn(1,3,512,512)
# flops = FlopCountAnalysis(model, tensor1)
# print(parameter_count_table(model))
# print ('img_size: %d, FLOPs: %.4fG' % (384, (1e-9) * flops.total ()))


'''
model = timm.create_model("mobilevitv2_150", pretrained=False,num_classes=7)
# model = M.resnet18(pretrained=False, num_classes=1000)
print(model)

x = torch.randn(2,3,320,320)
y_s = model.stem(x)
y0 = model.stages[0](y_s)
y10 = model.stages[1][0](y0)
y11 = model.stages[1][1](y10)
y20 = model.stages[2][0](y11)
y21 = model.stages[2][1](y20)
y30 = model.stages[3][0](y21)
y31 = model.stages[3][1](y30)
y40 = model.stages[4][0](y31)
y41 = model.stages[4][1](y40)
y_h = model.head(y41)
print("y_s   ", y_s.shape)
print("y_0   ", y0.shape)
print("y10   ", y10.shape)
print("y11   ", y11.shape)
print("y20   ", y20.shape)
print("y21   ", y21.shape)
print("y30   ", y30.shape)
print("y31   ", y31.shape)
print("y40   ", y40.shape)
print("y41   ", y41.shape)
print("y_h   ", y_h.shape)

tensor1 = torch.randn(1,3,320,320)
flops = FlopCountAnalysis(model, tensor1)
print(parameter_count_table(model))
print ('img_size: %d, FLOPs: %.4fG' % (320, (1e-9) * flops.total ()))
'''


'''
coat_lite_mini
coat_lite_small
coat_lite_tiny
coat_mini
coat_tiny
coatnet_0_rw_224
coatnet_1_rw_224
coatnet_bn_0_rw_224
coatnet_nano_rw_224
coatnet_rmlp_1_rw_224
coatnet_rmlp_2_rw_224
coatnet_rmlp_nano_rw_224

ghostnet_100

edgenext_base
edgenext_small
edgenext_small_rw
edgenext_x_small
edgenext_xx_small

repvgg_a2
repvgg_b0
repvgg_b1
repvgg_b1g4
repvgg_b2
repvgg_b2g4
repvgg_b3
repvgg_b3g4

mobilenetv3_large_100
mobilenetv3_large_100_miil
mobilenetv3_large_100_miil_in21k
mobilenetv3_rw

mobilevitv2_100
mobilevitv2_125
mobilevitv2_150
mobilevitv2_150_384_in22ft1k
mobilevitv2_150_in22ft1k
mobilevitv2_175
mobilevitv2_175_384_in22ft1k
mobilevitv2_175_in22ft1k
mobilevitv2_200
mobilevitv2_200_384_in22ft1k
mobilevitv2_200_in22ft1k

mixnet_l
mixnet_m
mixnet_s
mixnet_xl

vit_tiny_patch16_224
vit_tiny_r_s16_p8_224
'''


