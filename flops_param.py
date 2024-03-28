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



