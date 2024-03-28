import math

# a = [11778,256,7184,16437,3530,246,26,39,15746,1883,2424,3935,5981,17655,2123,177,1644,4578,7197]
import torch.nn

a = [1075,1750, 1140, 1030, 1460, 955,1020]
# a = [1280, 1269, 1104, 947, 860, 1044, 970, 831, 504, 637, 1056, 1088, 1152, 636, 508]
sum = sum(a)
n_cls = 7

re_lis, re_lis2 = [], []
for i in a:
    re = round((sum)/(n_cls*i), 2)
    re2 = round((1 / (math.log ((1.5 + i / sum)))), 2)
    print('%d: %.2f   %.2f'%(a.index(i)+1, re, re2))
    re_lis.append(re)
    re_lis2.append(re2)
print('算法1：\n',re_lis)
print("算法2：\n", re_lis2)








