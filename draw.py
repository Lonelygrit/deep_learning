import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T
import torch
import tqdm
import pandas as pd
import numpy as np
import itertools
import json

'''
# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes, path, normalize=False):
    """
    - cm : 计算出的混淆矩阵的值
    - classes : 混淆矩阵中每一行每一列对应的列
    - normalize : True:显示百分比, False:显示个数
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        cm = np.nan_to_num (cm, nan=0.0)
        # print("显示百分比：")
        np.set_printoptions(formatter={"float": "{: 0.2f}".format})
        # print(cm)
    else:
        pass
        # print("显示具体数字：")
        # print(cm)
    plt.clf()
    plt.imshow(cm, interpolation="nearest", cmap=plt.get_cmap('Blues'))
    # plt.title(label="")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size=10, family="Arial")
    plt.yticks(tick_marks, classes, rotation=45, size=10, family="Arial")
    # matplotlib版本问题，如果不加下面这行代码，则绘制的混淆矩阵上下只能显示一半，有的版本的matplotlib不需要下面的代码，分别试一下即可
    plt.ylim(len(classes) - 0.5, -0.5)
    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 fontsize=11,
                 family="Arial",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.gcf().subplots_adjust(bottom=0.2)
    plt.ylabel("True Label", size=11, family="Arial")
    plt.xlabel("Predicted Label", size=11, family="Arial")
    plt.savefig(path, dpi=300)
    plt.show()


CLASSES = ['PEN','PON','MYB','PMM','ABP','NST','NSG','EOS','BAS','LYM','MON','other']
hist = torch.tensor([[0.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01],   #原早红
                     [0.0,0.99,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.01,0.0,0.0],   # 中晚红
                     [0.0,0.0,0.96,0.01,0.01,0.0,0.0,0.0,0.0,0.0,0.02,0.0],   # 原粒
                     [0.0,0.01,0.0,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.01,0.0],    # 早中晚
                     [0.0,0.0,0.0,0.01,0.98,0.0,0.0,0.0,0.0,0.0,0.0,0.01],    # 异常早
                     [0.0,0.0,0.0,0.0,0.0,0.97,0.03,0.0,0.0,0.0,0.0,0.0],    # 杆状
                     [0.0,0.0,0.0,0.0,0.0,0.0,1.00,0.0,0.0,0.0,0.0,0.0],     # 分叶
                     [0.0,0.0,0.0,0.01,0.0,0.0,0.0,0.99,0.0,0.0,0.0,0.0],   #酸
                     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.00,0.0,0.0,0.0],    #碱
                     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,1.00,0.0,0.0],     # 淋巴
                     [0.0,0.0,0.02,0.02,0.0,0.0,0.0,0.0,0.0,0.01,0.94,0.01],    # 单核
                     [0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])   #其他
hist = (hist*100).type(torch.int8)
hist = hist.numpy()

CLASSES = ["ALL","AML","CML","L","MDS/MPN","MM"]
hist = torch.tensor([[11,0,0,0,0,0],
                     [0,11,0,1,0,0],
                     [0,0,8,0,0,0],
                     [0,0,0,10,0,0],
                     [0,0,0,1,9,0],
                     [0,0,0,0,0,10]])
hist = hist.type(torch.int8)
hist = hist.numpy()

plot_confusion_matrix(hist, CLASSES, 'test_cm.jpg', normalize=False)
'''


# 绘制损失曲线-cls
def curve(epoch_lis, train_los, test_los, train_ac, test_ac, path1, path2):
    """
    Draw the curves of loss and accuracy.
    """
    fig1 = plt.figure ()
    plt.plot(epoch_lis, train_los, "r", label="train")
    plt.plot(epoch_lis, test_los, "b", label="validation")
    # plt.title("Loss on training and validation sets")
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", size=13, family="Arial")
    plt.ylabel("Loss", size=13, family="Arial")
    plt.xticks(size=12, family="Arial")
    plt.yticks(size=12, family="Arial")
    fig1.savefig (path1,dpi=300)

    fig2 = plt.figure ()
    plt.plot (epoch_lis, train_ac, "r", label="train")
    plt.plot (epoch_lis, test_ac, "b", label="validation")
    # plt.title ("Accuracy on training and validation sets")
    plt.legend (fontsize=12)
    plt.xlabel("Epoch",size=13, family="Arial")
    plt.ylabel("Accuracy(%)",size=13, family="Arial")
    plt.xticks(size=12, family="Arial")
    plt.yticks(size=12, family="Arial")
    fig2.savefig (path2,dpi=300)


with open(r'F:\Blood\CODE\3-clsbaseline\runs_cell\3-ffnet_noconv-bce\train_result.json','r') as f:
    dic = json.load(f)

epoch_lis = dic['epoch']
train_los = dic['train_loss']
val_los = dic['val_loss']
train_ac = dic['train_acc']
val_ac = dic['val_acc']

curve(epoch_lis, train_los, val_los, train_ac, val_ac,
      path1="loss.png", path2="acc.png")


'''
# 绘制损失曲线-yolo5和p-r曲线
def curve(epoch_lis, box_los, obj_los, p, r, path1, path2):
    """
    Draw the curves of loss and accuracy.
    """
    fig1 = plt.figure ()
    plt.plot(epoch_lis, box_los, "r", label="box_loss")
    plt.plot(epoch_lis, obj_los, "b", label="obj_loss")
    # plt.title("Loss on training and validation sets")
    plt.legend(fontsize=12)
    plt.xlabel("Epoch", size=13, family="Arial")
    plt.ylabel("Loss", size=13, family="Arial")
    plt.xticks(size=12, family="Arial")
    plt.yticks(size=12, family="Arial")
    fig1.savefig(path1,dpi=300)

    fig2 = plt.figure ()
    plt.plot(r, p,  "g", label="NC 0.991 AP@0.5")
    # plt.title("Loss on training and validation sets")
    plt.legend(fontsize=12)
    plt.xlabel("Recall", size=13, family="Arial")
    plt.ylabel("Precision", size=13, family="Arial")
    plt.xticks(size=12, family="Arial")
    plt.yticks(size=12, family="Arial")
    fig2.savefig(path2, dpi=300)


dic = pd.read_csv(r"F:\Blood\CODE\4-yolo7\yolo5\runs\train\exp\results.csv")
epoch_lis = [i+1 for i in dic['epoch'].to_list()]
box_los = dic[r'train/box_loss'].to_list()
obj_los = dic[r'train/obj_loss'].to_list()

r = [0.0,   0.2,   0.3,   0.32,  0.34,  0.4,   0.51,  0.53,  0.68,  0.72,  0.8,   0.82,  0.85,  0.87,  0.89,  0.92,  0.93,  0.94,  0.95,  0.96,  0.965,  0.97,  0.98,  0.985,  0.99,  1.0]
p = [0.996, 0.996, 0.996, 0.996, 0.995, 0.995, 0.995, 0.993, 0.993, 0.988, 0.988, 0.983, 0.983, 0.983, 0.978, 0.978, 0.973, 0.963, 0.953, 0.943, 0.923,  0.903, 0.863, 0.783,  0.653, 0.33]

curve(epoch_lis, box_los, obj_los, p, r, path1=r"loss.png", path2=r"pr.png")
'''


'''
# 绘制多网络损失曲线
def curve(epoch_lis,train_los_vit,train_los_swin,train_los_swin2,train_los_deit3,train_los_movit2,path1):
    """
    Draw the curves of loss and accuracy.
    """
    fig1 = plt.figure ()
    plt.plot(epoch_lis, train_los_vit, label="ViT-S")
    plt.plot(epoch_lis, train_los_swin, label="Swin-S")
    plt.plot (epoch_lis, train_los_swin2, label="SwinV2-S")
    plt.plot (epoch_lis, train_los_deit3, label="DeiT III-S")
    plt.plot (epoch_lis, train_los_movit2, label="MobileViTV2")
    # plt.title("Loss on training and validation sets")
    plt.legend()
    plt.xlabel("Epoch", size=12, family="Arial")
    plt.ylabel("Loss", size=12, family="Arial")
    fig1.savefig (path1)


with open('tmp-draw/train-vit.json','r') as f:
    dic_vit = json.load(f)
with open('tmp-draw/train-swin.json','r') as f:
    dic_swin = json.load(f)
with open('tmp-draw/train-swin2.json','r') as f:
    dic_swin2 = json.load(f)
with open('tmp-draw/train-deit3.json','r') as f:
    dic_deit3 = json.load(f)
with open('tmp-draw/train-mobilevit2.json','r') as f:
    dic_movit2 = json.load(f)

epoch_lis = dic_vit['epoch']
train_los_vit = dic_vit['val_loss']
train_los_swin = dic_swin['val_loss']
train_los_swin2 = dic_swin2['val_loss']
train_los_deit3 = dic_deit3['val_loss']
train_los_movit2 = dic_movit2['val_loss']

curve(epoch_lis, train_los_vit,train_los_swin,train_los_swin2,train_los_deit3,train_los_movit2,
      path1="tmp-draw/vit_val_loss.png")
'''


'''
# 多图显示
p = r'E:\Program\TRANSLAB\cell_explain\细胞可解释性'
aug = T.Resize(size=(224,224))
fig = plt.figure(figsize=(15,55))
i = 1
for im_p in tqdm.tqdm(os.listdir(p)):
    img = Image.open(os.path.join(p,im_p)).convert("RGB")

    if '-a' in im_p:
        img = aug(img)

    ax = fig.add_subplot(11,3,i,xticks=[],yticks=[])
    plt.subplots_adjust(left=0.05,bottom=0.05,right=0.05,top=0.05,wspace=0.0,hspace=0)
    plt.tight_layout()

    plt.imshow(img)
    plt.axis('off')
    plt.savefig('explain.jpg')
    i += 1

plt.show()
'''




