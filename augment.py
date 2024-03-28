import PIL.Image as Image
import cv2
import matplotlib.pyplot as plt
import os
import torchvision.transforms as T
import torch
import tqdm
import shutil


'''
# 扩增ALL-IDB和SN-AM
# ALL-IDB1-->*3, SN-AM-->*5
root = r'/mnt/data/ygh/ALL-IDB1'
ratio = 3

aug = T.Compose([T.RandomHorizontalFlip(0.5),
                 T.RandomVerticalFlip(0.5),
                 T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
                 T.RandomPosterize(4, 0.5)])

for p in os.listdir(root):
    if (p == 'A-TRAIN') or (p == 'A-VAL'):
        print(f"正在扩增{p}中样本!\n")
        for cls in tqdm.tqdm(os.listdir(os.path.join(root, p))):
            ims = os.listdir(os.path.join(root, p, cls))
            for im in ims:
                img = Image.open(os.path.join(root, p, cls, im)).convert("RGB")
                qian = im.split('.')[0]
                for i in range(ratio):
                    img = aug(img)
                    img.save(os.path.join(root, p, cls, f'{qian}_{i + 1}.jpg'))

print("全部样本均已扩增完毕！")
'''

'''
# 扩增嗜碱和异常细胞
p1 = r'F:\Blood\coding\5-auggan\DATA'
p2 = r'F:\Blood\coding\5-auggan\acgan-gai\inference_images\final\temp'

aug = T.Compose([T.RandomHorizontalFlip(0.5),
                 T.RandomVerticalFlip(0.5),
                 T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
                 T.GaussianBlur(3,sigma=(0.8,1.6)),
                 T.RandomPosterize(6, 0.5)])

for p in os.listdir(p1):
    if p in ['1','3']:
        os.makedirs(os.path.join(p2,p),exist_ok=True)
        print(f'正在扩增类别{p}的样本！\n')

        imgs = os.listdir(os.path.join(p1,p))
        for im in imgs:
            ii = Image.open(os.path.join(p1,p,im)).convert('RGB')
            ii = aug(ii)
            qian = im.split('.')[0]
            ii.save(os.path.join(p2,p,f'aug_{qian}.jpg'))

print("全部样本均已扩增完毕！")
'''


# 扩增CELL_CLS的13类单细胞
p1 = r'F:\Blood\FinalCells\CELL_CLS\cell_splitted2\train'
p2 = r'F:\Blood\FinalCells\CELL_CLS\cell_splitted2\train_aug'
RATIO = [4,4,3,5,4,5,5,5,6,6,4,6,4]
'''
aug = T.Compose([T.RandomHorizontalFlip(0.5),
                 T.RandomVerticalFlip(0.5),
                 T.RandomRotation([90,90],expand=True),
                 T.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1),
                 T.GaussianBlur(3,sigma=(0.8,1.6)),
                 T.RandomPosterize(6, 0.5)])

n = 0
for cls in os.listdir(p1):
    os.makedirs(os.path.join(p2,cls),exist_ok=True)
    print(f'-----正在扩增类别 [{cls}] 的样本！')

    imgs = os.listdir(os.path.join(p1,cls))
    for im in tqdm.tqdm(imgs):
        ii = Image.open(os.path.join(p1,cls,im)).convert('RGB')

        for i in range(0,RATIO[n]-1):
            ii = aug(ii)
            ii.save(os.path.join(p2,cls,f'aug{i+1}_'+im))

    print(f'-----类别 [{cls}] 扩增出原来的{RATIO[n]-1}倍：{len(imgs)} ---> {len(imgs)*(RATIO[n]-1)}\n')
    n += 1

print("全部样本均已扩增完毕！")
'''

'''
for cls in os.listdir(p1):
    if (cls == 'CML') or (cls == 'MDS-MPN'):
        os.makedirs(os.path.join(p2,cls),exist_ok=True)
        print(f'-----正在扩增类别 [{cls}] 的样本！')

        imgs = os.listdir(os.path.join(p1,cls))
        for im in tqdm.tqdm(imgs):
            ii = Image.open(os.path.join(p1,cls,im)).convert('RGB')

            for i in range(0,1):
                ii = aug(ii)
                ii.save(os.path.join(p2,cls,f'aug{i+1}_'+im))

        print(f'-----类别 [{cls}] 扩增出原来的1倍：{len(imgs)} ---> {len(imgs)}\n')
'''

# 扩增后把原先的也复制过去
for cls in tqdm.tqdm(os.listdir(p1)):
    imps = os.listdir(os.path.join(p1,cls))
    for imp in imps:
        shutil.copy(os.path.join(p1,cls,imp), os.path.join(p2,cls,imp))








