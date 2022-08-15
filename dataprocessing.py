import os
import cv2
import torchvision
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.utils import save_image

# 简单的数据集，没有进行数据增强
class UnetDatasets(Dataset):
    def __init__(self, path):
        self.path = path
        # 语义分割需要的图片的加载进来，做标签，总共2913张图片
        self.file_list = []
        self.name =[]
        def getallpng(path):
            allfiles = os.listdir(path)
            for eachfile in allfiles:
                if eachfile.split(".")[-1] == "png":
                    self.file_list.append(os.path.join(path, eachfile))
                    self.name.append(eachfile.split("/")[-1])
                else:
                    newpath = os.path.join(path, eachfile)
                    getallpng(newpath)
            return self.file_list,self.name
        self.file_list, self.name=getallpng(self.path)
        self.trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    def __len__(self):
        return len(self.name)

    # 简单的正方形转换，把图片和标签转为正方形
    # 图片会置于中央，两边会填充为黑色，不会失真
    def __trans__(self, img, size,filepath):
        # 图片的宽高
        h, w = img.shape[0:2]
        # 需要的尺寸
        _w = _h = size
        # 不改变图像的宽高比例
        scale = min(_h / h, _w / w)
        h = int(h * scale)
        w = int(w * scale)
        # 缩放图像
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_CUBIC)
        # 上下左右分别要扩展的像素数
        top = (_h - h) // 2
        left = (_w - w) // 2
        bottom = _h - h - top
        right = _w - w - left
        # 生成一个新的填充过的图像，这里用纯黑色进行填充(0,0,0)
        new_img_img_o = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        new_img_img_binary = cv2.threshold(new_img_img_o, 122, 255, cv2.THRESH_BINARY)[1]
        # 将有数值的区域全都变换为对应的数字。
        thefigure =int(filepath.split("\\")[-2])
        # new_img_img_l=new_img_img_o/255*thefigure
        r = new_img_img_binary[:, :, 0]
        g = new_img_img_binary[:, :, 1]
        b = new_img_img_binary[:, :, 2]
        if thefigure==0:#品红色 （255,0,255）
            g=g*0
        elif thefigure==1:#蓝色 （0,0,255）
            r=r*0
            g=g*0
        elif thefigure==2:#青色 （0,255,255）
            r=r*0
        elif thefigure==3:#绿色 （0,255,0）
            r=r*0
            b=b*0
        elif thefigure==4:#黄色 （255,255,0）
            b=b*0
        elif thefigure==5:#红色 （255,0,0）
            g=g*0
            b=b*0
        elif thefigure==6:#紫色 (128,0,128)
            r=(r/255)*128
            g=g*0
            b=(b/255)*128
        elif thefigure==7:#深蓝色 （0,128,128）
            r=r*0
            g=(g/255)*128
            b=(b/255)*128
        elif thefigure==8:#深绿色 （0,128,0）
            r=r*0
            g=(g/255)*128
            b=b*0
        elif thefigure==9:#粉红 255,192,203
            r=(r/255)*255
            g=(g/255)*192
            b=(b/255)*203
        new_img_img_l = cv2.merge([r.astype(np.float), g.astype(np.float), b.astype(np.float)])
        return new_img_img_binary,new_img_img_l

    def __getitem__(self, index):
        # 拿到的图片
        file = self.file_list[index]
        # 读取原始图片和标签，并转RGB
        img_ori = cv2.imread(file)
        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        # 转成网络需要的正方形
        img_o,img_l = self.__trans__(img_ori, 128,file)
        return self.trans(img_o),self.trans(img_l),file

if __name__ == '__main__':
    i = 1
    # 路径改一下
    dataset = UnetDatasets(r"E:\PythonProject\UnetSeries\UNet-VAE\data\train")
    for a, b in dataset:
        print(i)
        print(a.shape)
        print(b.shape)
        # save_image(a, f"./img/{i}.jpg", nrow=1)
        # save_image(b, f"./img/{i}.png", nrow=1)
        i += 1
        if i > 2:
            break
