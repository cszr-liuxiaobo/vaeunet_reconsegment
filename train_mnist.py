"""
训练器模块
"""
import os

from sklearn.manifold import TSNE

from model import UNet,VAE
import torch
import dataprocessing
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib import pyplot as plt

# 训练器
class Trainer:
    def __init__(self, path, model, model_copy, img_save_path):
        self.path = path
        self.model = model
        self.model_copy = model_copy
        self.img_save_path = img_save_path
        # 使用的设备
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # 网络
        self.net = UNet().to(self.device)
        self.vaenet = VAE().to(self.device)
        # 优化器，这里用的Adam，跑得快点
        self.opt = torch.optim.Adam(self.net.parameters())
        self.vaeopt = torch.optim.Adam(self.vaenet.parameters())
        # 这里直接使用二分类交叉熵来训练，效果可能不那么好
        # 可以使用其他损失，比如DiceLoss、FocalLoss之类的
        self.loss_func = nn.BCELoss()
        # 设备好，batch_size和num_workers可以给大点
        self.loader = DataLoader(dataprocessing.UnetDatasets(path), batch_size=40, shuffle=True, num_workers=8)
        self.testloader = DataLoader(dataprocessing.UnetDatasets(path[:-5]+"test"), batch_size=40, shuffle=True, num_workers=8)

        # 判断unet模型是否存在模型
        if os.path.exists(self.model):
            self.net.load_state_dict(torch.load(model))
            print(f"Loaded{model}!")
        else:
            print("No unet Param!")
        # 判断unet模型是否存在模型

        if os.path.exists("./vaeunetmodel.plt"):
            self.vaenet.load_state_dict(torch.load("./vaeunetmodel.plt"))
            print(f"Loaded vaemodel!")
        else:
            print("No vae Param!")
        os.makedirs(img_save_path, exist_ok=True)

    # 训练
    def unettrain(self, stop_value):
        epoch = 1
        while True:
            batch_i = 0
            for inputs, labels,filename in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                batch_i+=1
                # 图片和分割标签
                inputs, labels = inputs.to(self.device), labels.to(self.device)/255

                # 输出生成的图像
                out = self.net(inputs)
                loss = self.loss_func(out, labels.float())
                # 后向
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                # 输入的图像，取第一张
                x = inputs[0]
                # 生成的图像，取第一张
                x_ = out[0]
                # 标签的图像，取第一张
                y = labels[0]
                thedir = int(filename[0].split("\\")[-2])
                thename = filename[0].split("\\")[-1]

                # 三张图，从第0轴拼接起来，再保存
                if batch_i%15 ==0:
                    ax[0].imshow(np.transpose(x.cpu().detach().numpy(),(1,2,0)), cmap='gray')
                    ax[1].imshow(np.transpose(x_.cpu().detach().numpy(),(1,2,0)), cmap='gray')
                    ax[2].imshow(np.transpose(y.cpu().detach().numpy(),(1,2,0)), cmap='gray')

                    plt.savefig(os.path.join(self.img_save_path, f"epoch-{epoch}_batch-{batch_i}_dir-{thedir}_name-{thename}"))
                # img = torch.stack([x, x_, y], 0)
                # save_image(img.cpu(), os.path.join(self.img_save_path, f"epoch-{epoch}_batch-{batch_i}_dir-{thedir}_name-{thename}"))
                # print("image save successfully !")
            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")
            torch.save(self.net.state_dict(), self.model)
            # print("model is saved !")

            # 备份
            if epoch % 50 == 0:
                torch.save(self.net.state_dict(), self.model_copy.format(epoch, loss))
                print("model_copy is saved !")
            if epoch > stop_value:
                break
            epoch += 1

    # 将UNET置为eval，然后将训练模式下的vae生成的数据传递到Unet中，计算loss，同时显示隐空间的分布情况
    def vaetrain(self, stop_value):
        self.net.eval()
        self.vaenet.train()
        epoch = 1
        while True:
            batch_i=0
            for inputs, labels,filename in tqdm(self.loader, desc=f"Epoch {epoch}/{stop_value}",
                                       ascii=True, total=len(self.loader)):
                batch_i+=1
                # 图片和分割标签
                inputs, labels = inputs.to(self.device)[:,0:1,:,:], labels.to(self.device)/255
                # 输出生成的图像
                x = inputs.view(-1,128*128)
                x_reconst, mu, log_var,z = self.vaenet(x)

                # 计算重构损失和KL散度
                # 重构损失
                reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
                # KL散度
                kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
                vae_image_out=x_reconst.view(-1, 1, 128, 128)
                # 将vae生成的数据放入unet中
                vae_image_out_rgb=torch.cat((vae_image_out, vae_image_out, vae_image_out), 1)
                out = self.net(vae_image_out_rgb)
                unetloss = self.loss_func(out, labels.float())
                loss = unetloss+kl_div+reconst_loss
                print("kl_divloss:",kl_div)
                print("unetloss:",unetloss)
                print("loss:",loss)

                # 后向传播
                self.vaeopt.zero_grad()
                loss.backward()
                self.vaeopt.step()

            print(f"\nEpoch: {epoch}/{stop_value}, Loss: {loss}")
            torch.save(self.vaenet.state_dict(), "./vaeunetmodel.plt")
            # print("model is saved !")

            # 备份
            if epoch % 10 == 0:
                torch.save(self.vaenet.state_dict(), "./vaeunetmodel_{}_{}.plt".format(epoch, loss))
                print("model_copy is saved !")
            # 验证一轮并绘图
            self.vaeunetevalue(epoch)

            if epoch > stop_value:
                break
            epoch += 1

    def vaeunetevalue(self,epoch_now):
        y_np = []
        z_np = []
        batch_i=0
        self.net.eval()
        self.vaenet.eval()

        for inputs, labels, filename in tqdm(self.testloader,ascii=True, total=len(self.testloader)):
            batch_i += 1
            # 图片和分割标签
            inputs, labels = inputs.to(self.device)[:, 0:1, :, :], labels.to(self.device) / 255
            # 输出生成的图像
            x = inputs.view(-1,128 * 128)
            x_reconst, mu, log_var, z = self.vaenet(x)

            dirnumbers=[]
            for i in filename:
                dirnumbers.append(int(i.split("\\")[-2]))

            thedir = int(filename[0].split("\\")[-2])
            thename = filename[0].split("\\")[-1]
            # 记录 latent space 输出情况
            y_cpu = dirnumbers
            z_cpu = z.cpu().detach().numpy()
            y_np.extend(y_cpu)
            z_np.extend(z_cpu)  # batch*20个大小

            # 计算重构损失和KL散度
            # 重构损失
            # reconst_loss = F.binary_cross_entropy(x_reconst, x, size_average=False)
            # KL散度
            # kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            vae_image_out = x_reconst.view(-1, 1, 128, 128)
            vae_image_out_rgb = torch.cat((vae_image_out, vae_image_out, vae_image_out), 1)
            # 将vae生成的数据放入unet中
            out = self.net(vae_image_out_rgb)

            # 后向传播
            self.vaeopt.step()

            # 输入的图像，取第一张
            x = inputs[0]
            # 记录latentspace输出情况-散点图
            # 完成一个epoch后再展示,放在plotdistribution中进行

            # vae生成的图像，第一张
            x_vae = vae_image_out[0]
            # 输入的第一张图像
            x_unet = vae_image_out_rgb[0]
            # 生成的图像，取第一张
            x_ = out[0]
            # 标签的图像，取第一张
            y = labels[0]

            # 三张图，从第0轴拼接起来，再保存
            if batch_i % 25 == 0:
                # 输入图像，散点图分布，vae输出图像，unet生成的分割图像，真实标签
                ax[0,0].imshow(np.transpose(x.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
                plotdistribution(y_np, z_np)
                ax[0,2].imshow(np.transpose(x_vae.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
                ax[1,0].imshow(np.transpose(x_unet.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
                ax[1,1].imshow(np.transpose(x_.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
                ax[1,2].imshow(np.transpose(y.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
                plt.savefig(
                    os.path.join(self.img_save_path, f"epoch-{epoch_now}_batch-{batch_i}_dir-{thedir}_name-{thename}"))
                ax[0, 1].cla()
        # 输出全部数据后的图像
        ax[0,0].imshow(np.transpose(x.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        plotdistribution(y_np, z_np)
        ax[0,2].imshow(np.transpose(x_vae.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        ax[1,0].imshow(np.transpose(x_vae.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        ax[1,1].imshow(np.transpose(x_.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        ax[1,2].imshow(np.transpose(y.cpu().detach().numpy(), (1, 2, 0)), cmap='gray')
        plt.savefig(
            os.path.join(self.img_save_path, f"epoch-{epoch_now}_batch-{batch_i}_dir-{thedir}_name-{thename}"))
        ax[0, 1].cla()

        if not os.path.exists("./result/"):
            os.mkdir("./result/")
        if os.path.exists("./result/eval_label_epoch_.npy".format(epoch_now)):
            os.remove("./result/eval_label_epoch_.npy".format(epoch_now))
        if os.path.exists("./result/eval_data_epoch.npy".format(epoch_now)):
            os.remove("./result/eval_data_epoch.npy".format(epoch_now))
        np.save("./result/eval_label_epoch_{}.npy".format(epoch_now), y_np)
        np.save("./result/eval_data_epoch{}.npy".format(epoch_now), z_np)

def plotdistribution(Label,Mat):
    """
    :param Label: 点的类别标签
    :param Mat: 二维点坐标矩阵
    :return:
    """
    tsne = TSNE(n_components=2, random_state=0)
    Mat = tsne.fit_transform(Mat[:])

    x = Mat[:, 0]
    y = Mat[:, 1]
    # map_size = {0: 5, 1: 5}
    # size = list(map(lambda x: map_size[x], Label))
    map_color = {0: [1,0,1], 1: [0,0,1],2:[0,1,1],3:[0,1,0],4:[1,1,0],5:[1,0,0],6:[0.5,0,0.5],7:[0,0.5,0.5],8:[0,0.5,0],9:[1,0.75,0.79]}
    color = list(map(lambda x: map_color[x], Label))
    # 代码会出错，因为marker参数不支持列表
    # map_marker = {-1: 'o', 1: 'v'}
    # markers = list(map(lambda x: map_marker[x], Label))
    #  plt.scatter(np.array(x), np.array(y), s=size, c=color, marker=markers)
    # 下面一行代码为修正过的代码
    ax[0,1].scatter(np.array(x), np.array(y), s=3, c=color, marker='o')  # scatter函数只支持array类型数据

if __name__ == '__main__':
    # 路径改一下
    fig, ax = plt.subplots(2, 3)

    # t = Trainer(r"E:\PythonProject\UnetSeries\UNet-VAE\data\train", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./train_img')
    # t.unettrain(300)

    t=Trainer(r"E:\PythonProject\UnetSeries\UNet-VAE\data\train", r'./model.plt', r'./model_{}_{}.plt', img_save_path=r'./vaetrain_img')
    t.vaetrain(2000)
    # for i in range(2):
    #     t.vaeunetevalue(i)
