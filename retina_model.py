import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import itertools
import numpy as np
import math
import random

class retina_polar(nn.Module):
    '''
    Log polar transformation
    '''
    def __init__(
            self,
            r_min=0.05,
            r_max=0.8,
            H=5,
            W=12,
            upsampling_factor_r=10,  # 对r轴进行放缩
            upsampling_factor_theta=10,  # 对theta轴进行放缩
            log_r=True,
            retina_field=1,
    ):
        super(retina_polar, self).__init__()
        if log_r:
            sample_r_log = np.linspace(
                np.log(r_min), np.log(r_max), num=upsampling_factor_r * H,
            )
            sample_r = np.exp(sample_r_log)

        else:
            sample_r = np.linspace(r_min, r_max, num=upsampling_factor_r * H)

        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        angles = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 1]
        )
        for h in range(H * upsampling_factor_r):
            radius = sample_r[h]
            for w in range(W * upsampling_factor_theta):
                angle = 2 * np.pi * w / W
                grid_2d[h, w] = torch.Tensor(
                    # 原来作者的代码坐标写反了
                    [radius, radius]
                )
                angles[h, w] = torch.Tensor(
                    [angle]
                )
        self.H = H
        self.W = W
        self.retina_field = retina_field
        self.register_buffer("radius", grid_2d)
        self.register_buffer("angles", angles)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def get_grid(self, b):
        radius = self.radius[None].clone().repeat(b, 1, 1, 1)
        angles = self.angles[None].clone().repeat(b, 1, 1, 1)
        angle = angles

        grid = torch.zeros_like(radius).to(device=radius.device)
        grid[:, :, :, 0] = radius[:, :, :, 0] * torch.sin(angle[:, :, :, 0])
        grid[:, :, :, 1] = radius[:, :, :, 1] * torch.cos(angle[:, :, :, 0])
        return grid

    def forward(self, x, l_t_prev):
        batch_size, *_ = x.shape
        l_t_prev = l_t_prev.to(x.device)
        grid_2d_batch = self.get_grid(batch_size) + l_t_prev.view(-1, 1, 1, 2)
        sampled_points = F.grid_sample(x, grid_2d_batch * self.retina_field, padding_mode='border')
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points


class inverse_retina_polar(nn.Module):
    def __init__(
            self,
            r_min=0.01,
            r_max=0.6,
            retinal_H=5,
            retinal_W=5,
            H=5,
            W=12,
            upsampling_factor_r=10,  # 对r轴进行放缩
            upsampling_factor_theta=10,  # 对theta轴进行放缩
    ):
        super(inverse_retina_polar, self).__init__()
        self.H = H
        self.W = W
        self.r_min = r_min
        self.r_max = r_max
        grid_2d = torch.empty(
            [H * upsampling_factor_r, W * upsampling_factor_theta, 2]
        )
        for i in range(H):
            for j in range(W):
                x = (i - int(H / 2)) / (H / 2)  # 这里除以H/2的依据是，r轴长H/2就覆盖了整个面积，然后归一化
                y = (j - int(W / 2)) / (W / 2)
                r = retinal_H * (np.log(np.clip(np.sqrt(x ** 2 + y ** 2), 1e-6, (H ** 2 + W ** 2))) - np.log(r_min)) / (
                            np.log(r_max) - np.log(r_min))
                a = np.arctan2(y, x)
                a = a if a > 0 else 2.0 * np.pi + a
                t = 0.5 * a * retinal_W / np.pi
                grid_2d[i, j] = torch.Tensor(
                    [t / (retinal_W / 2) - 1, r / (retinal_H / 2) - 1]
                )
        self.register_buffer("grid_2d", grid_2d)
        self.avg_pool = nn.AvgPool2d([upsampling_factor_r, upsampling_factor_theta])

    def forward(self, x, l_t_prev):
        l_t_prev = l_t_prev.to(x.device)
        grid_2d_batch = l_t_prev.view(-1, 1, 1, 2) * 0 + self.grid_2d[None]
        sampled_points = F.grid_sample(x, grid_2d_batch, padding_mode='border')
        sampled_points = self.avg_pool(sampled_points)
        return sampled_points

class Retina(nn.Module):
    """
    A brain-inspired retina module
    """
    expansion = 4
    def __init__(
            self,
            r_min=0.01,
            r_max=1.2,
            image_H=224, # 输出图像大小(retina size)
            image_W=224,
            retinal_H=224, # 中间变换图像大小
            retinal_W=224,
            upsampling_factor_r=1,
            upsampling_factor_theta=1,
            log_r=True,
            retina_field=1, # 感受野
            em_number=4, # 眼动点
            sampling_model='gaussian', # 眼动点选取模式
    ):
        super(Retina, self).__init__()
        self.em_number = em_number
        self.retina = retina_polar(
            r_min=r_min,
            r_max=r_max,
            H=retinal_H,
            W=retinal_W,
            upsampling_factor_r=upsampling_factor_r,
            upsampling_factor_theta=upsampling_factor_theta,
            log_r=log_r,
            retina_field=retina_field,
        )
        self.inverse_retina = inverse_retina_polar(
            r_min,
            r_max,
            retinal_H,
            retinal_W,
            image_H,
            image_W,
            upsampling_factor_r,
            upsampling_factor_theta,
        )
        self.retina_size = retinal_H
        if em_number==4: # 固定一下方便对比
            self.l_t = torch.tensor([[0.0000, 0.0000],
                    [0.2006, 0.2006],
                    [-0.2455, 0.2006],
                    [0.2006, -0.2455]])
        else:
            point_number = self.nearest_square_root(em_number)
            if sampling_model == 'uniform':
                random_x = 2 * torch.rand(int(np.sqrt(point_number))) - 1
                combinations = list(itertools.product(random_x, repeat=2))
                sample_points = combinations[:point_number]
                sample_points = torch.tensor(sample_points)
            elif sampling_model == 'gaussian':
                random_x = torch.randn(int(np.sqrt(point_number))) / 2
                random_x[random_x > 1] = 1
                random_x[random_x < -1] = -1
                combinations = list(itertools.product(random_x, repeat=2))
                sample_points = combinations[:point_number]
                sample_points = torch.tensor(sample_points)
            indices = torch.randperm(sample_points.size(0))[:em_number - 1]
            center = torch.tensor((0, 0)).unsqueeze(0)
            sample_points = torch.cat([center, sample_points[indices, :]], 0)
            self.l_t = sample_points
        print(self.l_t)
    def nearest_square_root(self, num):
        nearest = int(math.sqrt(num))
        while nearest * nearest < num:
            nearest += 1
        return nearest * nearest

    def forward(self, x):
        batch_size, c, h, w = x.shape
        x = x.unsqueeze(1).repeat(1,self.em_number,1,1,1).reshape(-1, c, h, w)
        l_t = self.l_t.repeat(batch_size, 1, 1)
        g_t = self.retina(x, l_t)
        i_t = self.inverse_retina(g_t, l_t)
        return i_t.reshape(batch_size,-1,c,self.retina_size, self.retina_size), l_t


if __name__ == '__main__':
    '''
        支持任意变换尺寸
        支持任意眼动点数量
        支持两种眼动点选取，均匀和高斯
        支持支持
    '''
    retina = Retina(
            image_H=112, # 输出图像大小(retina size)
            image_W=112,
            retinal_H=112, # 中间变换图像大小
            retinal_W=112,
            retina_field=1, # 感受野
            em_number=4, # 眼动点
            sampling_model='uniform', # 眼动点选取模式
        ).cuda()
    from PIL import Image
    from torchvision import transforms
    # 读取图像
    image_path = r"F:\dateset\Algonauts2021_data\images\1image-0001.jpg"
    image = Image.open(image_path)
    # 定义图像变换
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    # 应用变换并添加批量维度
    x = preprocess(image).unsqueeze(0).cuda()
    # 输入图像，输出retina图和眼动点
    y, l_t = retina(x)
    print(y.shape, l_t)

    import matplotlib.pyplot as plt
    for i in range(4):
        plt.imshow(y[0][i].permute(1,2,0).cpu().detach().numpy())
        plt.show()