import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial


nonlinearity = partial(F.relu, inplace=True)

class ChannelMeanMaxAttention(nn.Module):
    # 结合通道均值和通道最大值的注意力机制模块，被称为
    # ChannelMeanMaxAttention。这个模块的目的是引入通道间的不同关注程度，
    # 通过学习得到每个通道的权重，使得网络在处理输入时能够更加关注重要的通道。
    def __init__(self, num_channels):
        super(ChannelMeanMaxAttention, self).__init__()
        num_channels_reduced = num_channels // 2  # 计算了经过注意力模块后输出通道数的缩减量 ,这里将输出通道数缩减为输入通道数的一半
        self.fc1 = nn.Linear(num_channels, num_channels_reduced, bias=True)  # 是一个全连接层
        self.fc2 = nn.Linear(num_channels_reduced, num_channels, bias=True)  # 将输入特征的每个通道与一个学习到的权重相乘，并加上一个学习到的偏置，从而进行通道间的线性变换。这个操作有助于模型学习输入通道之间的关联性，以便更好地捕捉通道之间的信息。
        self.relu = nonlinearity

    def forward(self, input_tensor):
        batch_size, num_channels, H, W = input_tensor.size()
        squeeze_tensor_mean = input_tensor.view(batch_size, num_channels, -1).mean(dim=2)  # 均值池化

        fc_out_1_mean = self.relu(self.fc1(squeeze_tensor_mean))  # 对输入特征张量的通道维度进行均值池化
        fc_out_2_mean = self.fc2(fc_out_1_mean)

        squeeze_tensor_max = input_tensor.view(batch_size, num_channels, -1).max(dim=2)[0]  # 最大值池化
        fc_out_1_max = self.relu(self.fc1(squeeze_tensor_max))
        fc_out_2_max = self.fc2(fc_out_1_max)
        # 得到两个注意力权重张量

        a, b = squeeze_tensor_mean.size()
        result = torch.Tensor(a, b)
        result = torch.add(fc_out_2_mean, fc_out_2_max)
        fc_out_2 = F.sigmoid(result)
        output_tensor = torch.mul(input_tensor, fc_out_2.view(a, b, 1, 1))
        return output_tensor


class SpatialAttention(nn.Module):
    # 通过对输入特征图进行平均池化和最大池化，生成一个包含空间信息的辅助通道，然后通过卷积操作和
    # Sigmoid激活函数生成注意力权重，最终将输入特征与注意力权重相乘，以调整每个空间位置的重要性。
    def __init__(self, kernel_size=5):
        super(SpatialAttention, self).__init__()
        padding = 2
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        #print("x.shape000", x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征张量进行平均池化和最大池化，得到两个通道。
        x = torch.cat([avg_out, max_out], dim=1)  #将这两个通道合并为一个通道，构成输入通道为 2 的张量。
        #print("x.shape111", x.shape)
        x = self.conv1(x)
        #print("x.shape222", x.shape)
        output = self.sigmoid(x) * input_tensor
        return output # 通过卷积层 self.conv1 和 Sigmoid 激活函数生成注意力权重


class SpatialAttention1(nn.Module):
    # 通过对输入特征图进行平均池化和最大池化，生成一个包含空间信息的辅助通道，然后通过卷积操作和
    # Sigmoid激活函数生成注意力权重，最终将输入特征与注意力权重相乘，以调整每个空间位置的重要性。
    def __init__(self, kernel_size=3):
        super(SpatialAttention1, self).__init__()
        padding = 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_tensor = x
        #print("x.shape000", x.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 对输入特征张量进行平均池化和最大池化，得到两个通道。
        x = torch.cat([avg_out, max_out], dim=1)  #将这两个通道合并为一个通道，构成输入通道为 2 的张量。
        #print("x.shape111", x.shape)
        x = self.conv1(x)
        #print("x.shape222", x.shape)
        output = self.sigmoid(x) * input_tensor
        return output # 通过卷积层 self.conv1 和 Sigmoid 激活函数生成注意力权重



class EdgeDetectionModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeDetectionModule, self).__init__()
        # 使用Sobel卷积核进行水平和垂直边缘检测
        self.conv_horizontal = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 3), padding=(0, 1), bias=False)
        self.conv_vertical = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 1), padding=(1, 0), bias=False)

    def forward(self, x):
        # 分别对输入进行水平和垂直边缘检测
        edge_horizontal = self.conv_horizontal(x)
        edge_vertical = self.conv_vertical(x)

        # 合并水平和垂直边缘信息
        edges = torch.sqrt(edge_horizontal ** 2 + edge_vertical ** 2)
        return edges