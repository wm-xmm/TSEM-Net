"""
ALPModule
"""
import os.path

import torch
import math
from torch import nn
from torch.nn import functional as F
import numpy as np
from pdb import set_trace
import matplotlib.pyplot as plt
from .CS_Att import SpatialAttention
from .CS_Att import EdgeDetectionModule
import SimpleITK as sitk
from dataloaders.niftiio import convert_to_sitk


# for unit test from spatial_similarity_module import NONLocalBlock2D, LayerNorm

class MultiProtoAsConv(nn.Module):
    def __init__(self, proto_grid, feature_hw, upsample_mode = 'bilinear'):
        """
        ALPModule
        Args:
            proto_grid:     Grid size when doing multi-prototyping. For a 32-by-32 feature map, a size of 16-by-16 leads to a pooling window of 2-by-2
            feature_hw:     Spatial size of input feature map 输入特征图的空间大小

        """
        super(MultiProtoAsConv, self).__init__()
        self.proto_grid = proto_grid
        self.upsample_mode = upsample_mode
        kernel_size = [ft_l // grid_l for ft_l, grid_l in zip(feature_hw, proto_grid)  ] # proto_grid=[8,8], feature_hw=[32,32],kernel_size=[4,4]
        self.avg_pool_op = nn.AvgPool2d(kernel_size)  # 平均池化 kernel_size=[4,4]
        self.spa = SpatialAttention()
        self.edge = EdgeDetectionModule(256, 256)



    def forward(self, qry, sup_x, sup_y, mode, thresh, isval = False, val_wsize = None, vis_sim = False, **kwargs):
        """
        Now supports
        Args:
            mode: 'mask'/ 'grid'. if mask, works as original prototyping
            qry: [way(1), nc, h, w]
            sup_x: [nb, nc, h, w]
            sup_y: [nb, 1, h, w]
            vis_sim: visualize raw similarities or not
        New
            mode:       'mask'/ 'grid'. if mask, works as original prototyping
            qry:        [way(1), nb(1), nc, h, w]
            sup_x:      [way(1), shot, nb(1), nc, h, w]
            sup_y:      [way(1), shot, nb(1), h, w]
            vis_sim:    visualize raw similarities or not
        """
        qry = qry.squeeze(1) # [way(1), nb(1), nc, hw] -> [way(1), nc, h, w]
        sup_x = sup_x.squeeze(0).squeeze(1) # [nshot, nc, h, w]
        sup_y = sup_y.squeeze(0) # [nshot, 1, h, w]
        #print("sup_x000:",sup_x.shape)

        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            # 归一化的操作在深度学习中常用于处理梯度爆炸或梯度消失的问题
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x

        if mode == 'mask': # class-level prototype only  类级原型通常与分类任务和监督学习相关
            proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                / (sup_y.sum(dim=(-1, -2)) + 1e-5) # nb x C

            proto = proto.mean(dim = 0, keepdim = True) # 1 X C, take the mean of everything
            pred_mask = F.cosine_similarity(qry, proto[..., None, None], dim=1, eps = 1e-4) * 20.0 # [1, h, w]

            vis_dict = {'proto_assign': None} # things to visualize
            if vis_sim:
                vis_dict['raw_local_sims'] = pred_mask
            return pred_mask.unsqueeze(1), [pred_mask], vis_dict  # just a placeholder. pred_mask returned as [1, way(1), h, w]

        # no need to merge with gridconv+
        elif mode == 'gridconv': # using local prototypes only  #局部原型通常与聚类和无监督学习相关

            input_size = qry.shape  # 查询图像的shape
            nch = input_size[1]  #nch=256
            sup_nshot = sup_x.shape[0]
            #print("sup_x_shape:", sup_x.shape)


            # adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
            # n_sup_x = adaptive_avg_pool(sup_x)
            # sup_x = self.spa(sup_x)  # sup_x=([1, 256, 32, 32])，n_sup_x=([1, 256, 8, 8])
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)  # 每个局部原型只会在一个覆盖在support上的局部池化窗口中计算
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0) # way(1),nb, hw, nc 对平均池化后的支持集进行形状变换，以便后续的计算。
            # print("sup_x_shape222:", n_sup_x.shape)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0) # 进一步对形状进行调整，以适应后续的计算。
            #print("sup_x_shape333:", n_sup_x.shape)


            #sup_y = self.spa(sup_y)
            # adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
            # sup_y_g = adaptive_avg_pool(sup_y)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)  # 对支持集的标签 sup_y 进行平均池化，类似于对输入的处理
            #print("sup_y111:", sup_y_g.shape)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            #  print('sup_y_g:', sup_y_g)
            protos = n_sup_x[sup_y_g >= thresh, :] # npro, nc 使用阈值 thresh 来筛选满足条件的原型。
            pro_n = safe_norm(protos)
            qry_n = safe_norm(qry)
            if pro_n.shape[0]!=0:
                dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20  # qry_n 是查询的特征图，pro_n 是原型的特征图。pro_n[..., None, None] 的目的可能是在 pro_n 的最后两个维度上添加新的维度，
                # 以便与 qry_n 进行卷积。这个卷积的结果是一个二维的张量。
                pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)  # 融合每一个局部相似类
                #  对经过 softmax 的结果与原始卷积结果相乘，并在第一个维度上求和。keepdim=True 保持维度
                debug_assign = dists.argmax(dim = 1).float().detach()
                vis_dict = {'proto_assign': debug_assign} # things to visualize
                if vis_sim: # return the similarity for visualization
                    vis_dict['raw_local_sims'] = dists.clone().detach()
                return pred_grid, [debug_assign], vis_dict



        elif mode == 'gridconv+': # local and global prototypes ，sup_y就是根据超像素生成的前景标签掩码

            input_size = qry.shape
            nch = input_size[1]
            sup_nshot = sup_x.shape[0]
            #print("sup_x000:", sup_x.shape)
            # sup_x = self.spa(edges)
            # adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
            # n_sup_x = adaptive_avg_pool(sup_x)
            edges = self.edge(sup_x)
            edges = self.spa(edges)
            sup_x = edges + sup_x
            #print("sup_x111:", sup_x.shape)
            n_sup_x = F.avg_pool2d(sup_x, val_wsize) if isval else self.avg_pool_op(sup_x)   # 这里进行前景的原型计算
            n_sup_x = n_sup_x.view(sup_nshot, nch, -1).permute(0,2,1).unsqueeze(0)
            n_sup_x = n_sup_x.reshape(1, -1, nch).unsqueeze(0)  # -1是一个自动计算的值

            # sup_y = self.spa(sup_y)
            # adaptive_avg_pool = nn.AdaptiveAvgPool2d((8, 8))
            # sup_y_g = adaptive_avg_pool(sup_y)
            sup_y_g = F.avg_pool2d(sup_y, val_wsize) if isval else self.avg_pool_op(sup_y)
            sup_y_g = sup_y_g.view( sup_nshot, 1, -1  ).permute(1, 0, 2).view(1, -1).unsqueeze(0)
            protos = n_sup_x[sup_y_g > thresh, :]  # 从 n_sup_x 中选择出那些对应的 sup_y_g 大于 thresh 的行。


            glb_proto = torch.sum(sup_x * sup_y, dim=(-1, -2)) \
                     / (sup_y.sum(dim=(-1, -2)) + 1e-5)  # (这里是类级原型）\连接符， /对上述两个结果进行逐元素相除，其中 1e-5 是一个很小的常数，防止除法中的除零错误
            # 这两行代码的目的是计算每个类别的加权平均特征，其中权重是由 sup_y 决定的（可能是类别的像素数量）
            # sup_x,sup_y代表预测和标签,sup_x * sup_y 执行了逐元素相乘的操作，即将对应位置的元素相乘。这可能表示一种逐元素的加权或注意力操作。
            # dim=(-1, -2)沿着最后两个维度进行求和，即对 height 和 width 进行求和。这可以看作是对每个类别的加权特征的全局池化操作。
            pro_n = safe_norm( torch.cat( [protos, glb_proto], dim = 0 ) )
            qry_n = safe_norm(qry)

            dists = F.conv2d(qry_n, pro_n[..., None, None]) * 20  # dists 的输出形式是一个包含相似度值的张量，其中每个元素表示查询图像集合中的某个图像与支持图像集合中的某个图像之间的相似度。
            # 这个相似度矩阵的形状为 (batch_size, num_query, num_support, qry_height, qry_width)。
            pred_grid = torch.sum(F.softmax(dists, dim = 1) * dists, dim = 1, keepdim = True)
            debug_assign = dists.argmax(dim = 1).float()  # debug_assign包含了每个查询点所对应的最近邻的索引，这个索引是在支持点中距离最近的点的位置。
            # argmax方法来找到张量dists中每行最大值所在的列索引，并将结果转换为浮点数类型。

            vis_dict = {'proto_assign': debug_assign}  # 'proto_assign': 一个张量，表示样本分配给原型的情况。
            if vis_sim:
                vis_dict['raw_local_sims'] = dists.clone().detach()  # raw_local_sims': 一个张量，表示原始局部相似性。这可能包含模型计算的原始相似度分数。
            return pred_grid, [debug_assign], vis_dict  # 1包含了预测的结果,2可能包含了分配给每个样本或位置的某种标签或索引,3用于可视化的数据

        else:
            raise NotImplementedError

