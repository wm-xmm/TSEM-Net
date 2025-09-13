"""
Metrics for computing evalutation results
Modified from vanilla PANet code by Wang et al.
"""

import numpy as np
from scipy.spatial.distance import directed_hausdorff
import cv2
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from scipy.spatial import ConvexHull


class Metric(object):
    """
    Compute evaluation result

    Args:
        max_label:
            max label index in the data (0 denoting background)
        n_scans:
            number of test scans
    """
    def __init__(self, max_label=13, n_scans=None):
        self.labels = list(range(max_label + 1))  # all class labels
        self.n_scans = 1 if n_scans is None else n_scans  # n_scans=5

        # list of list of array, each array save the TP/FP/FN statistic of a testing sample
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]
        self.tn_lst = [[] for _ in range(self.n_scans)]
        self.pred_lst = [[] for _ in range(self.n_scans)]
        self.qrey_lst = [[] for _ in range(self.n_scans)]

    def reset(self):
        """
        Reset accumulated evaluation. 
        """
        # assert self.n_scans == 1, 'Should not reset accumulated result when we are not doing one-time batch-wise validation'
        del self.tp_lst, self.fp_lst, self.fn_lst, self.tn_lst, self.qrey_lst, self.pred_lst
        self.tp_lst = [[] for _ in range(self.n_scans)]
        self.fp_lst = [[] for _ in range(self.n_scans)]
        self.fn_lst = [[] for _ in range(self.n_scans)]
        self.tn_lst = [[] for _ in range(self.n_scans)]
        self.pred_lst = [[] for _ in range(self.n_scans)]
        self.qrey_lst = [[] for _ in range(self.n_scans)]

    def record(self, pred, target, labels=None, n_scan=None):
        """
        Record the evaluation result for each sample and each class label, including:
            True Positive, False Positive, False Negative

        Args:
            pred:
                predicted mask array, expected shape is H x W
            target:
                target mask array, expected shape is H x W
            labels:
                only count specific label, used when knowing all possible labels in advance
        """
        assert pred.shape == target.shape
        # print("target:", target.max())
        # print("n_scans:", self.n_scans)
        if self.n_scans == 1:
            n_scan = 0

        # array to save the TP/FP/FN statistic for each class (plus BG)
        tp_arr = np.full(len(self.labels), np.nan)  # 创建了一个长度为 len(self.labels) 的 NumPy 数组 tp_arr，元素初始化为 NaN
        fp_arr = np.full(len(self.labels), np.nan)
        fn_arr = np.full(len(self.labels), np.nan)
        tn_arr = np.full(len(self.labels), np.nan)
        # pred_label = np.full((256, 256), np.nan)
        # qrey_label = np.full((256, 256), np.nan)


        if labels is None:
            labels = self.labels
        else:
            labels = [0,] + labels



        for j, label in enumerate(labels):  # j 是标签的索引，label 是标签的值。j=0或1，lbael=[0, 2],[0,1]?
            # Get the location of the pixels that are predicted as class j
            # print("j:", j, "label:", labels)
            # print("n_scan:", n_scan)  # n_scan=[0,5]
            idx = np.where(np.logical_and(pred == j, target != 255))  # 逻辑与操作,查找预测 pred 中被分类为类别 j 且目标 target 不等于 255 的像素的位置
            # print("idx:", idx)
            pred_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))  # idx[0] 是一个包含行坐标的 NumPy 数组。idx[1] 是一个包含列坐标的 NumPy 数组
            # self.pred_lst[n_scan] = zip(idx[0].tolist(), idx[1].tolist())
            # print("pred_idx_j:", pred_idx_j)
            # Get the location of the pixels that are class j in ground truth
            idx = np.where(target == j)
            target_idx_j = set(zip(idx[0].tolist(), idx[1].tolist()))
            # self.qrey_lst[n_scan]= zip(idx[0].tolist(), idx[1].tolist())

            # this should not work: if target_idx_j:  # if ground-truth contains this class
            # the author is adding posion to the code
            tp_arr[label] = len(set.intersection(pred_idx_j, target_idx_j))  # 用于计算这两个集合的交集，即包含在两个集合中的相同位置的像素。
            fp_arr[label] = len(pred_idx_j - target_idx_j)
            fn_arr[label] = len(target_idx_j - pred_idx_j)
            tn_arr[label] = len(set.difference(set.difference(set(range(pred.size)), pred_idx_j), target_idx_j))


            pred_label = {label: [] }
            qrey_label = {label: [] }
            pred_label[label].append(pred)
            qrey_label[label].append(target)
            # print("pred_label:", pred_label)
            # print("len_pred_label:", len(pred_label))

        self.tp_lst[n_scan].append(tp_arr)  # 这里就已经两个类别了
        self.fp_lst[n_scan].append(fp_arr)
        self.fn_lst[n_scan].append(fn_arr)
        self.tn_lst[n_scan].append(tn_arr)
        self.pred_lst[n_scan].append(pred_label)
        self.qrey_lst[n_scan].append(qrey_label)
        key_list = list(self.pred_lst[0][0].keys())
        # if np.all(np.array((self.pred_lst[0][0].get(key_list[0])))[0] == 0):
        #     print("!!!!!!!!!!!!!")


    def get_mIoU(self, labels=None, n_scan=None):
        """
        Compute mean IoU

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely
            # Average across n_scans, then average over classes
            mIoU_class = np.vstack([tp_sum[_scan] / (tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU

    def get_mDice(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute mean Dice score (in 3D scan level)

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]  # self.n_scans=1
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Average across n_scans, then average over classes
            print('tp_sum[_scan]:', tp_sum)
            print('fp_sum[_scan]:', fp_sum)
            print('fn_sum[_scan]:', fn_sum)
            mDice_class = np.vstack([ 2 * tp_sum[_scan] / ( 2 * tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mDice = mDice_class.mean(axis=1)
            print("mDice_class:", mDice_class)
            if not give_raw:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0))
            else:
                return (mDice_class.mean(axis=0), mDice_class.std(axis=0),
                    mDice.mean(axis=0), mDice.std(axis=0), mDice_class)

        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mDice_class = 2 * tp_sum / ( 2 * tp_sum + fp_sum + fn_sum)
            mDice = mDice_class.mean()

            return mDice_class, mDice

    def get_mPrecRecall(self, labels=None, n_scan=None, give_raw = False):
        """
        Compute precision and recall

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely
            # Average across n_scans, then average over classes
            mPrec_class = np.vstack([ tp_sum[_scan] / ( tp_sum[_scan] + fp_sum[_scan] )
                                    for _scan in range(self.n_scans)])

            mRec_class = np.vstack([ tp_sum[_scan] / ( tp_sum[_scan] + fn_sum[_scan] )
                                    for _scan in range(self.n_scans)])

            mPrec = mPrec_class.mean(axis=1)
            mRec  = mRec_class.mean(axis=1)
            if not give_raw:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0))
            else:
                return (mPrec_class.mean(axis=0), mPrec_class.std(axis=0), mPrec.mean(axis=0), mPrec.std(axis=0), mRec_class.mean(axis=0), mRec_class.std(axis=0), mRec.mean(axis=0), mRec.std(axis=0), mPrec_class, mRec_class)


        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0).take(labels)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mPrec_class = tp_sum / (tp_sum + fp_sum)
            mPrec = mPrec_class.mean()

            mRec_class = tp_sum / (tp_sum + fn_sum)
            mRec = mRec_class.mean()

            return mPrec_class, mPrec, mRec_class, mRec

    def get_mSpec(self, labels=None, n_scan=None, give_raw = False):  # Specificity
        """
        Compute Specificity

        Args:
            labels:
                specify a subset of labels to compute mean IoU, default is using all classes
        """
        # NOTE: unverified
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            tn_sum = [np.nansum(np.vstack(self.tn_lst[_scan]), axis=0).take(labels)
                      for _scan in range(self.n_scans)]
            # Compute mean IoU classwisely
            # Average across n_scans, then average over classes
            mSpec_class= np.vstack([ tn_sum[_scan] / ( tn_sum[_scan] + fp_sum[_scan] )
                                    for _scan in range(self.n_scans)])

            mSpec  = mSpec_class.mean(axis=1)
            if not give_raw:
                return (mSpec_class.mean(axis=0), mSpec_class.std(axis=0), mSpec.mean(axis=0), mSpec.std(axis=0))
            else:
                return (mSpec_class.mean(axis=0), mSpec_class.std(axis=0), mSpec.mean(axis=0), mSpec.std(axis=0), mSpec_class)


        else:
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0).take(labels)
            tn_sum = np.nansum(np.vstack(self.tn_lst[n_scan]), axis=0).take(labels)

            # Compute mean IoU classwisely and average over classes
            mSpec_class = tn_sum / (tn_sum + fp_sum)
            mSpec = mSpec_class.mean()

            return mSpec_class, mSpec

    def get_mIoU_binary(self, n_scan=None):
        """
        Compute mean IoU for binary scenario
        (sum all foreground classes as one class)
        """
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            tp_sum = [np.nansum(np.vstack(self.tp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fp_sum = [np.nansum(np.vstack(self.fp_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]
            fn_sum = [np.nansum(np.vstack(self.fn_lst[_scan]), axis=0)
                      for _scan in range(self.n_scans)]

            # Sum over all foreground classes
            tp_sum = [np.c_[tp_sum[_scan][0], np.nansum(tp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fp_sum = [np.c_[fp_sum[_scan][0], np.nansum(fp_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]
            fn_sum = [np.c_[fn_sum[_scan][0], np.nansum(fn_sum[_scan][1:])]
                      for _scan in range(self.n_scans)]

            # Compute mean IoU classwisely and average across classes
            mIoU_class = np.vstack([tp_sum[_scan] / (tp_sum[_scan] + fp_sum[_scan] + fn_sum[_scan])
                                    for _scan in range(self.n_scans)])
            mIoU = mIoU_class.mean(axis=1)

            return (mIoU_class.mean(axis=0), mIoU_class.std(axis=0),
                    mIoU.mean(axis=0), mIoU.std(axis=0))
        else:
            tp_sum = np.nansum(np.vstack(self.tp_lst[n_scan]), axis=0)
            fp_sum = np.nansum(np.vstack(self.fp_lst[n_scan]), axis=0)
            fn_sum = np.nansum(np.vstack(self.fn_lst[n_scan]), axis=0)

            # Sum over all foreground classes
            tp_sum = np.c_[tp_sum[0], np.nansum(tp_sum[1:])]
            fp_sum = np.c_[fp_sum[0], np.nansum(fp_sum[1:])]
            fn_sum = np.c_[fn_sum[0], np.nansum(fn_sum[1:])]

            mIoU_class = tp_sum / (tp_sum + fp_sum + fn_sum)
            mIoU = mIoU_class.mean()

            return mIoU_class, mIoU


    def calculate_average_hausdorff_distance(self, labels=None, n_scan=None, give_raw = False):
        if labels is None:
            labels = self.labels
        # Sum TP, FP, FN statistic of all samples
        if n_scan is None:
            distances = []
            distances1 = []
            # 逐切片计算豪斯多夫距离
           # print("self.pred_lst:", self.pred_lst[0])
           #  print("len.pred_lst00:", len(self.pred_lst[0]))  # 53
           #  print("len.pred_lst11:", len(self.pred_lst[0][0])) # 1
            #print("self.pred_lst000:", self.pred_lst[0][0])

            # all_keys = set().union(*[item.keys() for item in self.pred_lst[i]])
            # selected_elements = {key: [item[key] for item in self.pred_lst[i] if key in item] for key in all_keys}
            # for key, values in selected_elements.items():
            #     print(f"Selected elements for '{key}': {values}")

            for i in range(len(self.pred_lst)):  # len(self.pred_lst)=5
                all_keys = set().union(*[item.keys() for item in self.pred_lst[i]])
                all_keys =list(all_keys)
                # key_list1 = list(self.pred_lst[i][0].keys())
                # # print("self：", (self.pred_lst[i][0].get(key_list1[0]))[0])
                # # if np.all(np.array((self.pred_lst[i][0].get(key_list1[0]))[0]) == 0):
                # #     print("++++++++++++++++++++++++++++++")
                for j in range(len(self.pred_lst[i])):
                    key_list = list(self.pred_lst[i][j].keys())
                    #print("key_list：", key_list)
                    # print("key_list[0]：", key_list[0])
                    #print("i:", i, "j:", j)
                    # if np.all(np.array((self.pred_lst[i][j].get(key_list[0]))[0]) == 0):
                    #     print("*******************************")
                    # if np.all(np.array((self.pred_lst[i][j].get(key_list[0]))[0])== 0):
                    #     print("*******************************")
                    if  key_list[0]==all_keys[0]:
                        # print("pred_lst[i][j].keys()：", self.pred_lst[i][j].keys())
                        # print("self：", (self.pred_lst[i][j].get(key_list[0])))
                        #print("self11：", (self.pred_lst[i][j].get(key_list[0]))[0])
                        array_pred = np.array((self.pred_lst[i][j].get(key_list[0]))[0])
                        array_qrey = np.array((self.qrey_lst[i][j].get(key_list[0]))[0])
                        points_pred = np.column_stack(np.where(array_pred != 0))
                        points_qrey = np.column_stack(np.where(array_qrey != 0))
                        distance = directed_hausdorff(points_pred, points_qrey)[0]
                        #distance = directed_hausdorff(np.array((self.pred_lst[i][j].get(key_list[0]))[0]), np.array((self.qrey_lst[i][j].get(key_list[0]))[0]))[0]
                        #print("distance：", distance)
                        hd95 = np.percentile(distance, 95)
                        distances.append(hd95)
                    if  key_list[0]==all_keys[1]:
                        # print("all_keys[1]：", all_keys[1])
                        array_pred = np.array((self.pred_lst[i][j].get(key_list[0]))[0])
                        array_qrey = np.array((self.qrey_lst[i][j].get(key_list[0]))[0])
                        points_pred = np.column_stack(np.where(array_pred != 0))
                        points_qrey = np.column_stack(np.where(array_qrey != 0))
                        distance1 = directed_hausdorff(points_pred, points_qrey)[0]
                        #distance1 = directed_hausdorff(np.array((self.pred_lst[i][j].get(key_list[0]))[0]), np.array((self.qrey_lst[i][j].get(key_list[0]))[0]))[0]
                        hd_95 = np.percentile(distance1, 95)
                        #print("hd_95：", hd_95)
                        distances1.append(hd_95)
            # 计算平均值
            average_distance = np.mean(distances)
            average_distance1 = np.mean(distances1)
            #print("average_distance:", average_distance, "average_distance1:", average_distance1)
            return average_distance, average_distance1

    def calculate_assd(self, labels=None, n_scan=None, give_raw=False):
        if n_scan is None:
            dis = []
            dis1 = []
            for i in range(len(self.pred_lst)):  # len(self.pred_lst)=5
                all_keys = set().union(*[item.keys() for item in self.pred_lst[i]])
                all_keys = list(all_keys)
                for j in range(len(self.pred_lst[i])):
                    key_list = list(self.pred_lst[i][j].keys())
                    if key_list[0] == all_keys[0]:

                        array_pred = np.array((self.pred_lst[i][j].get(key_list[0]))[0])
                        array_qrey = np.array((self.qrey_lst[i][j].get(key_list[0]))[0])
                        # points_pred = np.column_stack(np.where(array_pred != 0))
                        # points_qrey = np.column_stack(np.where(array_qrey != 0))
                        points_pred = np.unique(np.column_stack(np.where(array_pred != 0)), axis=0)
                        points_qrey = np.unique(np.column_stack(np.where(array_qrey != 0)), axis=0)
                        seg_tree = cKDTree(points_pred)
                        distances, _ = seg_tree.query(points_qrey)

                        gt_tree = cKDTree(points_qrey)
                        distances1, _ = gt_tree.query(points_pred)
                        non_zero_distances = distances[(distances > 0) & (np.isfinite(distances))]
                        non_zero_distances1 = distances1[(distances1 > 0) & (np.isfinite(distances1))]
                        # assd = np.mean(non_zero_distances)
                        # assd1 = np.mean(non_zero_distances1)
                        # print("assd:", assd)
                        # print("dis:", dis)
                        assd = (non_zero_distances.sum() + non_zero_distances1.sum()) / (
                                    len(non_zero_distances) + len(non_zero_distances1))
                        if ~np.isnan(assd):
                            dis.append(assd)
                    if key_list[0] == all_keys[1]:
                        # target_tree = cKDTree(np.array((self.qrey_lst[i][j].get(key_list[0]))[0]))
                        # distances, _ = target_tree.query(np.array((self.pred_lst[i][j].get(key_list[0]))[0]))
                        array_pred = np.array((self.pred_lst[i][j].get(key_list[0]))[0])
                        array_qrey = np.array((self.qrey_lst[i][j].get(key_list[0]))[0])
                        # points_pred = np.column_stack(np.where(array_pred != 0))
                        # points_qrey = np.column_stack(np.where(array_qrey != 0))
                        points_pred = np.unique(np.column_stack(np.where(array_pred != 0)), axis=0)
                        points_qrey = np.unique(np.column_stack(np.where(array_qrey != 0)), axis=0)
                        seg_tree = cKDTree(points_pred)
                        distances, _ = seg_tree.query(points_qrey)

                        gt_tree = cKDTree(points_qrey)
                        distances1, _ = gt_tree.query(points_pred)
                        non_zero_distances = distances[(distances > 0) & (np.isfinite(distances))]
                        non_zero_distances1 = distances1[(distances1 > 0) & (np.isfinite(distances1))]
                        # assd = np.mean(non_zero_distances)
                        # assd1 = np.mean(non_zero_distances1)
                        # print("assd:", assd)
                        # print("dis:", dis)
                        assd1 = (non_zero_distances.sum() + non_zero_distances1.sum()) / (
                                    len(non_zero_distances) + len(non_zero_distances1))
                        if ~np.isnan(assd1):
                            dis1.append(assd1)
            # 计算平均值
            # print("dis:", dis)
            avg_distance = np.mean(dis)
            average_distance1 = np.mean(dis1)
            # print("average_distance:", average_distance, "average_distance1:", average_distance1)
            return avg_distance, average_distance1


    def Perimeter_loss(self, pred_mask, true_mask, device = None):
        # 计算预测边界
        device = device if device is not None else torch.device("cpu")
        pred_boundary = F.conv2d(pred_mask.float().to(device), torch.ones(1, 2, 3, 3).to(device), padding=1) - pred_mask.float().to(device)
        pred_boundary = (pred_boundary != 0).float()  # 将边界设为1，其他部分设为0

        # 计算真实边界
        true_boundary = F.conv2d(true_mask.float().to(device), torch.ones(1, 1, 3, 3).to(device), padding=1) - true_mask.float().to(device)
        true_boundary = (true_boundary != 0).float()  # 将边界设为1，其他部分设为0

        # 计算周长之差损失
        diff = torch.abs(pred_boundary - true_boundary)
        boundary_loss = torch.sum(diff)
        return torch.mean(boundary_loss)



    def Hullarea_loss(self, y_pred_segmentation ,y_true_binary):
        # 计算原图标签和预测标签的凸包面积差异损失函数
        # 将标签转换为numpy数组
        y_true_segmentation = torch.cat([1 - y_true_binary, y_true_binary], dim=1)
        # 计算凸包面积差异
        convex_area_diff = 0
        for true_mask, pred_mask in zip(y_true_segmentation, y_pred_segmentation):
            true_mask_np = true_mask.cpu().detach().numpy().squeeze()
            pred_mask_np = pred_mask.cpu().detach().numpy().argmax(axis=0)

            true_hull = ConvexHull(np.transpose(np.nonzero(true_mask_np)))
            pred_hull = ConvexHull(np.transpose(np.nonzero(pred_mask_np)))

            true_area = true_hull.volume
            pred_area = pred_hull.volume
            convex_area_diff += np.abs(true_area - pred_area)

        # 归一化到[0, 1]范围
        max_area_diff = y_true_binary.size(2) * y_true_binary.size(3)
        normalized_loss = torch.tensor(convex_area_diff / max_area_diff, dtype=torch.float32)

        return normalized_loss

        #true_convex_area = calculate_convex_area(true_contours[0])


    def Eucdis_loss(self, y_pred ,y_true):
        # 归一化预测概率
        # 转换为NumPy数组以使用np.linalg.norm计算欧氏距离
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()
        # 计算欧氏距离
        euclidean_distance = np.linalg.norm(y_true_np-y_pred_np)
        #print("++++++++++++++++++++:", euclidean_distance)
        max_dist = torch.sqrt(torch.sum(y_true_np ** 2, dim=(1, 2)))
        normalized_dist = euclidean_distance / max_dist
        loss =torch.mean(normalized_dist)
        return loss

    def Holedet(self, y_pred ,y_true):
        # 形态学孔洞检测判别函数
        # 将标签转换为numpy数组
        y_true_np = y_true.cpu().detach().numpy()
        y_pred_np = (y_pred.cpu() > 0.5).detach().numpy().astype(np.uint8)
        # 执行形态学闭运算，填充孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        y_true_closed = cv2.morphologyEx(y_true_np.shape[1:], cv2.MORPH_CLOSE, kernel)
        y_pred_closed = cv2.morphologyEx(y_pred_np.shape[2:], cv2.MORPH_CLOSE, kernel)
        # 计算孔洞区域
        y_true_holes = y_true_closed - y_true_np.shape[1:]
        print("-------------:", y_true_holes)
        y_pred_holes = y_pred_closed - y_pred_np.shape[2:]
        print("??????????????:", y_pred_holes)
        # 判别条件：孔洞区域均为零或均不为零
        if np.sum(y_true_holes) == 0 and np.sum(y_pred_holes) == 0:
            return torch.tensor(0).to(y_true.device)  # 返回0
        elif np.sum(y_true_holes) != 0 and np.sum(y_pred_holes) != 0:
            return torch.tensor(0).to(y_true.device)  # 返回0
        else:
            return torch.tensor(1).to(y_true.device)  # 返回1

    def edge_histogram(self,img, y_pred ,y_true):
        # 边缘检测
        y_true_np = y_true.cpu().detach().numpy()
        #print("???????????????", y_true_np.shape)
        y_pred_np = y_pred.cpu().detach().numpy()
        img = img.cpu().detach().numpy()
        img = cv2.convertScaleAbs(img)
        img = cv2.cvtColor(np.transpose(img[0], (1, 2, 0)), cv2.COLOR_BGR2GRAY)
        #print("###############",img.shape )
        y_true_np = cv2.convertScaleAbs(y_true_np)
        y_pred_np = cv2.convertScaleAbs(y_pred_np)
        edges = cv2.Canny(img, 100, 200)
        # 计算边缘直方图
        hist1 = cv2.calcHist([edges], [0], y_true_np[0,:,:], [256], [0, 256])
        hist2 = cv2.calcHist([edges], [0], y_pred_np[0,0,:,:], [256], [0, 256])
        # 计算欧氏距离
        distance = np.linalg.norm(hist1 - hist2)
        # 计算相似度度量值
        similarity = (1 / (1 + distance)) *10000
        similarity= torch.tensor(similarity)
        return similarity

