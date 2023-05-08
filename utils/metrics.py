import numpy as np
import torch
import time, pdb


# 将输入转换为NumPy数组并展平
def converter(data):
    # 若输入数据 data 是张量
    if isinstance(data, torch.Tensor):
        # 将张量从 GPU 上转移到 CPU 并转换成 NumPy数组后展平
        data = data.cpu().data.numpy().flatten()
        # 否则直接展平
    return data.flatten()


# 计算预测标签和真实标签之间的直方图矩阵
#   l_pred：预测的标签
#   label_true：真实的标签
#   num_classes：分类任务中的类别总数
def fast_hist(label_pred, label_true, num_classes):
    # 1. 将label_true转换为整数类型并乘以num_classes （将每个类别的值映射到一个更大的范围）
    # 2. 将1.得到的值与label_pred逐元素相加获得新的组合值的数组
    # 3. 使用np.bincount函数对组合值进行计数，将minlength参数设置为num_classes的平方，以确保计数数组的长度始终相同。
    hist = np.bincount(num_classes * label_true.astype(int) + label_pred, minlength=num_classes ** 2)
    # 4. 将计数数组重新调整为shape为(num_classes, num_classes)的矩阵。
    hist = hist.reshape(num_classes, num_classes)
    return hist


# 计算分类任务的平均IoU
class Metric_mIoU:
    def __init__(self, class_num):
        # 类别数
        self.class_num = class_num
        # 创建一个形状为 (class_num, class_num) 的全零矩阵
        self.hist = np.zeros((self.class_num, self.class_num))

    # 更新
    def update(self, predict, target):
        # 1. 将预测标签 predict 和真实标签 target 转换为一维 NumPy 数组
        predict, target = converter(predict), converter(target)
        # 2. 使用 fast_hist 函数计算它们之间的直方图矩阵
        # 3. 将结果累加到 hist 矩阵中。
        self.hist += fast_hist(predict, target, self.class_num)

    # 重置
    def reset(self):
        # 将 hist 矩阵重置为全零矩阵
        self.hist = np.zeros((self.class_num, self.class_num))

    # 获取平均IoU
    def get_miou(self):
        # 1. 将 hist 矩阵的对角线元素除以矩阵行和列的和减去对角线元素，得到每个类别的 IoU 值
        miou = np.diag(self.hist) / (
                np.sum(self.hist, axis=1) + np.sum(self.hist, axis=0) -
                np.diag(self.hist))
        # 2. 计算所有类别 IoU 值的平均值，得到 mean-IoU
        miou = np.nanmean(miou)
        return miou

    # 计算平均准确率
    def get_acc(self):
        # 1. 通过将 hist 矩阵的对角线元素除以矩阵行的和，得到每个类别的准确率
        acc = np.diag(self.hist) / self.hist.sum(axis=1)
        # 2. 使用 np.nanmean 函数计算所有类别准确率的平均值。
        acc = np.nanmean(acc)
        return acc

    def get(self):
        return self.get_miou()


# 计算多标签分类任务的准确率
# 在多标签分类任务中，每个样本可以被分配一个或多个类别标签
class MultiLabelAcc:
    def __init__(self):
        self.cnt = 0
        self.correct = 0

    def reset(self):
        self.cnt = 0
        self.correct = 0

    def update(self, predict, target):
        # 1. 找到预测概率最大的类别索引
        predict = predict.argmax(1)
        # 2. 将预测标签 predict 和真实标签 target转换为一维 NumPy 数组
        predict, target = converter(predict), converter(target)
        # 3. 增加计数器 cnt 的值，使其等于样本数（即 predict 的长度）
        self.cnt += len(predict)
        # 4. 增加正确分类计数器 correct 的值，使其等于预测标签与真实标签相等的样本数。
        self.correct += np.sum(predict == target)

    def get_acc(self):
        # 将正确分类计数器 correct 除以计数器 cnt，得到准确率。
        return self.correct * 1.0 / self.cnt

    def get(self):
        return self.get_acc()


# 计算多类分类任务的 Top-k 准确率
# 在这种任务中，如果真实类别位于预测概率最高的前 k 个类别中，则认为预测是正确的。
class AccTopk:
    # background_classes: 负类
    def __init__(self, background_classes, k):
        self.background_classes = background_classes
        self.k = k
        self.cnt = 0
        self.top5_correct = 0

    def reset(self):
        self.cnt = 0
        self.top5_correct = 0

    def update(self, predict, target):
        # 1. 找到预测概率最大的类别索引
        predict = predict.argmax(1)
        # 2. 将预测标签 predict 和真实标签target转换为一维 NumPy 数组
        predict, target = converter(predict), converter(target)
        # 3. 增加计数器 cnt 的值，使其等于样本数（即 predict 的长度）
        self.cnt += len(predict)
        # 4. 计算背景类别的索引 background_idx
        background_idx = (target == self.background_classes)
        # 5. 使用逻辑非运算得到非背景类别的索引 not_background_idx
        not_background_idx = np.logical_not(background_idx)
        # 6. 增加 Top-k 正确分类计数器 top5_correct 的值，使其等于预测标签与真实标签之差小于 k 的样本数
        self.top5_correct += np.sum(np.absolute(predict[not_background_idx] - target[not_background_idx]) < self.k)

    def get(self):
        return self.top5_correct * 1.0 / self.cnt


class Mae:
    # dim_sel：要计算误差的维度
    def __init__(self, dim_sel, ignore=-1):
        self.dim_sel = dim_sel
        # ignore：要忽略的值（默认为 -1）
        self.ignore = ignore
        # all_res 用于存储每个样本的误差
        self.all_res = []

    def reset(self):
        self.all_res = []

    def update(self, predict, target):
        # 1. 从预测值 predict 和真实目标值 target中选择指定维度的数据
        predict = predict[..., self.dim_sel]
        target = target[..., self.dim_sel]

        cls_dim = predict.shape[1]
        grid = torch.arange(cls_dim, device=predict.device).view(1, cls_dim, 1)
        # 2. 对预测值沿着类别维度（cls_dim）进行 Softmax 操作
        predict = predict.softmax(1)
        # 3. 将结果与类别索引（grid）相乘，然后沿类别维度求和，最后对求和结果进行归一化
        predict = (predict * grid).sum(1) / (cls_dim - 1)
        # 4. 计算预测值和目标值的绝对误差，忽略目标值等于 ignore 的情况
        res = (predict - target).abs()[target != self.ignore]
        # 5. 将绝对误差转换为一维 NumPy 数组并添加到 all_res 列表中
        res = converter(res)
        if len(res) != 0:
            self.all_res.append(res)

    def get(self):
        if len(self.all_res) == 0:
            return 1
        return np.mean(np.concatenate(self.all_res))


def update_metrics(metric_dict, pair_data):
    for i in range(len(metric_dict['name'])):
        # 从 metric_dict 中获取相应的度量操作（metric_op）和数据源（data_src）。
        metric_op = metric_dict['op'][i]
        data_src = metric_dict['data_src'][i]
        # 使用 pair_data 中的预测值和目标值来更新度量操作
        # 具体地，从 pair_data 中获取数据源对应的预测值和目标值，并传递给 metric_op.update() 方法
        metric_op.update(pair_data[data_src[0]], pair_data[data_src[1]])

# 重置评估标准
def reset_metrics(metric_dict):
    # 遍历 metric_dict 中的所有度量操作，然后调用每个度量操作的 reset() 方法以重置它们
    for op in metric_dict['op']:
        op.reset()
        # 在每个训练或验证周期开始时重置度量指标，确保每个周期的度量指标是独立计算的


if __name__ == '__main__':

    a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
    b = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4, 0])
    me = AccTopk(0, 5)
    me.update(b, a)
    print(me.get())
