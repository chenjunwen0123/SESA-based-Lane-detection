import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np


# 难例挖掘(OHEM, Online Hard Example Mining)的交叉熵损失函数。
class OhemCELoss(nn.Module):
    # thresh：阈值，筛选难例
    # n_min：最小保留样本数量
    # ignore_lb：忽略标签值，表示某些像素点不参与损失计算，默认值为255
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        # 初始化交叉熵损失函数
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    # logits为预测值，labels为真实标签
    def forward(self, logits, labels):
        # 获取logits的大小：N, C, H, W
        N, C, H, W = logits.size()
        # 计算交叉熵损失，返回形状为(-1)的向量
        loss = self.criteria(logits, labels).view(-1)
        # 对损失进行降序排序
        loss, _ = torch.sort(loss, descending=True)
        # 判断排在self.n_min位置的损失是否大于阈值self.thresh
        if loss[self.n_min] > self.thresh:
            # 若是，保留所有大于阈值的损失
            loss = loss[loss > self.thresh]
        else:
            # 否则，保留前self.n_min个损失
            loss = loss[:self.n_min]
            # 返回保留损失的均值
        return torch.mean(loss)


# 计算在考虑目标标签及其邻域信息的情况下的损失值
# 有助于在训练过程中更好地捕捉目标对象的局部信息
def soft_nll(pred, target, ignore_index=-1):
    # 预测类别数量C
    C = pred.shape[1]
    # 哪些像素点应被忽略
    invalid_target_index = target == ignore_index

    # 复制目标标签，将忽略的像素点设置为C
    ttarget = target.clone()
    ttarget[invalid_target_index] = C

    # 创建目标标签的左邻域（target_l）和右邻域（target_r）
    target_l = target - 1
    target_r = target + 1

    # 对左右邻域进行边界条件处理，避免越界
    invalid_part_l = target_l == -1
    invalid_part_r = target_r == C

    invalid_target_l_index = torch.logical_or(invalid_target_index, invalid_part_l)
    target_l[invalid_target_l_index] = C

    invalid_target_r_index = torch.logical_or(invalid_target_index, invalid_part_r)
    target_r[invalid_target_r_index] = C

    # 创建左右补充部分（supp_part_l和supp_part_r），用于生成目标融合矩阵
    supp_part_l = target.clone()
    supp_part_r = target.clone()
    supp_part_l[target != 0] = C
    supp_part_r[target != C - 1] = C

    # 将目标标签、左邻域、右邻域、左补充部分和右补充部分转换为one-Hot编码
    target_onehot = torch.nn.functional.one_hot(ttarget, num_classes=C + 1)
    target_onehot = target_onehot[..., :-1].permute(0, 3, 1, 2)

    target_l_onehot = torch.nn.functional.one_hot(target_l, num_classes=C + 1)
    target_l_onehot = target_l_onehot[..., :-1].permute(0, 3, 1, 2)

    target_r_onehot = torch.nn.functional.one_hot(target_r, num_classes=C + 1)
    target_r_onehot = target_r_onehot[..., :-1].permute(0, 3, 1, 2)

    supp_part_l_onehot = torch.nn.functional.one_hot(supp_part_l, num_classes=C + 1)
    supp_part_l_onehot = supp_part_l_onehot[..., :-1].permute(0, 3, 1, 2)

    supp_part_r_onehot = torch.nn.functional.one_hot(supp_part_r, num_classes=C + 1)
    supp_part_r_onehot = supp_part_r_onehot[..., :-1].permute(0, 3, 1, 2)
    # 创建目标融合矩阵，将原始目标、左右邻域以及左右补充部分融合。给定的权重分别为0.9、0.05、0.05、0.05和0.05
    target_fusion = 0.9 * target_onehot + 0.05 * target_l_onehot + 0.05 * target_r_onehot + 0.05 * supp_part_l_onehot + 0.05 * supp_part_r_onehot
    # 计算损失值：乘以预测值（pred），求和，然后除以非忽略像素点的数量
    return -(target_fusion * pred).sum() / (target != ignore_index).sum()


# 计算分类损失（多了一个权重项，即 预测正样本概率 越高，则权重越低，控制对高难度的样本的适配性）
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, soft_loss=True, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self.soft_loss = soft_loss
        if not self.soft_loss:
            self.nll = nn.NLLLoss(ignore_index=ignore_lb)

    def forward(self, logits, labels):  # softmax(x) + log(x) + nn.NLLLoss
        # 归一化
        # logits: (batch,num_grid_row, num_row, num_lane)
        scores = F.softmax(logits, dim=1)

        # 当前样本的权重项
        factor = torch.pow(1. - scores, self.gamma)

        # 交叉熵，shape不变， shape = (batch, num_grid_row, num_row, num_lane)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score

        if self.soft_loss:
            loss = soft_nll(log_score, labels, ignore_index=self.ignore_lb)
        else:
            # 标签：（18,4) (num_row, num_lane) ，labels[x, i] 表示车道线i在行锚x上的列坐标
            # 比如说有201个（18，4），则只会看这201个cell中的对应位置上的预测值 比如说 117，则该位置的201个位置中 只看这117的位置上的预测值
            # [[
            #   [ 200,107,117,141],
            #   ...
            #   [ .. , .., .., ..]
            # ]]
            loss = self.nll(log_score, labels)

        # import pdb; pdb.set_trace()
        return loss


# 相邻需相似，结构损失
class ParsingRelationLoss(nn.Module):
    def __init__(self):
        super(ParsingRelationLoss, self).__init__()

    def forward(self, logits):
        # logits：预测值的结果
        n, c, h, w = logits.shape
        loss_all = []
        for i in range(0, h - 1):
            # 限制条件1：相邻行的正样本预测值的结果的差值（概率的差值）
            loss_all.append(logits[:, :, i, :] - logits[:, :, i + 1, :])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        # 平滑L1损失是一种在L1损失和L2损失之间折衷的方法，旨在平衡二者之间的优点。
        # 与L1损失相比，平滑L1损失对异常值更加稳健；与L2损失相比，它对于异常值的惩罚更小，因此可以更好地适应一些数据的分布情况。
        # 在上述代码中，torch.nn.functional.smooth_l1_loss函数用于计算loss张量和一个全零张量之间的平滑L1损失。
        # 该函数返回的是平滑L1损失的值，该值可用于计算模型的损失函数，并用于模型的训练和优化过程中。
        return torch.nn.functional.smooth_l1_loss(loss, torch.zeros_like(loss))


# 计算均值损失：度量预测值与实际标签之间的差异
class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, logits, label):
        n, c, h, w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
        # 计算logits的softmax，然后将其逐元素乘以grid，接着沿维度1求和 --> logits(n,h,w)
        logits = (logits.softmax(1) * grid).sum(1)
        # 使用nn.SmoothL1Loss计算logits和label之间的损失
        # 仅选择那些对应于label中非-1元素的损失值
        loss = self.l1(logits, label.float())[label != -1]
        # 返回损失值的均值
        return loss.mean()


# 计算可变损失:度量预测值与实际标签之间的差异
class VarLoss(nn.Module):
    def __init__(self, power=2):
        super(VarLoss, self).__init__()
        self.power = power

    def forward(self, logits, label):
        n, c, h, w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
        # 对logits沿着维度1计算softmax，将预测值转换为概率
        logits = logits.softmax(1)
        # 计算logits与grid的加权和 ---> mean(n, 1, h, w)
        mean = (logits * grid).sum(1).view(n, 1, h, w)
        # 计算mean与grid之间的绝对差值，将其乘以给定的幂次（self.power），然后再逐元素乘以logits ---> var(n,1,h,w)
        var = (mean - grid).abs().pow(self.power) * logits
        # 对var在维度1上求和，然后仅选择那些对应于label中非-1元素且label与mean之间的绝对差值小于1的损失值 ---> loss(n,c,h,w)
        loss = var.sum(1)[(label != -1) & ((label - mean.squeeze()).abs() < 1)]
        # 返回损失值的均值
        return loss.mean()


# Earth Mover's Distance (EMD)：衡量两个概率分布之间的差异
# 衡量预测概率分布与实际标签分布之间的差异，从而优化模型在图像分割和回归任务上的性能
class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()

    def forward(self, logits, label):
        # 获取logits的形状信息
        n, c, h, w = logits.shape
        # 在给定设备上创建一个形状为(1, c, 1, 1)的等差数列grid
        grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
        # 对logits沿着维度1计算softmax，将预测值转换为概率
        logits = logits.softmax(1)
        # 计算label与grid之间的差的平方，并将其逐元素乘以logits ---> var(n, c, h, w)
        var = (label.reshape(n, 1, h, w) - grid) * (label.reshape(n, 1, h, w) - grid) * logits
        # 对var在维度1上求和，然后仅选择那些对应于label中非-1元素的损失值 ---> loss(n,c,h,w)
        loss = var.sum(1)[label != -1]
        # 返回损失值的均值
        return loss.mean()


#
class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()

    def forward(self, x):
        n, dim, num_rows, num_cols = x.shape
        # 每个位置的预测概率
        x = torch.nn.functional.softmax(x[:, :dim - 1, :, :], dim=1)

        # 求期望
        embedding = torch.Tensor(np.arange(dim - 1)).float().to(x.device).view(1, -1, 1, 1)
        # 实际预测的位置
        pos = torch.sum(x * embedding, dim=1)

        diff_list1 = []
        for i in range(0, num_rows // 2):
            # 限制2：同一条车道线上相邻位置，不会发生剧烈抖动（实际位置的差值，区别限制1）
            diff_list1.append(pos[:, i, :] - pos[:, i + 1, :])

        loss = 0
        for i in range(len(diff_list1) - 1):
            loss += self.l1(diff_list1[i], diff_list1[i + 1])
        loss /= len(diff_list1) - 1
        return loss


# 交叉熵损失
# pred：模型预测值，通常为一个logits张量
# target: 实际标签，通常是一个独热编码(one-hot)的张量
# reduction：损失值的聚合方式，可以是'elementwise_mean'（默认值，计算元素平均值），'sum'（计算元素和）或None（保留原始损失张量）
def cross_entropy(pred, target, reduction='elementwise_mean'):
    # 1. 将预测值转换为概率对数：对预测值pred应用torch.nn.functional.log_softmax函数，沿着维度1计算log-softmax
    # 2. 计算损失张量：计算负的实际标签target与log-softmax结果的逐元素乘积，结果的每个元素表示对应预测与实际标签之间的损失
    res = -target * torch.nn.functional.log_softmax(pred, dim=1)
    # 3. 根据reduction参数的值，选择聚合损失值的方式
    if reduction == 'elementwise_mean':
        # 如果是'elementwise_mean'，计算损失张量在维度1上的和，然后取平均值
        return torch.mean(torch.sum(res, dim=1))

    elif reduction == 'sum':
        # 如果是'sum'，计算损失张量在维度1上的和，然后求和
        return torch.sum(torch.sum(res, dim=1))
    else:
        # 如果是None，返回原始损失张量
        return res


# L1损失
class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction='none')

    def forward(self, logits, label):
        n, c, h, w = logits.shape
        assert c == 1
        logits = logits.sigmoid()
        loss = self.l1(logits[:, 0], label)[label != -1]
        # print(logits[0], label[0])
        # import pdb; pdb.set_trace()
        return loss.mean()


# 计算二值交叉熵损失，优化模型的分割性能
class TokenSegLoss(nn.Module):
    def __init__(self):
        super(TokenSegLoss, self).__init__()
        # 二值交叉熵损失
        self.criterion = nn.BCELoss()
        # 最大池化
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, logits, labels):
        # 使用双线性插值（bilinear）对logits进行上采样，将其大小调整为（200, 400）
        # 对调整大小后的logits应用sigmoid函数
        # 对实际标签进行最大池化，池化大小为4x4，只保留第一个通道（labels[:, 0:1, :, :]）
        # 将池化后的结果与0比较，得到一个与上采样后的logits形状相同的二值张量
        # 计算上采样后的logits和二值张量之间的二值交叉熵损失（BCELoss）
        return self.criterion(F.interpolate(logits, size=(200, 400), mode='bilinear').sigmoid(),
                              (self.max_pool(labels[:, 0:1, :, :]) != 0).float())


def test_cross_entropy():
    pred = torch.rand(10, 200, 33, 66)
    target = torch.randint(200, (10, 33, 66))
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=200).permute(0, 3, 1, 2)
    print(torch.nn.functional.cross_entropy(pred, target))
    print(cross_entropy(pred, target_one_hot))
    print(soft_nll(torch.nn.functional.log_softmax(pred, dim=1), torch.randint(-1, 200, (10, 33, 66))))

    # assert torch.nn.functional.cross_entropy(pred,target) == cross_entropy(pred,target_one_hot)
    print('OK')


if __name__ == "__main__":
    test_cross_entropy()
