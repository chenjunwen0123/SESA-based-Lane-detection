import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class OhemCELoss(nn.Module):
    def __init__(self, thresh, n_min, ignore_lb=255, *args, **kwargs):
        super(OhemCELoss, self).__init__()
        self.thresh = -torch.log(torch.tensor(thresh, dtype=torch.float)).cuda()
        self.n_min = n_min
        self.ignore_lb = ignore_lb
        self.criteria = nn.CrossEntropyLoss(ignore_index=ignore_lb, reduction='none')

    def forward(self, logits, labels):
        N, C, H, W = logits.size()
        loss = self.criteria(logits, labels).view(-1)
        loss, _ = torch.sort(loss, descending=True)
        if loss[self.n_min] > self.thresh:
            loss = loss[loss>self.thresh]
        else:
            loss = loss[:self.n_min]
        return torch.mean(loss)

def soft_nll(pred, target, ignore_index = -1):
    C = pred.shape[1]
    invalid_target_index = target==ignore_index

    ttarget = target.clone()
    ttarget[invalid_target_index] = C

    target_l = target - 1
    target_r = target + 1

    invalid_part_l = target_l == -1
    invalid_part_r = target_r == C

    invalid_target_l_index = torch.logical_or(invalid_target_index, invalid_part_l)
    target_l[invalid_target_l_index] = C

    invalid_target_r_index = torch.logical_or(invalid_target_index, invalid_part_r)
    target_r[invalid_target_r_index] = C

    supp_part_l = target.clone()
    supp_part_r = target.clone()
    supp_part_l[target!=0] = C
    supp_part_r[target!=C-1] = C

    target_onehot = torch.nn.functional.one_hot(ttarget, num_classes=C+1)
    target_onehot = target_onehot[...,:-1].permute(0,3,1,2)

    target_l_onehot = torch.nn.functional.one_hot(target_l, num_classes=C+1)
    target_l_onehot = target_l_onehot[...,:-1].permute(0,3,1,2)

    target_r_onehot = torch.nn.functional.one_hot(target_r, num_classes=C+1)
    target_r_onehot = target_r_onehot[...,:-1].permute(0,3,1,2)

    supp_part_l_onehot = torch.nn.functional.one_hot(supp_part_l, num_classes=C+1)
    supp_part_l_onehot = supp_part_l_onehot[...,:-1].permute(0,3,1,2)

    supp_part_r_onehot = torch.nn.functional.one_hot(supp_part_r, num_classes=C+1)
    supp_part_r_onehot = supp_part_r_onehot[...,:-1].permute(0,3,1,2)

    target_fusion = 0.9 * target_onehot + 0.05 * target_l_onehot + 0.05 * target_r_onehot + 0.05 * supp_part_l_onehot + 0.05 * supp_part_r_onehot
    # import pdb; pdb.set_trace()
    return -(target_fusion * pred).sum() / (target!=ignore_index).sum()

# 计算分类损失（多了一个权重项，即 预测正样本概率 越高，则权重越低，控制对高难度的样本的适配性）
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma, ignore_lb=255, soft_loss = True, *args, **kwargs):
        super(SoftmaxFocalLoss, self).__init__()
        self.gamma = gamma
        self.ignore_lb = ignore_lb
        self.soft_loss = soft_loss
        if not self.soft_loss:
            self.nll = nn.NLLLoss(ignore_index=ignore_lb)


    def forward(self, logits, labels):   # softmax(x) + log(x) + nn.NLLLoss
        # 归一化
        # logits: (batch,num_grid_row, num_row, num_lane)
        scores = F.softmax(logits, dim=1)

        # 当前样本的权重项
        factor = torch.pow(1.-scores, self.gamma)

        # 交叉熵，shape不变， shape = (batch, num_grid_row, num_row, num_lane)
        log_score = F.log_softmax(logits, dim=1)
        log_score = factor * log_score


        if self.soft_loss:
            loss = soft_nll(log_score, labels, ignore_index = self.ignore_lb)
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
    def forward(self,logits):
        # logits：预测值的结果
        n,c,h,w = logits.shape
        loss_all = []
        for i in range(0, h-1):
            # 限制条件1：相邻行的正样本预测值的结果的差值（概率的差值）
            loss_all.append(logits[:,:,i,:] - logits[:,:,i+1,:])
        # loss0 : n,c,w
        loss = torch.cat(loss_all)
        # 平滑L1损失是一种在L1损失和L2损失之间折衷的方法，旨在平衡二者之间的优点。
        # 与L1损失相比，平滑L1损失对异常值更加稳健；与L2损失相比，它对于异常值的惩罚更小，因此可以更好地适应一些数据的分布情况。
        # 在上述代码中，torch.nn.functional.smooth_l1_loss函数用于计算loss张量和一个全零张量之间的平滑L1损失。
        # 该函数返回的是平滑L1损失的值，该值可用于计算模型的损失函数，并用于模型的训练和优化过程中。
        return torch.nn.functional.smooth_l1_loss(loss,torch.zeros_like(loss))

class MeanLoss(nn.Module):
    def __init__(self):
        super(MeanLoss, self).__init__()
        self.l1 = nn.SmoothL1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = (logits.softmax(1) * grid).sum(1)
        loss = self.l1(logits, label.float())[label != -1]
        return loss.mean()

class VarLoss(nn.Module):
    def __init__(self, power = 2):
        super(VarLoss, self).__init__()
        self.power = power
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1,c,1,1)
        logits = logits.softmax(1)
        mean = (logits * grid).sum(1).view(n,1,h,w)
        # n,1,h,w
        var = (mean - grid).abs().pow(self.power) * logits
        # var = ((mean - grid).abs() - 4) * logits
        # n,c,h,w
        loss = var.sum(1)[(label != -1 ) & ( (label - mean.squeeze()).abs() < 1) ]
        return loss.mean()

class EMDLoss(nn.Module):
    def __init__(self):
        super(EMDLoss, self).__init__()
    def forward(self, logits, label):
        n, c, h, w = logits.shape
        grid = torch.arange(c, device=logits.device).view(1, c, 1, 1)
        logits = logits.softmax(1)
        # n,1,h,w
        var = (label.reshape(n, 1, h, w) - grid) * (label.reshape(n, 1, h, w) - grid) * logits
        # n,c,h,w
        loss = var.sum(1)[label != -1]
        return loss.mean()

#
class ParsingRelationDis(nn.Module):
    def __init__(self):
        super(ParsingRelationDis, self).__init__()
        self.l1 = torch.nn.L1Loss()
        # self.l1 = torch.nn.MSELoss()
    def forward(self, x):
        n,dim,num_rows,num_cols = x.shape
        # 每个位置的预测概率
        x = torch.nn.functional.softmax(x[:,:dim-1,:,:],dim=1)

        # 求期望
        embedding = torch.Tensor(np.arange(dim-1)).float().to(x.device).view(1,-1,1,1)
        # 实际预测的位置
        pos = torch.sum(x*embedding,dim = 1)

        diff_list1 = []
        for i in range(0,num_rows // 2):
            # 限制2：同一条车道线上相邻位置，不会发生剧烈抖动（实际位置的差值，区别限制1）
            diff_list1.append(pos[:,i,:] - pos[:,i+1,:])

        loss = 0
        for i in range(len(diff_list1)-1):
            loss += self.l1(diff_list1[i],diff_list1[i+1])
        loss /= len(diff_list1) - 1
        return loss


def cross_entropy(pred, target, reduction='elementwise_mean'):
    res  = -target * torch.nn.functional.log_softmax(pred, dim=1)
    if reduction == 'elementwise_mean':
        return torch.mean(torch.sum(res, dim=1))
    elif reduction == 'sum':
        return torch.sum(torch.sum(res, dim=1))
    else:
        return res

class RegLoss(nn.Module):
    def __init__(self):
        super(RegLoss, self).__init__()
        self.l1 = nn.L1Loss(reduction = 'none')
    def forward(self, logits, label):
        n,c,h,w = logits.shape
        assert c == 1
        logits = logits.sigmoid()
        loss = self.l1(logits[:,0], label)[label != -1]
        # print(logits[0], label[0])
        # import pdb; pdb.set_trace()
        return loss.mean()

class TokenSegLoss(nn.Module):
    def __init__(self):
        super(TokenSegLoss, self).__init__()
        self.criterion = nn.BCELoss()
        self.max_pool = nn.MaxPool2d(4)

    def forward(self, logits, labels):
        return self.criterion(F.interpolate(logits, size=(200, 400), mode='bilinear').sigmoid(), (self.max_pool(labels[:, 0:1, :, :]) != 0).float())

def test_cross_entropy():
    pred = torch.rand(10,200,33,66)
    target = torch.randint(200,(10,33,66))
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=200).permute(0,3,1,2)
    print(torch.nn.functional.cross_entropy(pred,target))
    print(cross_entropy(pred,target_one_hot))
    print(soft_nll(torch.nn.functional.log_softmax(pred, dim=1),torch.randint(-1,200,(10,33,66))))

    # assert torch.nn.functional.cross_entropy(pred,target) == cross_entropy(pred,target_one_hot)
    print('OK')



if __name__ == "__main__":
    test_cross_entropy()