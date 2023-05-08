from utils.loss import SoftmaxFocalLoss, ParsingRelationLoss, ParsingRelationDis, MeanLoss, TokenSegLoss, VarLoss, \
    EMDLoss, RegLoss
from utils.metrics import MultiLabelAcc, AccTopk, Metric_mIoU, Mae
from utils.dist_utils import DistSummaryWriter

import torch
import itertools
import math


# 获取优化器
# 根据给定的网络（net）和配置（cfg）返回一个优化器
# 根据配置中的 use_aux 和 optimizer 参数，函数会创建并返回一个适当的优化器实例
# 支持两种优化器：Adam 和 SGD
def get_optimizer(net, cfg):
    # 从网络中筛选需要训练的参数
    training_params = filter(lambda p: p.requires_grad, net.parameters())
    se512_params = filter(lambda p: p.requires_grad, net.se512.parameters())
    if (cfg.use_aux == True):
        se256_params = filter(lambda p: p.requires_grad, net.se256.parameters())
        se128_params = filter(lambda p: p.requires_grad, net.se128.parameters())
        # 将所有需要训练的参数连接在一起
        training_params = itertools.chain(training_params, se512_params, se256_params, se128_params)
    else:
        training_params = itertools.chain(training_params, se512_params)

    # 根据配置中的优化器类型创建相应的优化器实例
    if cfg.optimizer == 'Adam':
        optimizer = torch.optim.Adam(training_params, lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == 'SGD':
        optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
                                    weight_decay=cfg.weight_decay)
        # 开启Nesterov加速： optimizer = torch.optim.SGD(training_params, lr=cfg.learning_rate, momentum=cfg.momentum,
        #                                     weight_decay=cfg.weight_decay, Nesterov=True)
    else:
        # 如果配置中的优化器类型未实现，则抛出异常
        raise NotImplementedError
    return optimizer


# 获取学习率调度器
# 该函数根据给定的优化器（optimizer）、配置（cfg）和每个训练周期的迭代次数（iters_per_epoch）返回一个学习率调度器
# 根据配置中的 scheduler 参数，函数会创建并返回一个适当的学习率调度器实例
# 支持两种学习率调度器：multi（MultiStepLR）和 cos（CosineAnnealingLR）。
def get_scheduler(optimizer, cfg, iters_per_epoch):
    # 根据配置中的学习率调度器类型创建相应的学习率调度器实例
    if cfg.scheduler == 'multi':
        scheduler = MultiStepLR(optimizer, cfg.steps, cfg.gamma, iters_per_epoch, cfg.warmup,
                                iters_per_epoch if cfg.warmup_iters is None else cfg.warmup_iters)
    elif cfg.scheduler == 'cos':
        scheduler = CosineAnnealingLR(optimizer, cfg.epoch * iters_per_epoch, eta_min=0, warmup=cfg.warmup,
                                      warmup_iters=cfg.warmup_iters)
    else:
        # 如果配置中的学习率调度器类型未实现，则抛出异常
        raise NotImplementedError
    return scheduler


# 初始化损失函数映射关系
def get_loss_dict(cfg):
    # 根据配置中的数据集类型创建适当的损失函数字典
    if cfg.dataset == 'CurveLanes':
        loss_dict = {
            # 定义损失函数的名称
            'name': ['cls_loss', 'relation_loss', 'relation_dis', 'cls_loss_col', 'cls_ext', 'cls_ext_col',
                     'mean_loss_row', 'mean_loss_col', 'var_loss_row', 'var_loss_col', 'lane_token_seg_loss_row',
                     'lane_token_seg_loss_col'],
            # 定义损失函数的实例
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(),
                   SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(),
                   MeanLoss(), MeanLoss(), VarLoss(cfg.var_loss_power), VarLoss(cfg.var_loss_power), TokenSegLoss(),
                   TokenSegLoss()],
            # 定义损失函数的权重
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w, 0.01, 0.01,
                       1.0, 1.0],
            # 定义损失函数的数据来源
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'),
                         ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label'),
                         ('cls_out', 'cls_label'), ('cls_out_col', 'cls_label_col'), ('cls_out', 'cls_label'),
                         ('cls_out_col', 'cls_label_col'), ('seg_out_row', 'seg_label'), ('seg_out_col', 'seg_label')
                         ],
        }
    elif cfg.dataset in ['Tusimple', 'CULane']:
        loss_dict = {
            'name': ['cls_loss', 'relation_loss', 'relation_dis', 'cls_loss_col', 'cls_ext', 'cls_ext_col',
                     'mean_loss_row', 'mean_loss_col'],
            'op': [SoftmaxFocalLoss(2, ignore_lb=-1), ParsingRelationLoss(), ParsingRelationDis(),
                   SoftmaxFocalLoss(2, ignore_lb=-1), torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss(),
                   MeanLoss(), MeanLoss(), ],
            'weight': [1.0, cfg.sim_loss_w, cfg.shp_loss_w, 1.0, 1.0, 1.0, cfg.mean_loss_w, cfg.mean_loss_w, ],
            'data_src': [('cls_out', 'cls_label'), ('cls_out',), ('cls_out',), ('cls_out_col', 'cls_label_col'),
                         ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label'),
                         ('cls_out', 'cls_label'), ('cls_out_col', 'cls_label_col'),
                         ],
        }
    else:
        raise NotImplementedError

    # 若开启分割分支，则分配对应的损失函数
    if cfg.use_aux:
        loss_dict['name'].append('seg_loss')
        loss_dict['op'].append(torch.nn.CrossEntropyLoss(weight=torch.tensor([0.6, 1., 1., 1., 1.])).cuda())
        loss_dict['weight'].append(1.0)
        loss_dict['data_src'].append(('seg_out', 'seg_label'))

    assert len(loss_dict['name']) == len(loss_dict['op']) == len(loss_dict['data_src']) == len(loss_dict['weight'])
    return loss_dict


# 初始化评估标准映射
def get_metric_dict(cfg):
    metric_dict = {
        # 定义评估指标的名称
        'name': ['top1', 'top2', 'top3', 'ext_row', 'ext_col'],
        # 定义评估指标的操作：分别使用三个不同的Top-k准确率计算（top-1，top-2，top-3），以及两个 MultiLabelAcc 类实例。
        'op': [AccTopk(-1, 1), AccTopk(-1, 2), AccTopk(-1, 3), MultiLabelAcc(), MultiLabelAcc()],
        # 定义评估指标的数据来源：指定数据来源与评估指标的操作相对应
        'data_src': [('cls_out', 'cls_label'), ('cls_out', 'cls_label'), ('cls_out', 'cls_label'),
                     ('cls_out_ext', 'cls_out_ext_label'), ('cls_out_col_ext', 'cls_out_col_ext_label')]
    }
    metric_dict['name'].extend(['col_top1', 'col_top2', 'col_top3'])
    metric_dict['op'].extend([AccTopk(-1, 1), AccTopk(-1, 2), AccTopk(-1, 3), ])
    metric_dict['data_src'].extend(
        [('cls_out_col', 'cls_label_col'), ('cls_out_col', 'cls_label_col'), ('cls_out_col', 'cls_label_col'), ])

    if cfg.use_aux:
        metric_dict['name'].append('iou')
        metric_dict['op'].append(Metric_mIoU(5))
        metric_dict['data_src'].append(('seg_out', 'seg_label'))

    assert len(metric_dict['name']) == len(metric_dict['op']) == len(metric_dict['data_src'])
    return metric_dict


# 在指定Step时动态调整优化器的学习率
# 在指定的阶梯式调整步骤中，学习率将乘以一个衰减因子
# 此外，还可以选择在训练开始时进行线性预热，以避免学习率初始值过大导致的不稳定
class MultiStepLR:
    def __init__(self, optimizer, steps, gamma=0.1, iters_per_epoch=None, warmup=None, warmup_iters=None):
        self.warmup = warmup  # 预热策略，这里使用线性预热
        self.warmup_iters = warmup_iters  # 预热迭代次数
        self.optimizer = optimizer  # 优化器
        self.steps = steps  # 阶梯式调整的步骤列表
        self.steps.sort()  # 将步骤排序
        self.gamma = gamma  # 衰减因子
        self.iters_per_epoch = iters_per_epoch  # 每个 epoch 的迭代次数
        self.iters = 0  # 当前迭代次数
        self.base_lr = [group['lr'] for group in optimizer.param_groups]  # 初始学习率列表

    def step(self, external_iter=None):
        self.iters += 1  # 更新迭代次数
        if external_iter is not None:
            self.iters = external_iter  # 如果提供外部迭代次数，则使用外部迭代次数

        # 检查是否处于预热阶段
        # 如果是，则根据线性预热策略更新学习率
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters  # 计算预热的线性增长率
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate  # 更新学习率
            return

        # 在预热阶段之后，将根据多步调整策略更新学习率
        # 判断是否到达一个新的epoch，即当前迭代次数是否能整除每个epoch的迭代次数
        if self.iters % self.iters_per_epoch == 0:
            epoch = int(self.iters / self.iters_per_epoch)  # 计算当前 epoch
            power = -1
            # 计算当前 epoch 对应的衰减次数（power），衰减次数即为当前 epoch 大于等于的最小步骤的索引。
            for i, st in enumerate(self.steps):
                if epoch < st:
                    power = i
                    break
            if power == -1:
                power = len(self.steps)
            # 使用衰减次数更新学习率。新的学习率等于初始学习率乘以衰减因子的衰减次数次方
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * (self.gamma ** power)  # 更新学习率


# 余弦退火学习率调度器
class CosineAnnealingLR:
    # 初始化调度器，需要传入优化器、最大退火周期、最小学习率以及预热策略和预热迭代次数。
    def __init__(self, optimizer, T_max, eta_min=0, warmup=None, warmup_iters=None):
        self.warmup = warmup  # 预热策略，这里使用线性预热
        self.warmup_iters = warmup_iters  # 预热迭代次数
        self.optimizer = optimizer  # 优化器
        self.T_max = T_max  # 余弦退火周期
        self.eta_min = eta_min  # 最小学习率

        self.iters = 0  # 当前迭代次数
        self.base_lr = [group['lr'] for group in optimizer.param_groups]  # 初始学习率列表

    # 更新学习率
    def step(self, external_iter=None):
        self.iters += 1  # 更新迭代次数
        if external_iter is not None:
            self.iters = external_iter  # 如果提供外部迭代次数，则使用外部迭代次数

        # 检查是否处于预热阶段
        # 如果是，则根据线性预热策略更新学习率
        if self.warmup == 'linear' and self.iters < self.warmup_iters:
            rate = self.iters / self.warmup_iters  # 计算预热的线性增长率
            for group, lr in zip(self.optimizer.param_groups, self.base_lr):
                group['lr'] = lr * rate  # 更新学习率
            return

        # 如果不是预热阶段，则根据余弦退火策略更新学习率
        for group, lr in zip(self.optimizer.param_groups, self.base_lr):
            group['lr'] = self.eta_min + (lr - self.eta_min) * (1 + math.cos(math.pi * self.iters / self.T_max)) / 2
