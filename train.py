import torch, os, datetime

from utils.dist_utils import dist_print, dist_tqdm, synchronize
from utils.factory import get_metric_dict, get_loss_dict, get_optimizer, get_scheduler
from utils.metrics import update_metrics, reset_metrics

from utils.common import calc_loss, get_model, get_train_loader, inference, merge_config, save_model, cp_projects
from utils.common import get_work_dir, get_logger

import time
from evaluation.eval_wrapper import eval_lane


# net:  训练好的神经网络模型
# data_loader：数据加载器
# loss_dict：各部分损失函数
# optimizer：优化器
# scheduler：学习率调节器
# logger：日志记录器
# epoch：当前训练轮数
# metric_dict：度量字典
# dataset：数据集
def train(net, data_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, dataset):
    net.train()
    # 打印训练进度的进度条
    progress_bar = dist_tqdm(train_loader)
    # 遍历每一个batch
    for b_idx, data_label in enumerate(progress_bar):
        # 当前训练步数（进度）
        global_step = epoch * len(data_loader) + b_idx

        # 输入数据后，模型输出的结果（走完全连接层）
        results = inference(net, data_label, dataset)

        # 计算损失函数值
        loss = calc_loss(loss_dict, results, logger, global_step, epoch)

        # 梯度清零
        optimizer.zero_grad()
        # 反向传播，计算梯度
        loss.backward()
        # 更新模型参数
        optimizer.step()
        # 调整学习率
        scheduler.step(global_step)

        # 每隔20个训练步骤执行1次
        if global_step % 20 == 0:
            # 重置指标字典中的指标值
            reset_metrics(metric_dict)

            # 更新指标字典中的指标值
            update_metrics(metric_dict, results)
            for me_name, me_op in zip(metric_dict['name'], metric_dict['op']):
                # 记录当前的指标值
                logger.add_scalar('metric/' + me_name, me_op.get(), global_step=global_step)

            # 记录当前的学习率
            logger.add_scalar('meta/lr', optimizer.param_groups[0]['lr'], global_step=global_step)

            # 如果进度条具有设置后缀的属性，则执行以下操作
            if hasattr(progress_bar, 'set_postfix'):
                # 将指标字典中的指标值格式化
                kwargs = {me_name: '%.3f' % me_op.get() for me_name, me_op in
                          zip(metric_dict['name'], metric_dict['op'])}
                new_kwargs = {}
                for k, v in kwargs.items():
                    if 'lane' in k:
                        continue
                    new_kwargs[k] = v
                progress_bar.set_postfix(loss='%.3f' % float(loss),
                                         **new_kwargs)


if __name__ == "__main__":
    # 是否使用Cudnn卷积加速
    torch.backends.cudnn.benchmark = True

    # 读取配置参数
    args, cfg = merge_config()

    # 当前线程是否为主线程
    if args.local_rank == 0:
        # 获取工作目录
        work_dir = get_work_dir(cfg)

    # 是否分布式训练
    distributed = False
    if 'WORLD_SIZE' in os.environ:
        distributed = int(os.environ['WORLD_SIZE']) > 1
    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend='nccl', init_method='env://')

        if args.local_rank == 0:
            with open('.work_dir_tmp_file.txt', 'w') as f:
                f.write(work_dir)
        else:
            while not os.path.exists('.work_dir_tmp_file.txt'):
                time.sleep(0.1)
            with open('.work_dir_tmp_file.txt', 'r') as f:
                work_dir = f.read().strip()
    # 同步所有线程
    synchronize()
    cfg.test_work_dir = work_dir
    cfg.distributed = distributed
    # 删除临时文件
    if args.local_rank == 0:
        os.system('rm .work_dir_tmp_file.txt')

    # 打印开始训练的时间
    dist_print(datetime.datetime.now().strftime('[%Y/%m/%d %H:%M:%S]') + ' start training...')
    # 打印读取的配置信息
    dist_print(cfg)
    # 检查配置文件中的骨干网络类型是否合法
    assert cfg.backbone in ['18', '34', '50', '101', '152', '50next', '101next', '50wide', '101wide', '34fca']

    # 初始化训练数据（dataloader）
    train_loader = get_train_loader(cfg)
    # 获取配置的模型
    net = get_model(cfg)

    # 如果是分布式训练，则将模型封装为分布式模型
    if distributed:
        net = torch.nn.parallel.DistributedDataParallel(net, device_ids=[args.local_rank])

    # 获取优化器
    optimizer = get_optimizer(net, cfg)

    # 如果配置文件中指定了预训练模型，则加载预训练模型，并将模型参数复制到当前模型中
    if cfg.finetune is not None:
        dist_print('finetune from ', cfg.finetune)
        state_all = torch.load(cfg.finetune)['model']
        state_clip = {}  # only use backbone parameters
        for k, v in state_all.items():
            if 'model' in k:
                state_clip[k] = v
        net.load_state_dict(state_clip, strict=False)

    # 如果配置文件中指定了断点续训，则加载保存的模型及优化器状态，并从断点处继续训
    if cfg.resume is not None:
        dist_print('==> Resume model from ' + cfg.resume)
        resume_dict = torch.load(cfg.resume, map_location='cpu')
        net.load_state_dict(resume_dict['model'])
        if 'optimizer' in resume_dict.keys():
            optimizer.load_state_dict(resume_dict['optimizer'])
        resume_epoch = int(os.path.split(cfg.resume)[1][2:5]) + 1
    else:
        resume_epoch = 0

    # 生成学习率调节器，用于调节学习率
    scheduler = get_scheduler(optimizer, cfg, len(train_loader))
    dist_print(len(train_loader))

    # 生成评价指标的字典，用于度量网络性能
    metric_dict = get_metric_dict(cfg)

    # 生成损失函数的字典，用于计算神经网络的损失函数值
    loss_dict = get_loss_dict(cfg)

    # 生成日志记录器，记录神经网络的训练和评估
    logger = get_logger(work_dir, cfg)
    # cp_projects(cfg.auto_backup, work_dir)

    # 初始化最大性能指标值
    max_res = 0

    # 初始化当前性能指标值
    res = None

    # 从resume_epoch开始，到cfg.epoch结束，迭代训练，epoch为当前轮数
    for epoch in range(resume_epoch, cfg.epoch):
        # 训练
        train(net, train_loader, loss_dict, optimizer, scheduler, logger, epoch, metric_dict, cfg.dataset)
        # 重置训练数据集迭代器
        train_loader.reset()
        # 评估当前神经网络，得到性能指标值
        res = eval_lane(net, cfg, ep=epoch, logger=logger)

        # 当前性能指标值不为空，且大于当前最大性能指标值，则更新最大性能指标值
        if res is not None and res > max_res:
            max_res = res
            # 保存训练好的神经模型（网络参数、优化器状态、当前训练的轮次）
            save_model(net, optimizer, epoch, work_dir, distributed)
        # 将当前最大的性能指标值 ‘max_res’ 记录到日志器，同时记录当前的训练轮次
        logger.add_scalar('CuEval/X', max_res, global_step=epoch)

    # 关闭日志器
    logger.close()
