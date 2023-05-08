import torch
from PIL import Image
import os
import pdb
import numpy as np
import cv2
from data.mytransforms import find_start_pos


def loader_func(path):
    return Image.open(path)

# 测试阶段
class LaneTestDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform=None, crop_size=None):
        super(LaneTestDataset, self).__init__()
        self.path = path
        self.img_transform = img_transform
        self.crop_size = crop_size
        with open(list_path, 'r') as f:
            self.list = f.readlines()
        self.list = [l[1:] if l[0] == '/' else l for l in self.list]  # exclude the incorrect path prefix '/' of CULane


    def __getitem__(self, index):
        name = self.list[index].split()[0]
        img_path = os.path.join(self.path, name)
        img = loader_func(img_path)

        if self.img_transform is not None:
            img = self.img_transform(img)
        img = img[:,-self.crop_size:,:]

        return img, name

    def __len__(self):
        return len(self.list)

# 训练阶段
class LaneClsDataset(torch.utils.data.Dataset):
    def __init__(self, path, list_path, img_transform = None,target_transform = None,simu_transform = None, griding_num=50, load_name = False,
                row_anchor = None,use_aux=False,segment_transform=None, num_lanes = 4):
        super(LaneClsDataset, self).__init__()
        self.img_transform = img_transform
        self.target_transform = target_transform
        self.segment_transform = segment_transform
        self.simu_transform = simu_transform
        self.path = path
        self.griding_num = griding_num
        self.load_name = load_name
        self.use_aux = use_aux
        self.num_lanes = num_lanes

        with open(list_path, 'r') as f:
            self.list = f.readlines()

        self.row_anchor = row_anchor
        self.row_anchor.sort()

    def __getitem__(self, index):
        # 从CULane的list的train_gt中读取数据（图片路径、标签路径） ，index为索引
        l = self.list[index]
        l_info = l.split()
        img_name, label_name = l_info[0], l_info[1]
        if img_name[0] == '/':
            img_name = img_name[1:]
            label_name = label_name[1:]

        # 读取标签
        label_path = os.path.join(self.path, label_name)
        label = loader_func(label_path)

        # 读取图片
        img_path = os.path.join(self.path, img_name)
        img = loader_func(img_path)
    
        # 是否做数据增强
        if self.simu_transform is not None:
            img, label = self.simu_transform(img, label)

        # 处理标签：图像 → 所有行锚中命中的车道线的列坐标数据
        # [
        #   [[x0,-1],[x1,-1], ... , [xk, yk], ..., [xn,yn]],     第一个车道在行锚中的位置, xk指的是第k个行锚的像素坐标， yk指的车道线是在第k个行锚的列坐标
        #   [],     第二个车道在行锚中的位置
        #   [],     第三个车道在行锚中的位置
        #   []      第四个车道在行锚中的位置
        # ]
        lane_pts = self._get_index(label)
        # get the coordinates of lanes at row anchors



        w, h = img.size
        # 处理网格，生成模型需要预测的分类结果
        cls_label = self._grid_pts(lane_pts, self.griding_num, w)
        # make the coordinates to classification label

        # 训练过程中，其中一个分支用分割做车道线检测，通过损失函数，提高特征的提取精确性
        if self.use_aux:
            assert self.segment_transform is not None
            seg_label = self.segment_transform(label)

        # 图像数据做数据增强
        if self.img_transform is not None:
            img = self.img_transform(img)

        # 分割：判断每一个点是不是车道线
        if self.use_aux:
            return img, cls_label, seg_label
        if self.load_name:
            return img, cls_label, img_name
        # 返回预处理后的图像数据和分类标签数据
        return img, cls_label

    def __len__(self):
        return len(self.list)

    def _grid_pts(self, pts, num_cols, w):
        # pts : numlane, n, 2
        # num_lane:行号， n：行锚数量，n2：值的个数，-1或者车道线的列坐标
        num_lane, n, n2 = pts.shape

        # 图片宽为 w，将其分为 构建 num_cols个grid cell
        col_sample = np.linspace(0, w - 1, num_cols)

        assert n2 == 2
        to_pts = np.zeros((n, num_lane))
        for i in range(num_lane):
            # pts即lane_pts， pti即 第i条车道线在所有行锚中的列坐标 得到一个一维数组[y0,y1, ... , yk, ..., yn]
            pti = pts[i, :, 1]
            # 遍历这些位置，将这些车道线所在的列坐标分别映射到每一个行锚的200个grid cells中
            to_pts[:, i] = np.asarray(
                [int(pt // (col_sample[1] - col_sample[0])) if pt != -1 else num_cols for pt in pti])
            # 结果是一个 4x18的矩阵 （4为车道线数量，18为行锚的数量）
            # [
            #   [200,200,200,200],    表示4条车道线在第一个行锚中 处于 第几个grid cell, 200表示没有车道线 （0~200）一共 w+1个， 第201个（200）表示没有车道线
            #   [.. ,.. ,.. ,..],
            #   [.. ,.. ,.. ,..],
            #   [.. ,.. ,.. ,..],
            # ]
        return to_pts.astype(int)

    def _get_index(self, label):
        # 标签图片的size
        w, h = label.size

        # resize后的图片的高是288，标签图像需要同步
        if h != 288:
            scale_f = lambda x : int((x * 1.0/288) * h)
            # 同幅度的映射和缩放，得到符合图像高度的先验值（18个行锚在288px中的行索引）  [.., .., .., .., ..]
            sample_tmp = list(map(scale_f,self.row_anchor))

        # 构建一个全0矩阵 4x18x2 (4条车道线，18个row anchor）
        all_idx = np.zeros((self.num_lanes,len(sample_tmp),2))

        # 遍历每一个行锚 （i：第几个行锚， r：行索引）
        for i,r in enumerate(sample_tmp):
            # 读取标签在该行的向量
            label_r = np.asarray(label)[int(round(r))]
            # 遍历每一条车道线
            for lane_idx in range(1, self.num_lanes + 1):
                # 定位第 lane_idx 条车道线的列坐标          00000000 1 0000000000 2 000000000 3

                pos = np.where(label_r == lane_idx)[0]
                # pos可能是空，代表没有该行锚没有命中第一条车道线，pos是个数组，因为车道线有宽度（或者说 cell是有宽度的）
                if len(pos) == 0:
                    # lane_idx 从 1开始的，所以要减1
                    # i 表示第几个行锚
                    all_idx[lane_idx - 1, i, 0] = r    # 行索引
                    all_idx[lane_idx - 1, i, 1] = -1   # 没有车道线，则为-1（无车道线）
                    continue
                pos = np.mean(pos) # 求的pos的平均值，获得该车道线的中心位置
                all_idx[lane_idx - 1, i, 0] = r
                all_idx[lane_idx - 1, i, 1] = pos  # 如果有车道线，则为其列坐标

        # data augmentation: extend the lane to the boundary of image
        # 先拷贝一份 all_idx，通过线性拟合，人工延伸（补全）车道线（到尽头）
        all_idx_cp = all_idx.copy()
        for i in range(self.num_lanes):
            if np.all(all_idx_cp[i,:,1] == -1):
                continue
            # if there is no lane

            valid = all_idx_cp[i,:,1] != -1
            # get all valid lane points' index
            valid_idx = all_idx_cp[i,valid,:]
            # get all valid lane points
            if valid_idx[-1,0] ==- all_idx_cp[0,-1,0]:
                # if the last valid lane point's y-coordinate is already the last y-coordinate of all rows
                # this means this lane has reached the bottom boundary of the image
                # so we skip
                continue
            if len(valid_idx) < 6:
                continue
            # if the lane is too short to extend

            valid_idx_half = valid_idx[len(valid_idx) // 2:,:]
            # 线性拟合
            p = np.polyfit(valid_idx_half[:,0], valid_idx_half[:,1],deg = 1)
            start_line = valid_idx_half[-1,0]
            pos = find_start_pos(all_idx_cp[i,:,0],start_line) + 1
            
            fitted = np.polyval(p,all_idx_cp[i,pos:,0])
            fitted = np.array([-1  if y < 0 or y > w-1 else y for y in fitted])

            assert np.all(all_idx_cp[i,pos:,1] == -1)
            # 将拟合后的值拼接在标签数据中
            all_idx_cp[i,pos:,1] = fitted
        if -1 in all_idx[:, :, 0]:
            pdb.set_trace()
        return all_idx_cp
