import torch
from model.backbone import resnet
from utils.common import initialize_weights
from model.seg_model import SegHead
from seblock import DualSEBlock


class ParsingNet(torch.nn.Module):
    def __init__(self, pretrained=True, backbone='50', num_grid_row=None, num_cls_row=None, num_grid_col=None,
                 num_cls_col=None, num_lane_on_row=None, num_lane_on_col=None, use_aux=False, input_height=None,
                 input_width=None, fc_norm=False):
        super(ParsingNet, self).__init__()
        self.num_grid_row = num_grid_row
        self.num_cls_row = num_cls_row
        self.num_grid_col = num_grid_col
        self.num_cls_col = num_cls_col
        self.num_lane_on_row = num_lane_on_row
        self.num_lane_on_col = num_lane_on_col
        self.use_aux = use_aux
        self.dim1 = self.num_grid_row * self.num_cls_row * self.num_lane_on_row  # 行锚的grid cell * 行锚的数量 * 用行锚的车道线
        self.dim2 = self.num_grid_col * self.num_cls_col * self.num_lane_on_col  # 列锚的grid cell * 列锚的数量 * 用列锚的车道线
        self.dim3 = 2 * self.num_cls_row * self.num_lane_on_row  # 行锚的数量 * 用行锚的车道线 * 2
        self.dim4 = 2 * self.num_cls_col * self.num_lane_on_col  # 列锚的数量 * 用列锚的车道线 * 2
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4
        mlp_mid_dim = 2048
        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = resnet(backbone, pretrained=pretrained)

        # for avg pool experiment
        # self.pool = torch.nn.AdaptiveAvgPool2d(1)
        # self.pool = torch.nn.AdaptiveMaxPool2d(1)

        # self.register_buffer('coord', torch.stack([torch.linspace(0.5,9.5,10).view(-1,1).repeat(1,50), torch.linspace(0.5,49.5,50).repeat(10,1)]).view(1,2,10,50))

        self.cls = torch.nn.Sequential(
            # fc_norm为True则使用LayerNorm对输入进行归一化，否则使用恒等函数Identity
            torch.nn.LayerNorm(self.input_dim) if fc_norm else torch.nn.Identity(),
            # 线性层，做线性变换（将输入和一组可学习的权重相乘并加上一个偏置）
            torch.nn.Linear(self.input_dim, mlp_mid_dim),  # in_feature, out_feature = 2048
            # 非线性激活函数
            torch.nn.ReLU(),
            # 线性层
            torch.nn.Linear(mlp_mid_dim, self.total_dim),  # in_feature, out_feature
        )
        self.pool = torch.nn.Conv2d(512, 8, 1) if backbone in ['34', '18', '34fca'] else torch.nn.Conv2d(2048, 8, 1)

        self.se512 = DualSEBlock(512)
        self.se256 = DualSEBlock(256)
        self.se128 = DualSEBlock(128)

        if self.use_aux:
            self.seg_head = SegHead(backbone, num_lane_on_row + num_lane_on_col)

        initialize_weights(self.cls, self.se512, self.se256, self.se128)

    def forward(self, x):
        # 输入的x的样例 [1,3,288,800]   1：batch为1， 3：通道 RGB一共3个   288：h    800：w
        # 输入backbone，返回3层特征图的结果（依次下采样） 分别为 x2(1,128,36,100)，x3(1,256,18,50)，fea(1,512,9,25)
        x2, x3, fea = self.model(x)
        # 注意力机制：
        fea_att = self.se512(fea)
        fea = fea_att * fea

        # 是否做分割任务，
        if self.use_aux:
            x2_att = self.se128(x2)
            x3_att = self.se256(x3)
            x2 = x2 * x2_att
            x3 = x3 * x3_att
            # 分割任务：将三组特征图拼接后 -> 卷积 -> 分割
            seg_out = self.seg_head(x2, x3, fea)  # 输出的channel都是128

        # 卷积
        fea = self.pool(fea)

        # reshape 为 列数为 input_dim 的矩阵
        fea = fea.view(-1, self.input_dim)

        # 连全连接  input_dim
        # 由于是分类任务，而标签为18x4的矩阵（4个车道线 ，18个行锚），每个行锚有201个cell，需要判断这201个位置的分类
        # 因此得到的是 [batch, 201, 18, 4]
        out = self.cls(fea)

        # loc_row 和 loc_col 为分别用 行锚和列锚得到的分类结果
        pred_dict = {
            'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
            'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col,
                                                                    self.num_lane_on_col),
            'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2, self.num_cls_row,
                                                                                              self.num_lane_on_row),
            'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col),
            'att_weights': {fea_att}
        }

        # 是否有分割任务的分支
        if self.use_aux:
            pred_dict['seg_out'] = seg_out

        return pred_dict

    # 测试时做数据增强
    def forward_tta(self, x):
        x2, x3, fea = self.model(x)

        pooled_fea = self.pool(fea)
        n, c, h, w = pooled_fea.shape

        left_pooled_fea = torch.zeros_like(pooled_fea)
        right_pooled_fea = torch.zeros_like(pooled_fea)
        up_pooled_fea = torch.zeros_like(pooled_fea)
        down_pooled_fea = torch.zeros_like(pooled_fea)

        left_pooled_fea[:, :, :, :w - 1] = pooled_fea[:, :, :, 1:]
        left_pooled_fea[:, :, :, -1] = pooled_fea.mean(-1)

        right_pooled_fea[:, :, :, 1:] = pooled_fea[:, :, :, :w - 1]
        right_pooled_fea[:, :, :, 0] = pooled_fea.mean(-1)

        up_pooled_fea[:, :, :h - 1, :] = pooled_fea[:, :, 1:, :]
        up_pooled_fea[:, :, -1, :] = pooled_fea.mean(-2)

        down_pooled_fea[:, :, 1:, :] = pooled_fea[:, :, :h - 1, :]
        down_pooled_fea[:, :, 0, :] = pooled_fea.mean(-2)
        # 10 x 25
        fea = torch.cat([pooled_fea, left_pooled_fea, right_pooled_fea, up_pooled_fea, down_pooled_fea], dim=0)
        fea = fea.view(-1, self.input_dim)

        out = self.cls(fea)

        return {'loc_row': out[:, :self.dim1].view(-1, self.num_grid_row, self.num_cls_row, self.num_lane_on_row),
                'loc_col': out[:, self.dim1:self.dim1 + self.dim2].view(-1, self.num_grid_col, self.num_cls_col,
                                                                        self.num_lane_on_col),
                'exist_row': out[:, self.dim1 + self.dim2:self.dim1 + self.dim2 + self.dim3].view(-1, 2,
                                                                                                  self.num_cls_row,
                                                                                                  self.num_lane_on_row),
                'exist_col': out[:, -self.dim4:].view(-1, 2, self.num_cls_col, self.num_lane_on_col)}


def get_model(cfg):
    return ParsingNet(pretrained=True, backbone=cfg.backbone, num_grid_row=cfg.num_cell_row, num_cls_row=cfg.num_row,
                      num_grid_col=cfg.num_cell_col, num_cls_col=cfg.num_col, num_lane_on_row=cfg.num_lanes,
                      num_lane_on_col=cfg.num_lanes, use_aux=cfg.use_aux, input_height=cfg.train_height,
                      input_width=cfg.train_width, fc_norm=cfg.fc_norm).cuda()
