from model.model_culane import ParsingNet
def get_model(cfg):
    return ParsingNet(pretrained = True, backbone=cfg.backbone, num_grid_row = cfg.num_cell_row, num_cls_row = cfg.num_row, num_grid_col = cfg.num_cell_col, num_cls_col = cfg.num_col, num_lane_on_row = cfg.num_lanes, num_lane_on_col = cfg.num_lanes, use_aux = cfg.use_aux, input_height = cfg.train_height, input_width = cfg.train_width, fc_norm=cfg.fc_norm).cuda()