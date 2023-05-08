# dataset path
dataset= 'CULane'
data_root= './dataset/CULane' # Need to be modified before running

# train strategy
epoch= 50
batch_size= 32
optimizer= 'SGD'     # ['SGD','Adam']
learning_rate= 0.05
weight_decay= 0.0001
momentum= 0.9
scheduler= 'multi'   # ['multi','cos']
steps= [25,38]
gamma= 0.1
warmup= 'linear'
warmup_iters= 695

# network
use_aux= False
griding_num= 200
backbone= '34'

# loss
sim_loss_w= 0.0
shp_loss_w= 0.0

# exp
note= ''
log_path= './log'

# finetune or resume model path
finetune= None
resume= None
test_model=''
test_work_dir = ''

# network
tta=True
num_lanes= 4
var_loss_power= 2.0
auto_backup= True

# 行锚和列锚的数量
num_row= 72
num_col= 81

# 输入图像的size
train_width= 1600
train_height= 320

# 行锚和列锚中的grid cell数
num_cell_row= 200
num_cell_col= 100
mean_loss_w= 0.05
fc_norm= True
crop_ratio = 0.6
