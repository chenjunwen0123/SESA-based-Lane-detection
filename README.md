# SESA-based-Lane-detection
# 成果样例
![demo-night-1.png](assets\demo-night-1.png)

![demo-night-2.png](assets\demo-night-2.png)

![image-20230508142935076](assets\demo-dazzle-1.png)

![image-20230508143012771](assets\demo-shadow-1.png)

![image-20230508143040448](assets\demo-curve-1.png)


# 预配置
包括依赖安装、数据集配置
详情见 [INSTALL.md](./INSTALL.md)

# 使用方法
### 启动训练

1. 需要指定训练配置文件`<config.py>`，使用`--log_path`指定模型文件和训练日志缓存的目录

2. **单卡训练：**

```shell
python train.py configs/<config.py> --log_path /path/to/your/work/dir
```
3. **分布式训练：**

   使用 `--nproc_per_node` 指定工作的GPU数量

```
python -m torch.distributed.launch --nproc_per_node=8 train.py configs/<config.py> --log_path /path/to/your/work/dir
```


### 验证模型

1. 创建一个`tmp`文件夹作为验证中间结果缓存目录

   ```shell
   mkdir tmp
   ```

2. **单卡测试**

   ```shell
   python test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
   ```

3. **分布式测试**

   ```shell
   python -m torch.distributed.launch --nproc_per_node=8 test.py configs/culane_res18.py --test_model /path/to/your/model.pth --test_work_dir ./tmp
   ```

   

# 可视化
## 可视化测试

使用`demo.py`基于CULane数据集可视化生成9个场景的测试结果

**注意：**输出的是每个场景下的一段后缀为`.avi`的视频，但视频并不是完全连续的，是将预测结果剪帧拼接的。

**使用：**通过`config.py`指定模型参数，使用`--test_model	`指定用于可视化的训练好的模型

```shell
python demo.py configs/<config.py> --test_model /path/to/your/culane_res18.pth
```

## 可视化统计

使用`Tensorboard`进行可视化统计。

### 启停TensorBoard

1. **启动：**使用`--port`指定端口（默认为6007），使用`--logdir`指定TensorBoard的监测目录

   监测对象为该文件夹下所有的以 `events.out.tfevents.`为前缀的训练日志文件 

   ```shell
   tensorboard --port 6007 --logdir /path/to/your/log
   ```

2. **停用**：销毁 tensorboard的后台进程

   ```shell
   kill $(pgrep -f tensorboard)
   ```

   也可以选择自行删除

   ```shell
   ps -ef | grep tensorboard
   
   kill -9 <tensorboard_pid>
   ```

3. **访问**：通过指定的端口，在本地访问Tensorboard面板（http://localhost:6007）

## 可视化网络

1. 使用`pt2onnx.py`将 `.pth` 模型文件转换为 `.onnx`模型文件

   ```shell
   python deploy/pt2onnx.py --config_path configs/culane_res34.py --model_path path/to/your/model.pth
   ```

2. 使用[Netron](https://netron.app/)可视化网络模型

   使用Netron app打开生成的`.onnx` 文件

   

# 可视化部署

1. 使用 trtexec 将`.onnx` 转换成`.engine` 模型，trtexec的具体使用部署方法见  [trtexec.md](./trtexec.md)

   ```shell
   trtexec --onnx=weights/culane_res34.onnx --saveEngine=weights/culane_res34.engine
   ```

2. 使用 `trt_infer.py` 对指定的视频源（可以为行车记录仪视频）进行检测 

   - 通过 `--config_path` 指定模型配置文件

   - 通过 `--engine_path` 指定 `.engine` 模型文件
   - 通过 `--video_path` 指定 目标视频源文件路径

   ```shell
   python deploy/trt_infer.py --config_path  configs/culane_res34.py --engine_path weights/culane_res34.engine --video_path example.mp4
   ```

# 工具

## 合并events

若在 `<config.py>` 中指定 `resume` 的话，会重新生成一个缓存文件夹（包括新的 `events.out.tfevents `文件和 `.pth` 模型文件）若需要在Tensorboard观察到新的和旧的events文件的完整连续曲线，需要将两者合并。

使用`merge_events.py` 合并两个events文件

- 使用`--first_event` 指定第一个`events.out.tfevents` 文件，使用`second_event` 指定第二个  `events.out.tfevents`  文件
- 使用 `--joint_point `指定衔接的step（第二个`events.out.tfevents` 文件接在第一个`events.out.tfevents` 的第一条记录的step）
- 新的events文件将会在当前目录下生成

```bash
python scripts/merge_events.py --first_event path/to/your/first_event  --second_event path/to/your/second_event --joint_point <joint_step>
```



