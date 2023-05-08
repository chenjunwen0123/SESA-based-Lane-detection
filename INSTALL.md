# 运行环境配置

### PS:建议使用[AutoDL](https://www.autodl.com/home)训练部署

1. 克隆项目

    ```Shell
    git https://github.com/chenjunwen0123/SESA-based-Lane-detection.git
    cd SESA-based-Lane-detection
    ```

2. 创建并激活Conda虚拟环境

    ```Shell
    conda create -n lane-det python=3.10 -y
    conda activate lane-det
    ```

3. 安装项目依赖

    ```Shell
    # install pytorch
    conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia
    
    # install requirement
    pip install -r requirements.txt
    
    # Nvidia DALI -fast data loading lib
    pip install --extra-index-url https://developer.download.nvidia.com/compute/redist --upgrade nvidia-dali-cuda110
    
    cd my_interp
    sh build.sh
    # If this fails, you might need to upgrade your GCC to v7.3.0
    ```

# 数据集配置

## TuSimple

下载TuSimple

```bash
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/train_set.zip
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_set.zip
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/datasets/1/test_baseline.json
wget https://s3.us-east-2.amazonaws.com/benchmark-frontend/truth/1/test_label.json
```

解压 `train_set.zip`、`test_set.zip` 到与 `test_label.json` 同一目录下

解压后的TuSimple 文件结构 

```shell
$TUSIMPLE
|──clips
|──label_data_0313.json
|──label_data_0531.json
|──label_data_0601.json
|──test_tasks_0627.json
|──test_label.json
|──readme.md
```
使用 `convert_tusimple.py` 手动生成训练和测试标签，通过`--root` 指定解压后的TuSimple数据集的目录，该操作会在那个目录下生成 `train_gt.txt` 和 `test.txt`

![image-20230508161226383](.\assets\tusimple-generate-result.png)

```bash
python scripts/convert_tusimple.py --root /path/to/your/tusimple
```
配置训练配置，回到项目目录中，找到configs目录，其文件结构如下，脚本名字格式为 `<dataset>_<backbone>.py`

```bash
$CONFIGS
|──culane_res18.py
|──culane_res34.py
|──curvelanes_res18.py
|──curvelanes_res34.py
|──tusimple_res18.py
|──tusimple_res34.py
```

配置`tusimple_<backbone>.py`的以下属性（后续CULane、CurveLanes步骤一致，不再赘述）：

- `data_root`：数据集的目录
- `log_path`：训练生成模型和event日志的缓冲目录（可选）

## CULane

### 下载数据集

访问官网：[CULane Dataset (xingangpan.github.io)](https://xingangpan.github.io/projects/CULane.html)

下载数据集的两个地址：

- [Google Drive](https://drive.google.com/open?id=1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)
- [Baidu Cloud](https://pan.baidu.com/s/1KUtzC24cH20n6BtU5D0oyw)

通过`tar_culane`解压下载的所有文件到同一个目录下

```bash
python scripts/tar_culane.py --tar_path path/to/culane
```

解压后文件目录结构为

```shell
$CULANE
|──driver_100_30frame
|──driver_161_90frame
|──driver_182_30frame
|──driver_193_90frame
|──driver_23_30frame
|──driver_37_30frame
|──laneseg_label_w16
|──list
```
### 预处理标签

由于CULane的标签是从原图像中将车道线标记并分割出来的等尺寸图像（每帧图像有一个对应的标签图像），我们需要对其进行预处理，批量处理为一个供模型一次性读取的文件，避免训练时因读取标签图像的频繁IO）

- 运行`cache_culane_ponits.py` ，处理CULane的`train_gt.txt` 和`test.txt` ，生成 `culane_anno_cache.json` 
- 存储形式为键值对，键为 训练图像的全名，值为一个维度为 `[4,35,2]` 的数组中，长度为 4 的维度表示 4 条道路标线。每条道路标线都有一个包含 35 个点的数组。这 35 个点表示沿着道路标线的采样点。对于每个包含 2 个元素的数组，第一个元素表示这个采样点的 x 坐标，第二个元素表示这个采样点的 y 坐标。所以，在这个数组中，每个点都是一个形如 `(x, y)` 的坐标对。

```Shell
python scripts/cache_culane_ponits.py --root /path/to/your/culane
```
## 安装验证工具

CULane数据集的验证阶段需要用到专门的Evaluator

- 安装Testing专用的Evaluator，需要安装Opencv C++

  - 安装`opencv`（核心库）和 `opencv_contrib`（贡献库，包含更强大的计算机视觉相关模块）

  ```bash
  # Install minimal prerequisites (Ubuntu 18.04 as reference)
  sudo apt update && sudo apt install -y cmake g++ wget unzip
  # Download and unpack sources
  wget -O opencv.zip <https://github.com/opencv/opencv/archive/4.x.zip>
  wget -O opencv_contrib.zip <https://github.com/opencv/opencv_contrib/archive/4.x.zip>
  unzip opencv.zip
  unzip opencv_contrib.zip
  ```

  - 手动安装IPPICV

    - 在解压好的`opencv-4.x`文件夹中，编辑 `/3rdparty/ippicv/ippicv.cmake`

    ```bash
    ## need part 1
    set(IPPICV_COMMIT "a56b6ac6f030c312b2dce17430eef13aed9af274")
    
     ...
     ...
    
    ## need part 2:filename
    if(X86_64)
          set(OPENCV_ICV_NAME "ippicv_2020_win_intel64_20191018_general.zip")
          set(OPENCV_ICV_HASH "879741a7946b814455eee6c6ffde2984")
        else()
          set(OPENCV_ICV_NAME "ippicv_2020_win_ia32_20191018_general.zip")
          set(OPENCV_ICV_HASH "cd39bdf0c2e1cac9a61101dad7a2413e")
    
     ...
     ...
    
    ## need part 3
    set(THE_ROOT "${OpenCV_BINARY_DIR}/3rdparty/ippicv")
      ocv_download(FILENAME ${OPENCV_ICV_NAME}
                   HASH ${OPENCV_ICV_HASH}
                   URL
                     "${OPENCV_IPPICV_URL}"
                     "$ENV{OPENCV_IPPICV_URL}"
                     "<https://raw.githubusercontent.com/opencv/opencv_3rdparty/${IPPICV_COMMIT}/ippicv/>"
                   DESTINATION_DIR "${THE_ROOT}"
                   ID IPPICV
                   STATUS res
                   UNPACK RELATIVE_URL)
    ```

    - 根据当前系统架构拼接下载地址并下载到指定文件夹

    ```bash
    <https://raw.githubusercontent.com/opencv/opencv_3rdparty/a56b6ac6f030c312b2dce17430eef13aed9af274/ippicv/ippicv_2020_win_intel64_20191018_general.zip>
    ```

    - 替换链接
      - 比如下载到了`/root/autodl-tmp/`
      - 则修改part3中的URL为

    ```bash
    ocv_download(FILENAME ${OPENCV_ICV_NAME}
                   HASH ${OPENCV_ICV_HASH}
                   URL
                     "${OPENCV_IPPICV_URL}"
                     "$ENV{OPENCV_IPPICV_URL}"
                     "file:///root/autodl-tmp/"
    ```

  - 编译opencv和opencv_contrib

  ```bash
  # Create build directory and switch into it
  mkdir -p build && cd build
  # Configure
  cmake -DOPENCV_EXTRA_MODULES_PATH=../opencv_contrib-4.x/modules ../opencv-4.x
  # Build
  cmake --build .
  ```

  - 默认地， OpenCV 会被安装到 `/usr/local`下
    - `/usr/local/bin` - executable files
    - `/usr/local/lib` - libraries (.so)
    - `/usr/local/cmake/opencv4` - cmake package
    - `/usr/local/include/opencv4` - headers
    - `/usr/local/share/opencv4` - other files (e.g. trained cascades in XML format)
  - 建立软链接

  ```bash
  ln -s /usr/local/include/opencv4/opencv2 /usr/local/include/opencv2
  ```

- 安装Evaluate

  - 回到项目目录，进入`evaluation/culane`
  - 执行编译

  ```bash
  cd evaluation/culane
  mkdir build && cd build
  cmake ..
  make
  mv culane_evaluator ../evaluate
  ```

  

## CurveLanes

### 下载数据集

CurveLanes的下载见其主页：[CurveLanes-HomePage](https://github.com/SoulmateB/CurveLanes) 

CurveLanes的文件目录

```bash
$CurveLanes
|──test
|──train
|──valid
```
### 预处理数据集

与CULane相同，需要将标签图像处理为一次性读取的标签文件，执行以下命令，会在CurveLanes数据集的目录下生成 `curvelanes_anno_cache_train.json` ，同时会在验证集中生成对应图像的 `.lines.txt` 车道线标记文件，以提供模型像处理CULane一样的便利

```bash
python scripts/convert_curvelanes.py --root /path/to/your/curvelanes

python scripts/make_curvelane_as_culane_test.py --root /path/to/your/curvelanes
```
