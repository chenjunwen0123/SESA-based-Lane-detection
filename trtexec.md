`trtexec` 是一个命令行工具，用于在 NVIDIA TensorRT 中基准测试和部署模型。TensorRT 是一个高性能的深度学习推理优化器和运行时库，可为 NVIDIA GPU 提供低延迟和高吞吐量。通过使用 `trtexec`，您可以将 TensorFlow、PyTorch 或其他框架训练的模型转换为 TensorRT 引擎，并评估其在不同设置下的性能。

## 安装

**Linux:**

在 Linux 上，您可以使用 NVIDIA SDK Manager 或在支持的平台上使用 Debian 包或 Tar 包来安装 TensorRT。安装完成后，`trtexec` 会作为一个可执行文件安装在系统中。

在 Linux 上安装 TensorRT，您可以使用以下方法之一：使用 NVIDIA SDK Manager，或者从 NVIDIA 官方网站下载并安装 Debian 包或 Tar 包。以下是详细步骤：

- **方法 1：使用 NVIDIA SDK Manager**

1. 访问 [NVIDIA SDK Manager](https://developer.nvidia.com/nvidia-sdk-manager) 页面并下载适用于 Linux 的安装程序。
2. 安装并运行 SDK Manager。
3. 登录您的 NVIDIA 开发者帐户。如果您还没有帐户，请注册一个。
4. 选择要安装的平台（例如，Jetson 或 x86_64）。
5. 从 SDK Manager 中选择要安装的组件。确保选择 TensorRT。
6. 按照屏幕上的说明安装所选组件。

- **方法 2：从 NVIDIA 官方网站下载并安装 Debian 包或 Tar 包**

1. 访问 [NVIDIA TensorRT 下载页面](https://developer.nvidia.com/tensorrt-getting-started)。

2. 登录您的 NVIDIA 开发者帐户。如果您还没有帐户，请注册一个。

3. 选择 Linux 作为目标操作系统，并根据您的 Linux 发行版选择 Debian 包或 Tar 包。

   - **安装 Debian 包：**

     1. 下载并安装 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)（TensorRT 需要 CUDA 支持）。

     2. 下载 cuDNN 库（与您安装的 CUDA 版本兼容）并将其解压缩到 `/usr/local/cuda`。您可以在 [cuDNN 下载页面](https://developer.nvidia.com/cudnn) 上找到它。

     3. 下载相应的 TensorRT Debian 包。

     4. 使用 `dpkg` 安装下载的 TensorRT 包。例如，如果您下载了名为 `nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.3.4-ga-20210226_1-1_amd64.deb` 的包，请运行以下命令：

        ```bash
        sudo dpkg -i nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.2.3.4-ga-20210226_1-1_amd64.deb
        ```

     5. 更新 APT 包索引：

        ```bash
        sudo apt-get update
        ```

     6. 安装 TensorRT：

        ```
        sudo apt-get install tensorrt
        ```


   - **安装 Tar 包：**

     1. 下载并安装 [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)（TensorRT 需要 CUDA 支持）。

     2. 下载 cuDNN 库（与您安装的 CUDA 版本兼容）并将其解压缩到 `/usr/local/cuda`。您可以在 [cuDNN 下载页面](https://developer.nvidia.com/cudnn) 上找到它。

     3. 下载相应的 TensorRT Tar 包并将其解压缩。

     4. 将 TensorRT 解压缩目录添加到 `LD_LIBRARY_PATH` 环境变量：

        ```bash
        export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to the extracted TensorRT directory>
        ```

        请将 `<path to the extracted TensorRT directory>` 替换为实际的解压缩目录。 5. （可选）如果您计划经常使用 TensorRT，可以将上述 `export` 命令添加到您的 `~/.bashrc` 文件中，以便在每次打开新终端时自动设置 `LD_LIBRARY_PATH`：

         ```bash
         bash echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:<path to the extracted TensorRT directory>/lib' >> ~/.bashrc
         ```

     5. 安装 Python 的 TensorRT 绑定（如果需要）：

        将 Python 的 TensorRT 绑定安装到您的 Python 环境中。首先，导航到 TensorRT 解压缩目录中的 `python` 文件夹，找到与您的 Python 版本和 CUDA 版本匹配的 `tensorrt-<version>-cp<python_version>-none-linux_x86_64.whl` 文件。接着，使用 `pip` 安装它：

        ```bash
        bashCopy code
        pip install <path to the extracted TensorRT directory>/python/tensorrt-<version>-cp<python_version>-none-linux_x86_64.whl
        ```

        请将 `<path to the extracted TensorRT directory>` 替换为实际的解压缩目录，并根据实际情况替换 `<version>` 和 `<python_version>`。安装完成后，您应该可以在系统中使用 TensorRT 和 `trtexec`。通过在终端中键入 `trtexec`，验证其是否已正确安装。如果一切正常，您应该会看到关于 `trtexec` 用法的信息。



**Windows:**

在 Windows 上，您需要下载 TensorRT 的 zip 文件，解压缩它，并将 `trtexec` 可执行文件添加到您的系统路径中。您可以在 [NVIDIA TensorRT 下载页面](https://developer.nvidia.com/tensorrt-getting-started) 上找到相应的下载链接。

**Docker:**

NVIDIA 也提供了一个预先安装了 TensorRT 的 Docker 镜像。要使用 Docker 镜像，请确保您已安装了 [NVIDIA Docker](https://github.com/NVIDIA/nvidia-docker)，然后运行以下命令：

```bash
docker pull nvcr.io/nvidia/tensorrt:<version>-<os>-<os-version>
```

请将 `<version>`、`<os>` 和 `<os-version>` 替换为相应的值。例如：

```bash
docker pull nvcr.io/nvidia/tensorrt:21.09-py3
```

然后使用以下命令启动 Docker 容器：

```bash
docker run --rm -it --gpus all nvcr.io/nvidia/tensorrt:<version>-<os>-<os-version>
```

现在您应该在 Docker 容器内，可以直接使用 `trtexec`。

## 使用

`trtexec` 提供了许多参数，允许您灵活地优化和测试模型。一些常见的参数包括：

- `--onnx`：指定 ONNX 模型文件的路径
- `--uff`：指定 UFF 模型文件的路径
- `--model`：指定 Caffe 模型文件的路径
- `--deploy`：指定 Caffe deploy 文件的路径
- `--fp16`：使用半精度（FP16）执行推理
- `--int8`：使用 INT8 精度执行推理
- `--batch`：指定推理的批量大小
- `--workspace`：设置工作空间大小（以兆字节为单位）

要使用 `trtexec`，首先确保您已安装了 TensorRT。然后，您可以在命令行中运行 `trtexec` 命令，指定模型文件和其他所需参数。例如，要使用半精度（FP16）基准测试 ONNX 模型，您可以运行以下命令：

```
trtexec --onnx=model.onnx --fp16
```

有关 `trtexec` 的更多详细信息和选项，请参阅 [NVIDIA TensorRT 文档](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec)。



