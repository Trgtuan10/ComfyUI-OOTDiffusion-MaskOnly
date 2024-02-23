# ComfyUI OOTDiffusion

A ComfyUI custom node that simply integrates the [OOTDiffusion](https://github.com/levihsu/OOTDiffusion) functionality.

一个简单接入 OOTDiffusion 的 ComfyUI 节点。

👇 拖动下面的图片到 ComfyUI 前端即可运行 👇 Drag into ComfyUI frontend

![](./assets/graph.png)

## Instruction 指南

根据 https://git-lfs.com 安装 git lfs：

Ubuntu / Debian:

```txt
sudo apt install git-lfs
```

git lfs 初始化：

```txt
git lfs install
```

拉取 huggingface 🤗 库至 ComfyUI 根目录下的 `models/OOTDiffusion` 目录：

```txt
git clone https://huggingface.co/levihsu/OOTDiffusion models/OOTDiffusion
```

拉取 huggingface 时大约会下载 8 个模型，假如断开连接，可以使用下面命令恢复下载：

```txt
cd models/OOTDiffusion
git lfs fetch
git checkout main
```

创建环境并下载依赖：

```txt
conda create -n ootd
conda activate ootd

# 选择安装 11.8 / 12.1 cuda toolkit
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

# 安装此项目的依赖
pip install -r custom_nodes/ComfyUI-OOTDiffusion/requirements.txt
```

启动 ComfyUI 即可。

## FAQ 常见错误

> OSError: CUDA_HOME environment variable is not set. Please set it to your CUDA install root.
>
> 解决方法：`conda install -c conda-forge cudatoolkit-dev`。
> 参照：https://github.com/conda/conda/issues/7757

> subprocess.CalledProcessError: Command '['where', 'cl']' returned non-zero exit status 1.
>
> 解决办法：仅在 Windows 下出现，可能需要配置一下 MSVC 编译器。

## Node 节点

Load OOTDiffusion: 加载 OOTDiffusion Pipeline

OOTDiffusion Generate: 生成图像

## Example image 示例图片

[衣服 1](./assets/cloth_1.jpg)

[模特 1](./assets/model_1.png)

## Detail 细节

目前此项目只是对 OOTDiffusion 的功能做了个简单的迁移。
OOTDiffusion 本体依赖于 `diffusers==0.24.0` 实现，所以假如有其他节点的依赖冲突是没办法解决的（本就不该依赖 diffusers）。
靠 vendor 也能解决，所以也不是大问题。

不使用 huggingface_hub 是因为 OOTD 提供的仓库并不是一个单纯的 diffusion model structure，
里面还包含了独立的 openpose 和 humanparsing 模型文件。
目前只有 openai/clip-vit-large-patch14 是使用 huggingface_hub 下载的。

在 `Ubuntu 22.02` / `Python 3.10.x` 下可以正常运行。Windows 没有测试过。