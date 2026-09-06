# H3-V100-GGUF-YF

MiniMax H3 的 NVIDIA V100 (SM70) 优化节点 —— **GGUF 支持版（基于官方 v1.3.x）**。

> ⚠️ **本项目基于 [rwashy/H3-V100](https://github.com/rwashy/H3-V100) v1.3.x 基线修改，原作者：rwashy。感谢原作者的优秀工作！**

## 为什么有这个分支

官方 `rwashy/H3-V100` 的 **v1.3.0 / v1.4.0 / v1.4.1** 开始要求传入的模型必须是 **Dynamic ModelPatcher**（`h3_optimize.py` 里有 `is_dynamic` 准入检查），而 **GGUF 加载的模型不是 Dynamic 类型**，会直接报错：

```
RuntimeError: H3 V100 requires ComfyUI DynamicVRAM. Remove --disable-dynamic-vram and --lowvram, restart ComfyUI, and reload the model.
```

这意味着 **官方新版（v1.3+）不支持 GGUF 模型**，只支持 int8/混合精度权重。

本分支基于官方 **v1.3.x 的代码基线**，**移除了这个 `is_dynamic` 准入检查**，使 GGUF 模型能正常通过；在你的 V100 16G 环境上验证过能用 GGUF 跑通。

## 与官方 v1.3.x 的差异

- **基于官方 v1.3.x 基线，移除了 `is_dynamic` 准入检查**，以恢复对 GGUF 模型的兼容。
- 加入了若干自用的增强/诊断文件（`diagnostics.py`、`sol_route_diagnostics.py`、独立的 `h3_optimize.py` 等）。
- `.pyd` 为在本机环境（PyTorch 2.10.0 + cu128、CPython 3.12.10）编译的版本。

> 注意：如果您只是想用官方最新（v1.4.1）+ int8 模型，请直接用官方；本分支解决的是「**16G 显存上想用 GGUF 模型**」这一场景。

## 使用前提

| 项 | 要求 |
|---|---|
| GPU | NVIDIA Tesla V100 / SM70 |
| 模型 | MiniMax H3 扩散模型（GGUF 或其他非 Dynamic 类型）|
| 启动参数 | 本分支**不需要** `--lowvram`/`--disable-dynamic-vram` 之外的 Dynamic 相关设置；与官方 v1.3.0 的启动方式一致 |
| Python / Torch | CPython 3.12、PyTorch 2.x（CUDA 12.x）。`.pyd` **与 Python/Torch/CUDA ABI 绑定** |

## 目录结构

```
H3-V100-GGUF-YF/
├── comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd  # 预编译 CUDA 扩展（SM70 Flash Attention）
├── h3_optimize.py          # 节点主逻辑（GGUF 兼容，无 is_dynamic 检查）
├── h3_mixed_precision.py   # V100 混合精度（FP16 计算岛 + FP32 残差/安全路径）
├── flash_attention.py      # SM70 Flash Attention 调度
├── sol_attention.py        # Sol-Attention（块稀疏）路线
├── tokenwise_chunking.py   # MLP/QKV 自适应分块 + 缓存
├── diagnostics.py          # （自用）诊断
├── sol_route_diagnostics.py# （自用）Sol 路由诊断
└── native/                 # Flash Attention CUDA 源码 + 编译脚本（可自编译 .pyd）
    ├── csrc/               # C++/CUDA 内核源码（SM70）
    ├── setup.py
    ├── build_isolated.py
    └── BUILD.md            # 编译说明
```

## 如何重新编译 `.pyd`（如果你的 Torch/CUDA 不同）

见 `native/BUILD.md`。需要：
- Visual Studio 2022 C++ 工具、Windows SDK；
- CUDA Toolkit 12.8；
- 用 ComfyUI 的 `python_embeded` 安装 build 依赖。

## Lua License

本分支沿用原项目许可证（GPL-3.0-only；含 BSD-3-Clause 组件）。详见 `LICENSE`、`NOTICE.md`、`licenses/`。
