# H3-V100 (GGUF-YF) 在 ComfyUI 0.35 上的兼容性与实测报告

> 测试日期：2026-09-10
> 测试者：firstddf
> 被测版本：本仓库（基于 rwashy/H3-V100 v1.3.0 的 GGUF 兼容分支）
> 相关文档：[`V100优化方案.md`](https://github.com/firstddf)（更早的 0.29/0.33 时期结论）

---

## 0. 摘要（TL;DR）

| 结论 | 判定 |
|---|---|
| **ComfyUI 0.35 没有让 V100-H3 变慢** | ✅ 本节点全部补丁正常挂载；21 次运行中**没有一次**由本节点自身引发故障 |
| **`ModelAttentionBackend`（ComfyUI 内置节点）静默覆盖本节点注意力后端** | ⚠️ 真因之一：每步 **+20~25%** → **must not connect** |
| **第三方闭源包 `TE-Speed-MiniMaxH3` 与 0.35 不兼容** | ❌ 真因之二：内嵌 0.33 版 H3 前向 → `FinalLayer` 签名不符 → **必崩**（实测 2 次） |
| **ComfyUI-Manager 启动抓取会污染测速** | ⚠️ 真因之三：启动后前 ~7 分钟与采样并发，s/it 虚高 **~8%** |
| **段档（MLP 分块数）不是成本驱动因素** | ❌ 伪因：把段数从 10–14 降到 5，s/it 反升 9%；**该补丁已回滚** |
| **可用显存 `eff` 也不是成本驱动因素** | ❌ 伪因：`--reserve-vram 1.0` 使 eff +534 MiB，s/it 无改善 |
| **`sol_attn` 在本分支不可用** | ❌ 该 `.pyd` 未编译 Sol kernel → 静默回退纯 PyTorch → **慢 2.55×**（上游 v2.0 已提供原生实现） |
| **DynamicVRAM 与 H3_V100 不兼容** | ❌ 与自适应分块 + Comfy Compiler 冲突 → **必崩**（`aimdo memory compile error`） |
| **`capacity_guard_tokens = 31744` 不是总长度上限** | 只是"单次不分块 MLP 分配"上限，超过即强制分块（34706 token 实测照常运行） |

### 各配置实测最优

| 分辨率 × 时长 | token | 4 步 s/it | 采样 | 总时 | 条件 |
|---|---|---|---|---|---|
| 0.3 MP × 10 s | 23845 | **42.56** | 170 s | 247 s | flash + 跳 203/222 |
| 0.5 MP × 5 s | 21181 | 42.77 | 171 s | 356 s | flash（203 在链） |
| 0.4 MP × 10 s | 31699 | 68.90 | 275 s | 387 s | flash（203 在链） |
| 0.3 MP × 15 s | 34706 | 77.31 | 308 s | 515 s | flash + 跳 203/222 |

**成本几乎只由 token 数决定**：`s/it ≈ 6.05e-4·N + 4.95e-8·N²`（±5%）。因此**怎么调分辨率/时长，都比"少接一个多余节点"更划算**——删掉 203 一次就省 20~25%。

> **为什么用 GGUF**：V100 没有 INT8 tensor core —— 实测 INT8 GEMM 只有 FP16 的 **0.63×**、权重还大 1.78×，FP8 直接不可用（`supports_fp8_compute=False`）。所以 **GGUF Q4_K_M 在这台卡上是最优格式，不是妥协**（见 §12）。
>
> **一句话**：这次"0.35 升级后变慢"是**工作流里多接的两个节点**（一个覆盖注意力、一个内嵌旧前向）造成的，外加一个测试环境陷阱（Manager 抓取）。H3-V100 与 0.35 本身没有问题。
>
> 全部 21 次运行的逐次数据见 §6，完整结论见 §11；两条已评估并关闭的岔路见 §12 / §13。

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| ComfyUI | **0.35.0**（revision `40c4fcdf`，升级前为 `v0.33.3`） |
| PyTorch | 2.10.0+cu128 |
| xformers | 0.0.35（日志 `Using xformers attention`） |
| comfy_kitchen | 0.2.33（cuda 后端 `disabled: True`，cu128 下不满足 cu130 要求） |
| GPU | Tesla V100-SXM2-16GB（SM70），显存 16 GiB |
| RAM | 64 GB |
| Python | CPython 3.12.10（`python_embeded`） |
| 启动参数 | `--listen 0.0.0.0 --port 8188 --lowvram --reserve-vram 0.5 --disable-dynamic-vram --preview-method auto --cache-ram 4` |
| 扩散模型 | `minimax_h3_fl2va_turbo_Q4_K_M.gguf`（GGUF，经 `UnetLoaderGGUFAdvanced` 加载，`dequant_dtype=float16, patch_dtype=float16`） |
| 加速 LoRA | `minimax_h3_turbo_v4_step600_ema_pruned_comfyui.safetensors` @ strength 1.0 |
| 采样 | `SamplerCustomAdvanced` + `BasicScheduler(beta, steps=4, denoise=1)` |
| 文本编码器 | Qwen3VL-8B-Instruct-Q4_K_M.gguf（BooguTE，7587 MB）+ ClipProj v3 |
| 工作负载 | ① 0.3 MP × 10 s = **23845 token**　② 0.3 MP × 15 s = **34706 token**　③ 0.4 MP × 10 s = **31699 token**　④ 30813 / 16242 token（早期实验）|

> 注：`--disable-dynamic-vram` 会让 `comfy.memory_management.aimdo_enabled` 保持 `False`，因此 0.35 新增的 Comfy Compiler / malloc graph / dynamic VBAR prefetch **在本测试中全部未启用**。

---

## 2. 结论一：H3-V100 本体在 0.35 上正常

每次节点执行时，全部补丁均成功挂载，无异常、无回退：

```
MiniMax H3 V100 mixed precision active: 50 blocks patched, 0 block wrappers refreshed, 0 attention patches reused, v100_only=True.
H3 bounded-memory token-wise MLP active: blocks=50, adaptive=True, chunk_tokens=640, cache_trim=True
H3 validated FP16 linear path active: MLP fc1/fc2 use FP16 with FP32 SwiGLU and fc2 scale=256.
H3 adaptive QKV token chunking armed: threshold=38000 tokens, chunk_tokens=1024
Workflow-scoped V100 attention dispatcher active: mode=flash_attn, ... sol_allowed=False, sol_min_tokens=4096
H3 runtime guard active: prefetch_dynamic_vbars=False, adaptive_memory=True, synchronous_cast_streams=True
H3 V100 Optimize active: attention_backend=flash_attn, mixed_precision=True, adaptive_memory=True, ...
```

本节点依赖的 `optimized_attention_override` 钩子在 v0.33.0 与 v0.35.0 中**均存在**（`comfy/ldm/modules/attention.py:187`），`prefetch_dynamic_vbars` / `NUM_STREAMS` / `comfy.model_prefetch` 亦均在 0.35 中存在。

### 0.33 → 0.35 期间 `comfy/ldm/minimax/model.py` 的变更（+247 / −57）

| 变更 | 影响 |
|---|---|
| `FinalLayer.forward(x, t_emb, video_seg, audio_seg, **sigma, sample_sigmas, shifts**)` | **新增 3 个必填参数** —— 任何内嵌 0.33 版 H3 前向的第三方包都会崩 |
| `MiniMaxH3Model.forward/_forward(..., denoise_mask, audio_denoise_mask)` | 新增 per-token 潜空间掩码路径（本测试工作流未喂 mask，不触发） |
| `Attention(..., gate_compress)` + `to_gate_compress` 模块 | 新增 VSA gate 线性层（dense 前向不使用） |
| `TokenRefiner.forward(..., attention=None)`、模型内 `attention=args.get("attention")` | 新的 attention 传参路径 |
| 移除 `v = v.clone()`（"Remove now unnecessary minimax memory workaround"） | 内存行为变化 |

> **风险提示**：由于 `FinalLayer` 是向前向结束时必走的算子，任何"复制了 0.33 版 H3 前向"的第三方包在 0.35 上都会在**第一步末尾**崩溃，且报错位置与本节点无关（堆栈里会夹着 `h3_prefetch_guard.py`，容易误判为本节点的问题）。见第 4 节。

---

## 3. 结论二：`ModelAttentionBackend` 会静默覆盖本节点的注意力后端（−23%）

### 机制

`comfy_extras/nodes_model_advanced.py` 的 **`ModelAttentionBackend`** 节点：

```python
attention_function = comfy.ldm.modules.attention.get_attention_function(attention_name, None)
if attention_function is None:
    logging.warning("Attention backend '%s' is unavailable; using PyTorch attention.", attention)
    attention_function = comfy.ldm.modules.attention.get_attention_function("pytorch")
m = model.clone()
m.set_model_optimized_attention(attention_function)          # ← 写入同一个 key
```

而 `comfy/model_patcher.py`：

```python
def set_model_optimized_attention(self, optimized_attention):
    def optimized_attention_override(_, *args, **kwargs):
        return optimized_attention(*args, **kwargs)
    self.model_options["transformer_options"]["optimized_attention_override"] = optimized_attention_override
```

**这正是本节点安装 V100 flash attention 调度器所用的同一个 key**（`flash_attention.py:312`）。

**关键点：节点在模型链上的位置决定胜负，后写的赢。** 实测工作流中的连线为：

```
UnetLoaderGGUFAdvanced → TESpeedMiniMaxH3 → H3V100Optimize → LoraLoaderModelOnly
                        → ModelAttentionBackend → BasicGuider / BasicScheduler
```

`ModelAttentionBackend` 是链上**最后一个**模型补丁节点，输出直连 guider/scheduler → **它的 override 完全生效，本节点的 flash attention 被顶掉**。

### 为什么它必然是 PyTorch

该节点的下拉项 `comfy kitchen attention` 依赖 `COMFY_KITCHEN_INT8_ATTENTION_IS_AVAILABLE`，即 INT8 tensor core。**V100 是 SM70，没有 INT8 tensor core，此选项永远不可用** → 每次都回退 `attention_pytorch`（SDPA），只在日志留一行 WARNING。

### 实测代价（受控对照）

| 对照 | token | 段档 | 装载状态 | attention | s/it | 采样 |
|---|---|---|---|---|---|---|
| A | 31699 | 3×10752 | 7576.99 / 3576.49 offloaded（**完全一致**） | **PyTorch** | **91.8** | 367s |
| A | 31699 | 3×10752 | 同上 | V100 flash | **68.9** | 275s |
| B | 23845 | 5→6 | 9392.19 / 1761.30 offloaded（一致） | **PyTorch** | **55.3** | 221s |
| B | 23845 | 6 | 9392 级 | V100 flash | **43.8** | 175s |

- 惩罚量 **Δ = 22.9 s/it**（31699 token）与 **Δ = 11.5 s/it**（23845 token），**符合 N² 比例**（22.9 × (23845/31699)² = 13.0 ≈ 11.5），与"这是注意力开销"一致。
- 相对惩罚 **+33% / +26%**。

**处置：bypass 或删除该节点。**

---

## 4. 结论三：第三方闭源包 `TE-Speed-MiniMaxH3` 与 0.35 不兼容（必崩）

### 现象

```
[INFO]  TE-Speed-MiniMaxH3: acceleration 0.0%
[ERROR] !!! Exception during processing !!!
TypeError: FinalLayer.forward() missing 3 required positional arguments: 'sigma', 'sample_sigmas', and 'shifts'
```

两种注意力模式（`sol_attn` 与 `flash_attn`）**均在第一步末尾（约 42–47 s 处）以完全相同的堆栈崩溃**，证明**与注意力后端无关**。

### 根因

崩溃堆栈中的 `nodes._SamplingScope.__call__` / `nodes._make_runtime_forward.runtime_forward` / `nodes._patched_forward`，其 `nodes.py` **在 `custom_nodes` 与 ComfyUI 核心中均不存在源码**。原因：该包为**编译后的 `.pyd`**：

```
custom_nodes/TE-Speed-MiniMaxH3/
├── __init__.py        (251 B, 仅 `from .nodes import TESpeedMiniMaxH3`)
└── nodes.pyd          (396 KB, 闭源编译模块)
```

二进制取证（ASCII 字符串扫描）：

```
_make_runtime_forward ✓   _patched_forward ✓   _SamplingScope ✓
final_layer ✓   t_emb ✓   video_seg ✓   audio_seg ✓   sigma_shift_video ✓
_MiniMaxH3Cache ✓   accumulated_delta ✓   acceleration ✓
```

即该包**内嵌了一份 MiniMax H3 模型 forward 的拷贝**，按 0.33 签名调用：

```python
self.final_layer(h, t_emb, video_seg, audio_seg)          # 4 参数（0.33）
```

而 0.35 的调用点（`comfy/ldm/minimax/model.py:778`）是：

```python
v, a = self.final_layer(h, t_emb, video_seg, audio_seg, sigma_v,
                        transformer_options.get("sample_sigmas"), (shift_v, shift_a))   # 7 参数
```

→ 只要它的运行时包装进入前向链，**第一步末尾必崩**。

### 触发条件（重要，具有迷惑性）

| 情形 | 模型链 | 包装是否激活 | 结果 |
|---|---|---|---|
| ComfyUI 缓存命中（未改任何参数） | 命中缓存 | **未激活**（日志无 `acceleration` 行） | 正常运行 |
| **任何缓存失效**（改任一 widget、重启后首次运行、改工作流） | 全链重打补丁 | **激活** | **必崩** |

这解释了一个常见误判："之前一直好好的，改了个参数就崩了"——并不是参数导致的，而是缓存失效让这个包装重新上车。**崩溃后缓存被"污染"，会连续复现。**

### 处置

- **bypass 或删除 `TESpeedMiniMaxH3` 节点**。它自报 `acceleration 0.0%`（在本配置下未产生任何加速），删除后实测 **s/it 无损失**（43.84 → 42.56，差异来自段档 6→4，约 1.3 s/it）。
- 它也会打印 `Audio query attention remains FP32; TE-Speed wrappers are preserved.`——本节点在设计上与其共存，但**不依赖它**。
- ⚠️ **不要**通过给 0.35 的 `FinalLayer.forward` 补默认参数来"兼容"：该包的前向整体是 0.33 版，同时缺少 0.35 的 per-token 掩码行、PDD head、`gate_compress`、新 attention 传参，**会静默算错**。

---

## 5. 结论四：`sol_attn` 在本安装上不可用（慢 2.55×）

### 现象

同样 23845 token / 同一 seed / 仅切换 `attention_backend`：

| attention_backend | 段档 | s/it | 采样 | 总时 |
|---|---|---|---|---|
| `flash_attn` | 4×6144 | **42.56** | 170s | 247.24s |
| `sol_attn` | 5×4864 | **108.25** | 433s | 509.74s |

**慢 2.55 倍**。段档差异（1 段）仅值 ≈0.8 s/it，无法解释 +65.7 s/it。

### 根因：原生 SM70 Sol kernel 从未被编译进 `.pyd`

四条证据闭合：

1. Sol 的调用要求一个原生算子（`sol_attention.py:514`）：
   ```python
   native = backend.try_native_sol_attention(q, k, v, tau=..., block_size=..., prefix_stop=...)
   ```
2. `backend.py` 中，算子不存在时**静默返回 `None`**：
   ```python
   if not hasattr(ops, "sol_prepare"):
       return None
   ```
3. **已安装 `.pyd` 的字符串扫描**：92,366 条字符串中含 `sol` 的 **0 条**（含 `attn` 的有数十条，证明扫描有效）。符号只有 flash attention（`flash_fwd_hdim128_sm70.cu`、`flash_fwd_hdim256_sm70.cu`、`compute_attn`）。
4. `native/setup.py` 只编译 4 个源：
   ```
   csrc/flash_attn/flash_api.cpp
   csrc/flash_attn/flash_api_torch_lib.cpp
   csrc/flash_attn/src/flash_fwd_hdim128_sm70.cu
   csrc/flash_attn/src/flash_fwd_hdim256_sm70.cu
   ```
   且在 `native/` 下所有 `.cu/.cuh/.cpp/.h/.py` 中搜索 `sol_prepare|sol_attn|SolAttn` → **零命中**。

→ `try_native_sol_attention` 必然返回 `None`，随后 `run_reference` 落入 **`sol_attention_reference`（纯 PyTorch 逐 query-block 循环）**，即 `sol_attention.py:539` 的兜底路径。**该回退不打任何日志**（代码仅在抛异常时 warning），所以控制台看不到任何提示。

代码本身也把这条路径当作兜底：

```python
if native is None:
    if int(q.shape[2]) >= 38_000:
        raise RuntimeError("Native Sol-Attn operators are unavailable for a long sequence; "
                           "the PyTorch reference path is intentionally disabled. ...")
```

### 澄清三个易混的数字

| 数字 | 含义 | 与 Sol 的关系 |
|---|---|---|
| `sol_min_tokens = 4096` | **Sol 的启用门槛** | 23845 / 31699 / 34706 token **远超，Sol 确实启用了** |
| `capacity_guard_tokens = 31744` | 单次**不分块** MLP 分配的实测上限（超出即强制分块，**不是长度上限**） | **与 Sol 无关** |
| `threshold = 38000 tokens` | **QKV 分块**阈值（超出走 1024-token 分块 QKV，更省显存但更慢） | **与 Sol 无关** |

**不要**为了"触发 Sol"去突破 32K —— Sol 的门槛是 **4096**，而 31744 / 38000 都是别的机制的门槛。

### 处置

- **本分支不要使用 `sol_attn`**，保持 `flash_attn`。
- **更正**：这不是"上游也没做"，而是**版本差**。上游 **v2.0.0 自带 `h3_v100_sol_cuda.cp312-win_amd64.pyd`**（原生 SM70 Sol kernel），并配套 `sol_native.py` / `fused_sol_speed.py` / `sol_calibration.py` / `sol_adaptive_policy.py` / `sol_adaptive_budget.py` / `sol_range.py`，且带 **`sol_minimum_gain_percent`（收益不达标就不启用）**——本次实测踩到的"Sol 反而更慢"正是它要防的问题。要做 Sol，正确路径是**升级/移植上游 v2.0 的原生实现**，而不是自研。
- 顺带发现：`native/csrc/flash_attn/src/` 下有 `flash_fwd_sparse_hdim128_sm70.cu` / `flash_fwd_sparse_kernel.h` 等**稀疏 flash 源码，但未列入 `setup.py` 编译**——未知是否完整，但方向比自研 Sol 现实得多。

---

## 6. 全部实测数据

| # | 时刻 | token | 段档 | attention | TE-Speed | s/it | 采样 | 总时 | 备注 |
|---|---|---|---|---|---|---|---|---|---|
| R1 | 19:49 | 16242 | 640×26 | V100 flash | 在链（未激活） | 55.38 | 240 s | 779 s | DiT **完全驻留** 11153 MB |
| R2 | 20:03 | 16242 | 640×26（推） | V100 flash | 在链（未激活） | 40.66 | 162 s | 508.7 s | |
| R3 | 20:17 | 30813 | 10496×3 | V100 flash | 在链（未激活） | 66.16 | 264 s | 401.4 s | 分辨率未记录 |
| R4 | 20:27 | 23845 | 4864×5 → 4096×6 | **PyTorch（203）** | 在链（未激活） | 55.34 | 221 s | 439.4 s | 运行中降档 |
| R5 | 20:39 | 31699 | 10752×3 | **PyTorch（203）** | 在链（未激活） | 91.81 | 367 s | 593.9 s | 0.4 MP 对照（带 203） |
| **R6** | 20:46 | 31699 | 10752×3 | V100 flash | 在链（未激活） | **68.90** | 275 s | 387.3 s | ← **0.4 MP×10 s 基线** |
| R7 | 20:57 | 23845 | 4096×6（推） | V100 flash | 在链（未激活） | 43.84 | 175 s | 346.0 s | 跳 203 |
| C1 | 21:01 | 23845 | 4864×5 → 3584×7 | `sol_attn` | **激活** | — | — | **崩** 56.7 s | `FinalLayer` 签名 |
| C2 | 21:10 | 23845 | 6144×4 → 4864×5 | V100 flash | **激活** | — | — | **崩** 56.0 s | 同上（证明与 sol 无关） |
| **R8** | 21:17 | 23845 | **6144×4** | V100 flash | **bypass** | **42.56** | **170 s** | **247.2 s** | ← **全项目最优** |
| S1 | 21:28 | 23845 | 4864×5 | `sol_attn` | bypass | 108.25 | 433 s | 509.7 s | Sol 走纯 PyTorch 兜底 |
| **R9** | 21:47 | **34706** | 11776×3 | V100 flash | bypass | **77.31** | 308 s | 515.5 s | **0.3 MP×15 s**，越过 31744 |
| F1 | — | 21181 | 2304×10 → 1536×14 | V100 flash | 在链（未激活） | 42.77 | 171 s | 356.2 s | **0.5 MP×5 s（改动前对照）** |
| D1 | 22:10 | 23845 | 640×38 | V100 flash | bypass | — | — | **崩** 105.6 s | 开 DynamicVRAM → `aimdo memory compile error` |
| P1 | 22:26 | 23845 | 6144×4 | V100 flash | bypass | 49.03 | 196 s | 436.3 s | 分段补丁版 · 冷启动 + Manager 抓取 |
| P2 | 22:35 | 23845 | 6144×4 | V100 flash | bypass | 49.01 | 196 s | 275.3 s | 分段补丁版 · Manager 抓取中 |
| P3 | 22:42 | 21181 | **4352×5** | V100 flash | bypass | 46.62 | 187 s | 381.5 s | **0.5 MP×5 s（改动后）** |
| T1 | 22:57 | 23845 | 6144×4 | V100 flash | bypass | 48.47 | 193 s | 434.0 s | `--reserve-vram 1.0` · Manager 抓取中 |
| **T2** | 23:03 | 23845 | **8192×3 → 6144×4** | V100 flash | bypass | **44.56** | **178 s** | **254.1 s** | `--reserve-vram 1.0` · **Manager 已空闲** |

### 6.0 按配置汇总

| 分辨率 × 时长 | token | 运行数 | 段档范围 | **最优 s/it** / 最差 | 采样 | 总时 |
|---|---|---|---|---|---|---|
| **0.3 MP × 10 s** | 23845 | 7 | 6144×4 ~ 640×38 | **42.56** / 49.03 | 170–196 s | 247–436 s |
| **0.5 MP × 5 s** | 21181 | 2 | 4352×5 / 1536×14 | 42.77 / 46.62 | 171–187 s | 356–382 s |
| **0.4 MP × 10 s** | 31699 | 2 | 10752×3 | **68.90** / 91.81 | 275–367 s | 387–594 s |
| **0.3 MP × 15 s** | 34706 | 1 | 11776×3 | **77.31** | 308 s | 515 s |
| （分辨率未记录） | 30813 / 16242 | 3 | 10496×3 / 640×26 | 40.66 / 66.16 | 162–264 s | 401–779 s |

> 上表"总时"含 TE/VAE 装载与编码、VAE 解码，**不可直接横向比较**（实测 `got prompt → DiT 装载` 在 **0.24 s 与 ~99 s** 之间波动）。**判定一律以 s/it 为准**。

> R9 是本报告唯一触发 `selection_reason=balanced_measured_full_limit` 的运行（34706 token 已越过 `capacity_guard_tokens=31744`），说明**该阈值只强制分块，不阻止运行**。

> R9 的完整时间构成：pre-load（VAE+TE 装载与编码，因时长改变而上游重算）**99.4 s** → DiT 装载 13.4 s → **采样 308 s（60%）** → 解码 92.4 s。

### 6.1 段档（MLP 分块）行为观察

段档由**首个 MLP 调用那一刻**的 `free + allocator 可复用缓存` 决定：

| 运行 | driver_free | reusable_cache | effective | budget_tokens | 段档 |
|---|---|---|---|---|---|
| R1 | 1994.8 MiB | 138.5 MiB | 2133.4 | **1** | 640 × **26** |
| R4 (step0) | 865.1 | 2442.0 | 3307.2 | 5490 | 4864 × 5 |
| R4 (step1) | **2946.7** | **145.7** | 3092.4 | 4327 | 4096 × **6** |
| R5/R6 | 917.3 | 3656.9 | 4574.1 | 12355 | 10496/10752 × 3 |
| R8 | 3493.6 | 147.5 | 3641.2 | 7300 | 6144 × 4 |

两个已核实的机制：

1. **`_trim_if_needed` 会清掉被计入额度的缓存**（`tokenwise_chunking.py:262` 的 `torch.cuda.synchronize()` + `empty_cache()`）。R4 中 step0→step1：**空闲显存从 865 升到 2947 MiB（+240%），段档却从 5 降到 6** —— 唯一变量是缓存被清掉（2442 → 145.7 MiB）。
2. **降档不可逆**：`_UPGRADE_STABLE_CALLS=3` 且要求 `budget_tokens >= selected + 1024`，而 `selected` 本身贴着 `budget_tokens` 取整 → 升级门槛结构性难以满足（日志中 `upgrade_candidate=None, upgrade_stable_calls=0/3` 恒定）。

#### ⚠️ 段档与耗时的因果关系：两次改口，最终结论在此

| 阶段 | 当时的假说 | 后续验证 |
|---|---|---|
| 早期 | "3 段 → 41 s/it"，段档决定 s/it | ❌ **错配**：41 s/it 那对属于 R2（26 段），不是 3 段那次 |
| 中期 | "崩档（≥10 段）代价可达 20%" | ❌ 用 R6/R8 拟合的模型把**其它效应**（低 eff / Manager 污染）误算进了残差，又归因给段数 |
| **最终** | **段档不是成本驱动因素** | ✅ 由下面这个受控实验证实 |

**受控实验**：把 `safety_reserve` 改成随可用显存缩放（1638 → 1044~1456 MiB），使 **0.5 MP×5 s** 的段数大幅下降：

| 0.5 MP×5 s（21181 token） | 改动前 | 改动后 |
|---|---|---|
| effective_budget | 2748 MiB | 2610 MiB |
| safety_reserve | 1638.4 | **1044.1** |
| **段档** | **2304×10 → 2048×11 → 1536×14** | **4352 × 5** |
| **s/it** | **42.77** | **46.62（+9%）** |
| 总时 | 356.2 s | 381.5 s |

**段数少了 60%，每步反而慢了 9%**；而这 9% 可由"当时 eff 低 138 MiB"完全解释——把 eff 归一化后，改前/改后落在同一条线上。**该补丁已回滚**（`tokenwise_chunking.py` 恢复原状，插件目录与 git 仓库逐字节一致）。

**仍然成立的机制**（保留，供排查参考）：

1. **`_trim_if_needed` 会清掉被计入额度的缓存**（`tokenwise_chunking.py:262` 的 `torch.cuda.synchronize()` + `empty_cache()`）。R4 step0→step1：空闲显存 865 → 2947 MiB（+240%），段档却从 5 降到 6 —— 唯一变量是缓存被清掉（2442 → 145.7 MiB）。
2. **eff 抖动会引起段档抖动**：T2 中 eff 4062 → 3723 → 3691，段档随之 8192×3 → 6144×4。
3. **升级通道是"有条件可用"而非死的**：T2 第一块因上一轮留下 **2974 MiB** allocator 缓存使 `budget_tokens = 9582 ≥ 8192 + 1024`，触发了 `selection_reason=stable_runtime_budget_upgrade`（此前多轮都是 `upgrade_candidate=None`，所以我先前"升级通道结构性不可达"的说法过强）。

### 6.2 成本模型：token 与耗时的关系（可用于规划）

分辨率固定时，**token 与时长近似线性**：

```
token ≈ 2145 + 2172 × 秒        # 0.3 MP（10 s → 23865，15 s → 34725；实测 23845 / 34706 ✓）
token ≈ 2145 + 2955 × 秒        # 0.4 MP
```

**每步耗时是超线性的**。用两个干净的 V100 flash 运行点（R6：31699 → 68.90；R8：23845 → 42.56）拟合：

```
s/it ≈ 6.05e-4 · N  +  4.95e-8 · N²          （N = token，±5%）
```

| 配置 | token | 预测 s/it | 实测最优 | 偏差 | 采样 |
|---|---|---|---|---|---|
| 0.3 MP × 5 s | ~13005 | 17.9 | — | — | ~72 s |
| 0.5 MP × 5 s | 21181 | 35.0 | 42.77 | +22%（低 eff / 多段 / 污染） | 171 s |
| **0.3 MP × 10 s** | 23845 | 42.6 | **42.56** | **0%**（拟合点） | 170 s |
| **0.4 MP × 10 s** | 31699 | 68.9 | **68.90** | **0%**（拟合点） | 275 s |
| **0.3 MP × 15 s** | 34706 | 80.6 | **77.31** | −4.1% | 308 s |
| 0.3 MP × 20 s | ~45585 | 126 | — | — | ~504 s |

**注意力占比**（`c·N²` 项占 s/it 的比例）随长度单调上升：**~40%（13k）→ 55%（24k）→ 66%（35k）→ 70%（46k）**。

**两个可直接用于决策的结论：**

1. 序列越长，稀疏注意力（Sol）的潜在收益越大——本分支因缺原生 kernel 无法利用，上游 v2.0 的原生实现正是为此准备（见第 10 节）。
2. **15 s 是 0.3 MP 的性价比拐点**：时长 +50%（10 s → 15 s），采样 170 → 308 s（**1.81×**）。

> 二次项（`N²`）在 transformer 中只有注意力，因此可由拟合系数直接读出注意力占每步的比例；这也是判定"该不该上稀疏注意力"的依据。

#### 6.2.1 测试环境陷阱：ComfyUI-Manager 启动抓取

启动后约 7 分钟内，ComfyUI-Manager 会抓取 **185 条 registry 条目**，与采样**并发**：

```
22:49:07  启动
22:53:30  ─── 采样开始 ─────────────────── 22:56:43 采样结束
22:55:56  FETCH ComfyRegistry Data [DONE]      ← 抓取跑到采样 ~75% 才结束
```

| 运行 | Manager 状态 | s/it |
|---|---|---|
| T1（22:53） | 正在抓 185 条 | 48.47 |
| T2（22:59） | `All startup tasks have been completed.` | **44.56** |

**同一配置、同一进程，仅因抓取结束就差 8.1%。** → **性能测试必须等 Manager 启动任务完成后再跑**（或关闭其启动抓取）。这也解释了本报告中若干"430 s+"的总时。

#### 6.2.2 另一条纪律：重复测量必须改 seed

ComfyUI 按输入缓存整图执行结果，**输入全不变时直接返回上次结果**（实测 `Prompt executed in 0.07 seconds`，一次采样都没跑）。改 `RandomNoise` 的 seed 是最干净的重复测量方式：**token 数不变（工作量恒定）**，且 seed 位于噪声节点，**不会失效上游 TE/VAE 缓存**（`got prompt → DiT 装载` 仍为 ~0.3 s）。

---

## 7. 推荐配置（0.35 最优）

```
UnetLoaderGGUFAdvanced
  └─ LoraLoaderModelOnly            ← 加速 LoRA，保留
       └─ H3V100Optimize            ← 保留，attention_backend = flash_attn
            └─ BasicGuider / BasicScheduler（采样）
```

| 节点 | 处置 | 理由 |
|---|---|---|
| `UnetLoaderGGUFAdvanced` | 保留 | GGUF 加载 |
| `LoraLoaderModelOnly` | **保留** | 4 步加速 LoRA；MODEL 类节点，与 TE/attention 无关 |
| `H3V100Optimize` | **保留** | **注意力正主**（安装 `optimized_attention_override`）+ 混合精度 + MLP 分块 + runtime guard |
| `ModelAttentionBackend` | **bypass / 删除** | 第二个 attention 覆盖者 → 每步 +20~25% |
| `TESpeedMiniMaxH3` | **bypass / 删除** | 闭源、0% 加速、与 0.35 必崩 |
| 其它 attention patcher（`H3DistanceAttentionPatcher`、SolAttn 包等） | **不要同时接** | 写同一个 key，互相覆盖 |
| `attention_backend = sol_attn` | **不要选** | 无原生 kernel，慢 2.55× |

**原则：注意力与文本编码，每类只保留一个"能工作"的提供者。**
- **注意力**：只留 `H3V100Optimize` → `flash_attn`
- **文本编码**：**不留任何包装节点**（本机唯一的候选 TE-Speed 不可用）

> 注意：在 UI 里 bypass 只对当前会话有效，**需要保存工作流**，否则重新打开会复原。

---

## 8. 复现与判定口径

1. **必须固定 seed**（`RandomNoise.control_after_generate = fixed`），否则任何画质/输出对比都无意义。
2. **只看 `s/it`**，不要看总时——总时会被上游缓存命中与否污染（0.35 s ↔ 98 s）。
3. 每次记录四项：
   - `V100 adaptive MLP memory:` 行的 `tokens` / `selected_chunk_tokens` / `chunks`
   - 进度条末行的 `NN.NNs/it`
   - `Prompt executed in N seconds`
   - 有无 `TE-Speed-MiniMaxH3: acceleration` / `Attention backend ... is unavailable` 等异常行
4. 跨条件对比前**先确认 `chunks` 一致**，否则段档差异会混入结论。

---

## 9. 待办 / 未解决

| 项 | 说明 |
|---|---|
| ~~段档决策抖动~~ | **已结案**：段档不是成本驱动因素（见 §6.1），改动已回滚 |
| ~~GGUF + DynamicVRAM~~ | **已结案**：loader 使 `is_dynamic()` 恒为 `False`；且 DynamicVRAM 与本节点冲突必崩（见 §10.2） |
| ~~INT8 / FP8 格式~~ | **已结案**：V100 无 INT8 tensor core（实测仅 FP16 的 0.63×），FP8 不可用 → GGUF 才是最优（见 §12） |
| ~~SolAttn-V100 稀疏路线~~ | **已结案**：短序列慢 5×，且与本节点 attention 槽位互斥（见 §13） |
| 稀疏 flash 源码（本分支） | `flash_fwd_sparse_*` 在源码树中但未列入 `setup.py`，完整性未知（不影响使用） |
| 非采样耗时 | 解码 62–92 s、装载与编码 0.24–99 s，与采样同量级 |
| R1 残差 / T2 偏差 | R1 **+32 s/it**、T2 **+4.7%**，均未定位 |
| 上游 v2.0 迁移 | 整体升级受 DynamicVRAM 限制不可行；如需 Sol，只能在长序列（S ≥ 50k）上重新评估独立插件 |

---

## 10. 与上游 v2.0.0 的对照与可借鉴项

> 来源：`rwashy/H3-V100` tag `v2.0.0`（`RELEASE_NOTES.md` 与源码）。本分支基线为 v1.3.0。

### 10.1 上游 v2.0 的变化

| 更新 | 对应文件 |
|---|---|
| 双卡 V100 | `dual_attention.py` `dual_gpu_plan.py` `dual_runtime.py` |
| FP8 E4M3 scaled + INT8 ConvRot | `weight_profile.py` |
| **SOL 升级**（Quality / Speed / Ultra / Manual） | `sol_native.py` `fused_sol_speed.py` `sol_calibration.py` `sol_adaptive_policy.py` `sol_adaptive_budget.py` `sol_range.py` |
| **集成 EasyCache**（Off / Quality / Speed） | `h3_easycache.py` |
| 双采 / latent 放大 / Sigma Refiner | `sampling_schedule.py` |
| **显存与稳定性**（阶段切换、长序列、连续运行） | `native_dynamic_vbar.py` `phase_allocation.py` `phase_model_release.py` `runtime_memory.py` `block_lifetime.py` `embedding_lifetime.py` |

v2.0 自带**四个**预编译扩展：`comfy_v100_flash_attn_cuda` / `h3_v100_mlp_cuda` / `h3_v100_qk_cuda` / **`h3_v100_sol_cuda`**；本分支只有第一个——这正是结论四的直接原因。

### 10.2 已验证：**GGUF 无法满足 `is_dynamic()`，且 DynamicVRAM 与本节点冲突**

> 本节先前给出过一个"10 分钟验证法"，**方向是错的**，已按实测更正。

**① 准入检查查的是 patcher 类型，不是文件格式**

```python
# 上游 v2.0 h3_optimize.py
is_dynamic = getattr(configured, 'is_dynamic', None)
if not callable(is_dynamic) or not bool(is_dynamic()):
    raise RuntimeError('H3 V100 requires ComfyUI DynamicVRAM. ...')
```

`comfy/model_patcher.py`：`ModelPatcher.is_dynamic()` → `False`（L402）；`ModelPatcherDynamic.is_dynamic()` → `True`（L1791）。

**② 但 GGUF loader 会把 patcher 无条件降级回普通类型**

`custom_nodes/comfyui-gguf-loader/nodes.py`：

```python
L37 : class GGUFModelPatcher(comfy.model_patcher.ModelPatcher)   # 普通子类，未重写 is_dynamic
L132: def clone(self, *args, **kwargs):
L133:     src_cls = self.__class__
L134:     self.__class__ = GGUFModelPatcher      # ← 无条件降级
L135:     n = super().clone(*args, **kwargs)
L136:     n.__class__ = GGUFModelPatcher         # ← 克隆体也降级
```

**→ 无论开不开 DynamicVRAM，GGUF 模型经这个 loader 之后 `is_dynamic()` 恒为 `False`。** 上游 v1.4+/v2.0 因此**无法直接用于 GGUF**——本分支的存在是必要的（README 结论正确，原因应表述为"loader 强制 class 替换"，而非启动参数）。

**③ 直接实验：DynamicVRAM + 本分支 = 必崩**

移除 `--disable-dynamic-vram` 与 `--lowvram` 后（启动日志出现 `DynamicVRAM support detected and enabled`、`Set vram state to: NORMAL_VRAM`）：

| 指标 | `--disable-dynamic-vram` | **DynamicVRAM 开启** |
|---|---|---|
| aimdo | 未加载 | `aimdo_setup_hooks: installing 6 hooks` |
| **可用显存 eff** | 3641 MiB | **820 MiB**（TE 改为常驻 `cuda:0`、VAE 被 2677 MB staged） |
| 段档 | 6144 × 4 | **640 × 38** |
| 结果 | 正常 | **第一步崩**：`RuntimeError: aimdo memory compile error` |

崩溃链：`comfy/model_prefetch.py:147` → `comfy_aimdo/malloc_graph.py:58` → `malloc_graph_pop`；前置信号 `Comfy model compiler graph breaks: 0, rogues: 19`。

**机制**：0.35 的 Comfy Compiler / malloc graph 需要**记录并重放固定的分配序列**，而本节点的自适应 MLP 分块**每块的分配尺寸都在变** → 图无法编译。本节点的 runtime guard 只关闭 `prefetch_dynamic_vbars` / `NUM_STREAMS`，**管不到 malloc graph**（它由 `aimdo_enabled` 独立开启）。

**结论**：`--lowvram` + `--disable-dynamic-vram` **不是可选项**——它们同时把 Comfy Compiler 关掉了，而这与本节点的工作方式互斥。上游 v2.0 之所以必须绑定 DynamicVRAM，是因为它**为那套栈重写**（原生 VBAR 流式，而非自适应分块）；两者不可混用。

### 10.3 真正值得借鉴的架构变化

| | v1.3（本分支） | v2.0 |
|---|---|---|
| 显存策略 | 经验性 chunk 档位选择 + `cache_trim` | **`NativeDynamicVBARPolicy` + `comfy_aimdo.model_vbar`**，原生动态 VBAR 流式权重 |
| 本次实测到的症状 | 段档 3↔26 抖动、`trim` 反致降档、升级棘轮死锁、offload 量决定速度 | 阶段化资源协调（`phase_allocation` / `phase_model_release` / `runtime_memory` / 张量生命周期） |

**"段档抖动"的正解方向在这里**：不是继续调 `_select_chunk_tokens` 的阈值（整个段档杠杆上限仅 ≈4%，见 6.1），而是把权重流式交给 0.35 原生的 VBAR。

### 10.4 可单独移植的模块（若 10.2 验证不通过）

| 模块 | 借鉴价值 | GGUF 可行性 |
|---|---|---|
| `h3_easycache.py` | 跨步缓存（Off/Quality/Speed，音视频分别保护） | **与权重格式无关，建议先试** |
| `cast_failure_cleanup.py`、`lora_failure_guard.py` | cast / LoRA 失败后的清理与保护 | 独立模块，很可能可直接搬 |
| `bounded_norm.py` | 数值有界 norm（防 NaN/Inf） | 独立，可搬 |
| `sampling_schedule.py` | 8 步 Turbo LoRA 使用已验证的 Euler | 零成本借鉴 |
| `mlp_native.py` / `qk_native.py` + 对应 `.pyd` | 原生 MLP / QK 算子 | 需 v2.0 Python 侧配套 |
| `sol_*` 全套 + `h3_v100_sol_cuda.pyd` | 原生稀疏注意力（本分支最缺的一块） | 需整体搬迁，不宜拆 |
| `dual_*`、FP8 / ConvRot | 单卡 + Q4_K_M 用不上 | ❌ |

**不建议**直接把 v2.0 合入本分支：它深度依赖 dynamic VBAR（而当前启动参数正把它关闭），等同于重写内存层。

---

## 11. 结论

### 11.1 这次"0.35 升级后变慢"到底是什么

**不是 ComfyUI 0.35 的问题，也不是本节点的问题。** 实测 21 次运行，本节点全部补丁正常挂载、未出现一次自身故障。变慢由三件事造成：

| # | 真因 | 代价 | 处置 |
|---|---|---|---|
| 1 | 工作流里的 `ModelAttentionBackend`（ComfyUI 内置节点）静默覆盖本节点的注意力调度器 | 每步 **+20~25%** | **bypass / 删除** |
| 2 | 工作流里的第三方闭源包 `TE-Speed-MiniMaxH3` 内嵌 0.33 版 H3 前向 | **必崩**（2 次），且自报 **0% 加速** | **bypass / 删除** |
| 3 | ComfyUI-Manager 启动后 ~7 分钟的后台抓取与采样并发 | s/it 虚高 **~8%** | 等 `All startup tasks have been completed.` 再跑 |

### 11.2 明确排除的（含两次自我更正）

| 假说 | 否证 |
|---|---|
| ❌ **段档 / MLP 分块数是成本驱动因素** | 把段数从 10–14 降到 5，s/it 反而 **+9%**（42.77 → 46.62）；补丁已回滚 |
| ❌ **可用显存 eff 是成本驱动因素** | `--reserve-vram 1.0` 使 eff **+534 MiB**（机制确认），s/it 无改善（42.56 → 44.56） |
| ❌ **`sol_attn` 没效果是因为序列不够长** | 门槛是 4096 token（23845/31699/34706 均远超）；实测走纯 PyTorch 兜底，**慢 2.55×** |
| ❌ **开 DynamicVRAM 对 GGUF 更好** | 与本节点自适应分块 + Comfy Compiler **冲突必崩**；且 GGUF loader 使 `is_dynamic()` 恒为 `False` |

### 11.3 0.35 + GGUF + V100 的推荐配置

```
UnetLoaderGGUFAdvanced
  └─ LoraLoaderModelOnly          ← 保留（加速 LoRA，与 TE/attention 无关）
       └─ H3V100Optimize          ← 保留，attention_backend = flash_attn
            └─ guider / scheduler
```

- **启动参数**：`--lowvram` + `--disable-dynamic-vram` **必须保留**（它们同时关掉了与本节点互斥的 Comfy Compiler）
- **不要接**：`ModelAttentionBackend`、`TESpeedMiniMaxH3`、任何第三方 attention patcher（写同一个 key，互相覆盖）
- **不要选**：`attention_backend = sol_attn`（本 build 无原生 kernel）
- **测试纪律**：等 Manager 启动任务完成后再跑；重复测量改 seed；判定只看 `s/it`

**实测水平**（4 步、GGUF Q4_K_M、4-step LoRA、V100 16 G）：

| 0.3 MP × 10 s | 0.3 MP × 15 s | 0.4 MP × 10 s | 0.5 MP × 5 s |
|---|---|---|---|
| **42.56 s/it** / 采样 170 s | **77.31 s/it** / 308 s | **68.90 s/it** / 275 s | 42.77 s/it / 171 s |

### 11.4 仍未解决

| 项 | 状态 |
|---|---|
| R1（16242 token、26 段、eff 2133）残差 **+32 s/it** | 与段数、eff 都对不上，未定位 |
| R8（42.56）与今天干净复测 T2（44.56）相差 **+4.7%** | 疑为温度/噪声量级，未验证 |
| 稀疏注意力（Sol） | 本 build 无原生 kernel（§5）；本机另有带原生 SM70 kernel 的独立插件，已验证并**关闭** → 见 §13 |
| `flash_fwd_sparse_*` | 源码在 `native/csrc` 中但未列入 `setup.py`，完整性未知 |
| 非采样耗时 | 解码 62–92 s、装载与编码 0.24–99 s，与采样同量级，未优化 |
| ~~INT8 / FP8 模型格式~~ | **已评估：V100 上更慢或不可用** → 见 §12 |
| ~~SolAttn-V100 稀疏路线~~ | **已评估：短序列不划算 + 槽位互斥** → 见 §13 |

---

## 12. 为什么 V100 上 GGUF 是最优格式（格式对比实测）

ComfyUI 0.35 启动日志会建议使用原生格式：

> *"ComfyUI native formats like **fp8, int8 and w4a8** will be faster even if they are larger than your memory."*

同一段还附了前提：*"If you are on nvidia **20 series and above** it is required that you update your pytorch to cu130"*。**这个建议对 V100（SM70）不成立**，原因在硬件。

### 12.1 硬件能力对照

| 能力 | V100 (SM70) | Turing+ (SM75) | Ada+ (SM89) |
|---|---|---|---|
| FP16 tensor core | ✅ 125 TFLOPS（峰值） | ✅ | ✅ |
| **INT8 tensor core** | ❌ **没有**（INT8 TC 自 Turing 引入） | ✅ | ✅ |
| FP8 | ❌ | ❌ | ✅ |

本机实测（`comfy.model_management`）：

```
supports_int8_compute : True      # 可运行，但走 DP4A，不是 tensor core
supports_fp8_compute  : False     # FP8 不可用
supports_nvfp4_compute: False
supports_mxfp8_compute: False
torch._int_mm         : 可用（返回 int32）
```

### 12.2 GEMM 基准（H3 真实 fc1 形状 `[4096, 5376] × [5376, 28672]`，单次 1262.7 GFLOP）

| 精度 | 耗时 | 吞吐 | 相对 FP16 |
|---|---|---|---|
| **FP16（tensor core）** | 14.03 ms | **90.0 TFLOPS** | 1.00× |
| **INT8（`torch._int_mm`）** | 22.45 ms | **56.2 TOPS** | **0.63×（慢 1.6 倍）** |

> FP16 已达 V100 峰值的 **72%** —— 说明本节点的 fp16 路径本身也已接近该硬件的实用上限（这也是"重写 kernel 现实收益只有 1.3–1.8×"的实证）。

### 12.3 同一层的权重体积

| 格式 | 体积 | 相对 |
|---|---|---|
| **Q4_K_M（~4.5 bit）** | **82.7 MB** | **1.00×** |
| INT8 / FP8 | 147.0 MB | 1.78× |
| FP16 | 294.0 MB | 3.56× |

### 12.4 结论

| 判断 | 依据 |
|---|---|
| **不要为 V100 换 INT8 / FP8 格式** | INT8 算力仅 0.63×，权重还大 1.78×；FP8 `supports_fp8_compute=False` 直接不可用 |
| **GGUF 唯一的代价（在线反量化）实测仅 ~3.5%** | 见 `V100优化方案.md` 的实测结论 |
| **GGUF Q4_K_M + 本节点的 fp16 混合精度 = 本机最优组合** | 体积最小 + 算力最高 + 反量化代价极小 |
| INT8 会因体积变大而**加重显存压力** | §6.1 已验证显存状态主导波动 |
| 上游 v2.0 的 `dual_gpu` 要求 FP8 / INT8-ConvRot 核心 | V100 上前者不可用，后者按基准更慢 |

**一句话**："INT8/FP8 比 GGUF 快"只在 **Turing 及以后**成立。V100 没有 INT8 tensor core，所以在这台卡上 **GGUF Q4_K_M 不是妥协，而是最优解**；只有升级到 SM75+/SM89+ 时，原生格式 + DynamicVRAM 才会全面超越 GGUF —— 届时本分支也就可以退场。

---

## 13. 已评估并关闭：SolAttn-V100 稀疏注意力路线

### 13.1 起因

本节点（v1.3.0 build）的 `sol_attn` 模式**没有原生 kernel**（见 §5）：`.pyd` 中 `sol_prepare` 命中 **0** 次 → 永远回退纯 PyTorch 参考实现 → 实测慢 2.55×。

但本机另有一个独立插件 **`ComfyUI-MiniMaxH3-SolAttn-V100` v1.2.2**（作者 aaalll12322），它**自带预编译的 SM70 稀疏 kernel**：

```
comfy_v100_solattn_cuda.cp312-win_amd64.pyd            974,848 B
native/csrc/flash_attn/src/flash_fwd_sparse_hdim128_sm70.cu
native/csrc/flash_attn/src/flash_fwd_sparse_kernel.h   (39 KB)
```

实测验证：`torch.ops.load_library(...)` → `torch.ops.comfy_v100_solattn_cuda.varlen_fwd_sparse` **存在 ✓**。
但其 API 与本节点**完全不同**（本节点需要 `comfy_v100_flash_attn_cuda.sol_prepare`）→ 二者无法互相调用。

### 13.2 实测结果（0.3 MP × 5 s，S = 12984，密度 37.2%）

| 配置 | token | s/it | 按 §6.2 成本模型的合理水平 | 倍数 |
|---|---|---|---|---|
| **本节点（flash，纯 dense）** | 23845 | **42.56** | 42.6 | **1.00×** |
| **SolAttn-V100（稀疏 + FP16Safe）** | 12984 | **80.92** | 16.2 | **≈5.0× 慢** |

单位 token 耗时：**40.1 ms vs 10.4 ms（3.9 倍）**。

### 13.3 三条原因 + 一条硬性限制

1. **它用 FP16Safe 替代了本节点更省的混合精度**：`x/16` prescale + 熔断 + **fp32 重跑兜底**（触发即整层 4×）；本节点是 `fc1/fc2 FP16 + FP32 SwiGLU + fc2 scale=256`，设计上不发生。
2. **它没有 MLP token 分块**：16 G 上极易进入"装得越满越慢"的状态 —— 同配置两次运行仅因装载状态不同，s/it 从 80.92 变成 **166.71（2 倍）**。
3. **作者的基准在 S≈98512**（比本测试长 7.6 倍）：attention 是 N²，短序列上该栈的固定开销占比过高。
4. **两者 attention 槽位硬性互斥**：`nodes.py:301-303` 检测到已有 `optimized_attention_override` 会直接 `raise` → **无法实现"本节点管 fp16 + 显存、SolAttn 只管 attention"的叠加**。

### 13.4 处置

**在当前配置（0.3 MP 级短序列 + GGUF + 16 G V100）下关闭此路线。**

唯一值得将来重回的场景：**S ≥ 50000 token 的长序列**（attention 占比升到 ~70%+），且需先解决"无 MLP 分块"带来的显存问题。

> 另注：上游 v2.0 自带 `h3_v100_sol_cuda.pyd` 与 `sol_minimum_gain_percent`（收益不达标不启用），但整体迁移受 DynamicVRAM 限制（见 §10.2）。

---

## 附录 A：关键代码位置（ComfyUI 0.35.0）

| 文件:行 | 内容 |
|---|---|
| `comfy/ldm/minimax/model.py:312` | `FinalLayer.forward(..., sigma, sample_sigmas, shifts)` 新签名 |
| `comfy/ldm/minimax/model.py:778` | 7 参数调用点 |
| `comfy_extras/nodes_model_advanced.py:402-405` | `ModelAttentionBackend` 后端不可用时回退 PyTorch |
| `comfy/model_patcher.py:688-694` | `set_model_optimized_attention` 写入 `optimized_attention_override` |
| `comfy/ldm/modules/attention.py:187` | `optimized_attention_override` 消费点（0.33/0.35 相同） |
| `custom_nodes/H3_V100/flash_attention.py:312` | 本节点安装 override 的位置 |
| `custom_nodes/H3_V100/h3_optimize.py:16,163` | `SOL_MIN_TOKENS = 4096`、`allow_sol = (attention_backend == MODE_SOL)` |
| `custom_nodes/H3_V100/sol_attention.py:514,532-543` | native 尝试 → 纯 PyTorch 参考实现兜底 |
| `custom_nodes/H3_V100/tokenwise_chunking.py:97-198` | 段档选择；`:262` trim；`:299-329` 棘轮 |
| `custom_nodes/H3_V100/native/setup.py:33-37` | 只编译 4 个 flash attention 源（**不含 Sol**） |
| 上游 v2.0 `h3_optimize.py`（tag `v2.0.0`） | `is_dynamic()` 准入检查与全部新参数（sol_* / easycache_* / dual_gpu） |
| 上游 v2.0 预编译扩展 | `comfy_v100_flash_attn_cuda` / `h3_v100_mlp_cuda` / `h3_v100_qk_cuda` / **`h3_v100_sol_cuda`** |

## 附录 B：证据命令

```powershell
# 确认 H3-V100 补丁全部挂载
Select-String -Path user\comfyui_8188.log -Pattern 'V100|adaptive MLP memory'

# 二进制取证：TE-Speed 包内嵌 H3 前向
$s=[Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes(
  'custom_nodes\TE-Speed-MiniMaxH3\nodes.pyd'))
'_make_runtime_forward','final_layer','video_seg' | ForEach-Object { "$_ : $($s.Contains($_))" }

# 二进制取证：本节点 .pyd 是否含 Sol 算子
$s=[Text.Encoding]::ASCII.GetString([IO.File]::ReadAllBytes(
  'custom_nodes\H3_V100\comfy_v100_flash_attn_cuda.cp312-win_amd64.pyd'))
"sol 出现次数: $(([regex]::Matches($s,'sol')).Count)"    # => 0

# 查看上游 v2.0 源码（部分克隆：只取元数据，按需拉单个文件）
New-Item -ItemType Directory temp_h3v100_v2 -Force | Out-Null
Set-Location temp_h3v100_v2
git init -q .
git remote add origin https://github.com/rwashy/H3-V100.git
git fetch --depth 1 --filter=blob:none --no-tags origin refs/tags/v2.0.0
git show FETCH_HEAD:RELEASE_NOTES.md
git ls-tree -r --name-only FETCH_HEAD | Select-String '\.pyd$'   # => 四个扩展

# INT8 vs FP16 GEMM 基准（§12 数据来源）—— H3 真实 fc1 形状
$py = @'
import torch, time
torch.backends.cuda.matmul.allow_tf32 = False
def bench(fn, iters=30):
    for _ in range(5): fn()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize(); return (time.perf_counter() - t0) / iters * 1000
N, K, M = 4096, 5376, 28672
x16 = torch.randn(N, K, dtype=torch.float16, device="cuda"); w16 = torch.randn(K, M, dtype=torch.float16, device="cuda")
x8  = torch.randint(-128, 127, (N, K), dtype=torch.int8, device="cuda")
w8  = torch.randint(-128, 127, (K, M), dtype=torch.int8, device="cuda")
print("FP16:", bench(lambda: torch.matmul(x16, w16)), "ms")   # ~14.0 ms / 90 TFLOPS
print("INT8:", bench(lambda: torch._int_mm(x8, w8)), "ms")    # ~22.5 ms / 56 TOPS
'@
$py | D:\mt-tool\ComfyUI\python_embeded\python.exe -

# SolAttn-V100 是否自带原生 SM70 kernel（§13 数据来源）
$py2 = @'
import torch, glob, os
p = r"D:\mt-tool\ComfyUI\ComfyUI\custom_nodes\ComfyUI-MiniMaxH3-SolAttn-V100"
torch.ops.load_library(sorted(glob.glob(os.path.join(p, "comfy_v100_solattn_cuda*")))[0])
print(hasattr(torch.ops.comfy_v100_solattn_cuda, "varlen_fwd_sparse"))   # => True
'@
$py2 | D:\mt-tool\ComfyUI\python_embeded\python.exe -
```
