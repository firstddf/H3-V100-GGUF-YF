# H3-V100 (GGUF-YF) 在 ComfyUI 0.35 上的兼容性与实测报告

> 测试日期：2026-09-10
> 测试者：firstddf
> 被测版本：本仓库（基于 rwashy/H3-V100 v1.3.0 的 GGUF 兼容分支）
> 相关文档：[`V100优化方案.md`](https://github.com/firstddf)（更早的 0.29/0.33 时期结论）

---

## 0. 摘要（TL;DR）

| 结论 | 判定 |
|---|---|
| **H3-V100 本体在 ComfyUI 0.35 上工作正常** | ✅ 全部补丁正常挂载，9 次运行通过 |
| **`sol_attn` 路线在本安装上不可用** | ❌ 原生 SM70 Sol kernel 从未被编译进 `.pyd`，静默回退纯 PyTorch 参考实现，**慢 2.55×** |
| **`ModelAttentionBackend`（ComfyUI 内置节点）会静默覆盖本节点的注意力后端** | ⚠️ 每步慢 **20–25%**，必须不要接 |
| **第三方闭源包 `TE-Speed-MiniMaxH3` 与 0.35 不兼容** | ❌ 内嵌 0.33 版 H3 前向，撞上 0.35 的 `FinalLayer` 签名变更 → **必崩** |
| **本节点在 0.35 上的实测最优** | 0.3MP×10s / 23845 token / 4 步：**42.56 s/it**，采样 170s，总时 247.24s |

**一句话**：0.35 升级后"变慢"的主因不是 H3-V100，也不是 0.35 本身，而是**工作流里多接了两个会互相覆盖/不兼容的节点**。拆掉后恢复到当前最优。

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
| 工作负载 | ① 0.3 MP × 10 s = **23845 token**　② 0.4 MP × 10 s = **31699 token**　③ 30813 / 16242 token（早期实验）|

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

### 澄清两个常见误解

| 数字 | 含义 | 与 Sol 的关系 |
|---|---|---|
| `sol_min_tokens = 4096` | **Sol 的启用门槛** | 23845 / 31699 token **远超，Sol 确实启用了** |
| `capacity_guard_tokens = 31744` | **MLP 分块**的实测 OOM 上限 | **与 Sol 无关** |

**不要**为了"触发 Sol"去突破 32K —— 门槛是 4096。

### 处置

- **不要使用 `sol_attn`**，保持 `flash_attn`。
- 想让 Sol 可用，需要从零实现 `sol_prepare` + 融合 forward 的 SM70 CUDA kernel（`backend.py` 引用的算子目前只是"预期存在"）。工作量以周计，暂不做。
- 顺带发现：`native/csrc/flash_attn/src/` 下有 `flash_fwd_sparse_hdim128_sm70.cu` / `flash_fwd_sparse_kernel.h` 等**稀疏 flash 源码，但未列入 `setup.py` 编译**——这是比 Sol 现实得多的方向，但未知是否完整。

---

## 6. 全部实测数据

| # | 时刻 | token | 段档 | attention | TE-Speed | s/it | 采样 | 总时 |
|---|---|---|---|---|---|---|---|---|
| R1 | 19:49 | 16242 | 26×640 | V100 flash | 在链（未激活） | 55.4 | 240s | 779s |
| R2 | 20:03 | 16242 | 26×640 | V100 flash | 在链（未激活） | 40.7 | 162s | 508.7s |
| R3 | 20:17 | 30813 | 3×10496 | V100 flash | 在链（未激活） | 66.2 | 264s | 401.4s |
| R4 | 20:27 | 23845 | 5→6 | **PyTorch（203）** | 在链（未激活） | 55.3 | 221s | 439.4s |
| R5 | 20:39 | 31699 | 3×10752 | **PyTorch（203）** | 在链（未激活） | 91.8 | 367s | 593.9s |
| R6 | 20:46 | 31699 | 3×10752 | V100 flash | 在链（未激活） | 68.9 | 275s | 387.3s |
| R7 | 20:57 | 23845 | 6×4096 | V100 flash | 在链（未激活） | 43.8 | 175s | 346.0s |
| C1 | 21:01 | 23845 | 5→7 | `sol_attn` | **激活** | — | — | **崩**（56.7s） |
| C2 | 21:10 | 23845 | 4→5 | V100 flash | **激活** | — | — | **崩**（56.0s） |
| **R8** | 21:17 | 23845 | **4×6144** | V100 flash | **bypass** | **42.6** | **170s** | **247.2s** |
| S1 | 21:20 | 23845 | 5×4864 | `sol_attn` | bypass | 108.3 | 433s | 509.7s |

> ⚠️ **总时不可直接横向比较**：它包含 llama-yf 提示词、TE/VAE 装载与解码，且会被 ComfyUI 缓存命中与否污染（实测 `got prompt → DiT 装载` 在 0.35 s 与 98 s 之间波动）。**判定一律以 s/it 为准。**

> R2 与 R7 相对同 token 的其它运行偏离较大（+27% / −20%），说明仍有未识别的变量（可能是 token 构成/工作流差异），仅作参考。

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

**但段档的时间代价很小**：R7（6 段）43.84 → R8（4 段）42.56，即 **≈0.6–0.8 s/it 每段**。在 42.5 s/it 的量级下，段档这一整个杠杆的上限只有 **≈4%**。因此**不建议**为此改动分块逻辑；仅需留意日志中的 `chunks=` 是否跳到两位数。

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
| 段档决策抖动 | 已知机制（trim + 棘轮），但收益上限仅 ≈4%，暂不处理 |
| `sol_attn` | 缺原生 SM70 kernel，不可用；需要重写 CUDA 实现 |
| 稀疏 flash 源码 | `flash_fwd_sparse_*` 在源码树中但未编译，待验证完整性 |
| 非采样耗时 | 实测解码 ~62 s、装载/前处理波动大（0.35–98 s），与采样同量级 |
| R2 / R7 离群 | 同 token 下偏离 20%+，未定位 |

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
| `custom_nodes/H3_V100/native/setup.py:33-37` | 只编译 4 个 flash attention 源 |

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
```
