# Examples

## simple_language_prediction.py

演示如何利用本项目的 Mamba 模型实现一个**字符级语言预测模型**。

### 功能说明

该脚本包含以下五个部分：

| 模块 | 说明 |
|------|------|
| `CharTokenizer` | 从训练文本自动构建字符级词表，并完成文本 ↔ token 的双向转换 |
| `build_model` | 使用 `MambaConfig` + `MambaLMHeadModel` 构建小规模 Mamba 语言模型 |
| `train` | 基于"下一个 token 预测"任务进行训练（交叉熵损失 + AdamW 优化器） |
| `generate` | 给定提示文本，逐字符自回归生成后续内容（支持温度采样） |
| `main` | 端到端运行示例（训练 → 生成） |

### 模型结构

```
输入 token ids
     ↓
Embedding（vocab_size → d_model=64）
     ↓
Mamba Block × 2   ← 选择性状态空间模型（SSM）核心
     ↓
LayerNorm
     ↓
LM Head（d_model=64 → vocab_size，与 Embedding 权重共享）
     ↓
输出 logits → softmax → 下一个字符的概率分布
```

### 运行方法

1. 安装依赖（需要 NVIDIA GPU 及 CUDA 11.6+）：

   ```bash
   pip install mamba-ssm einops
   ```

   或从本仓库源码安装：

   ```bash
   pip install -e .
   ```

2. 运行示例：

   ```bash
   python examples/simple_language_prediction.py
   ```

### 核心配置说明

```python
MambaConfig(
    d_model=64,            # 隐层维度（越大表达能力越强，训练越慢）
    n_layer=2,             # Mamba 块数量
    d_intermediate=0,      # 不使用额外 MLP 层
    vocab_size=...,        # 词表大小（由训练文本自动确定）
    ssm_cfg={"use_fast_path": False},  # 兼容 CPU 模式
    rms_norm=False,        # 使用标准 LayerNorm
    fused_add_norm=False,  # 关闭 Triton 融合算子
    tie_embeddings=True,   # 输入/输出 Embedding 权重共享（减少参数）
)
```
