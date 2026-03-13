"""
简单语言预测模型示例
===================

本示例演示如何利用该项目的 Mamba 模型实现一个字符级（character-level）语言预测模型。

整体流程：
1. 准备训练文本，构建字符级词表（vocab）
2. 配置并实例化 MambaLMHeadModel（小规模，可在 CPU 上运行）
3. 训练模型（最小化交叉熵损失）
4. 使用训练好的模型进行文本生成（预测下一个字符）

运行要求：
    pip install mamba-ssm   # 需要 NVIDIA GPU 及 CUDA 11.6+
    pip install einops

或直接从仓库根目录安装：
    pip install -e .
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel


# ---------------------------------------------------------------------------
# 1. 字符级分词器（Character-level Tokenizer）
# ---------------------------------------------------------------------------

class CharTokenizer:
    """将文本映射到整数 token（字符级）。"""

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.vocab = chars
        self.vocab_size = len(chars)
        self._stoi = {ch: i for i, ch in enumerate(chars)}
        self._itos = {i: ch for i, ch in enumerate(chars)}

    def encode(self, text: str) -> list[int]:
        """将文本转为 token id 列表。text 中的字符必须全部出现在训练词表中。"""
        unknown = set(text) - set(self._stoi)
        if unknown:
            raise ValueError(
                f"以下字符不在词表中，请确认 prompt 仅包含训练文本中出现的字符：{sorted(unknown)}"
            )
        return [self._stoi[ch] for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(self._itos[i] for i in ids)


# ---------------------------------------------------------------------------
# 2. 构建小规模 Mamba 语言模型
# ---------------------------------------------------------------------------

def build_model(vocab_size: int, device: torch.device) -> MambaLMHeadModel:
    """
    创建一个小规模的 MambaLMHeadModel。

    配置说明：
    - d_model=64       : 模型隐层维度
    - n_layer=2        : Mamba 块的数量
    - d_intermediate=0 : 不使用额外的 MLP 层（节省参数）
    - rms_norm=False   : 使用标准 LayerNorm（无需 Triton 内核）
    - fused_add_norm=False : 不使用融合算子（兼容 CPU 及无 Triton 环境）
    - ssm_cfg={'use_fast_path': False} : 关闭 CUDA 快速路径（兼容纯 PyTorch 模式）
    """
    config = MambaConfig(
        d_model=64,
        n_layer=2,
        d_intermediate=0,
        vocab_size=vocab_size,
        ssm_cfg={"use_fast_path": False},
        rms_norm=False,
        residual_in_fp32=False,
        fused_add_norm=False,
        pad_vocab_size_multiple=1,
        tie_embeddings=True,
    )
    model = MambaLMHeadModel(config, device=device)
    return model


# ---------------------------------------------------------------------------
# 3. 训练循环
# ---------------------------------------------------------------------------

def train(
    model: MambaLMHeadModel,
    token_ids: torch.Tensor,
    seq_len: int = 32,
    batch_size: int = 16,
    num_steps: int = 500,
    lr: float = 1e-3,
    device: torch.device = torch.device("cpu"),
) -> None:
    """
    在给定 token 序列上训练语言模型（下一个 token 预测）。

    参数：
        token_ids  : 全部训练 token（1D LongTensor）
        seq_len    : 每条训练样本的上下文长度
        batch_size : 批大小
        num_steps  : 训练步数
        lr         : 学习率
        device     : 训练设备
    """
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    data_len = len(token_ids)

    for step in range(1, num_steps + 1):
        # 随机采样 batch_size 个起始位置，使用高级索引批量构造训练对
        starts = torch.randint(0, data_len - seq_len, (batch_size,))
        idx = starts.unsqueeze(1) + torch.arange(seq_len)  # (B, L)
        x = token_ids[idx].to(device)                       # 输入 (B, L)
        y = token_ids[idx + 1].to(device)                   # 目标 (B, L)

        # 前向传播
        logits = model(x).logits          # (B, L, vocab_size)

        # 计算交叉熵损失
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),  # (B*L, vocab_size)
            y.view(-1),                         # (B*L,)
        )

        # 反向传播 & 参数更新
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % 100 == 0 or step == 1:
            print(f"step {step:4d}/{num_steps} | loss = {loss.item():.4f}")


# ---------------------------------------------------------------------------
# 4. 文本生成（语言预测）
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate(
    model: MambaLMHeadModel,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    device: torch.device = torch.device("cpu"),
) -> str:
    """
    给定提示文本（prompt），逐个字符地预测并生成后续内容。

    参数：
        model          : 训练好的 MambaLMHeadModel
        tokenizer      : 字符级分词器
        prompt         : 提示文本
        max_new_tokens : 最多生成的新 token 数
        temperature    : 采样温度（越高越随机，越低越确定）
        device         : 推理设备

    返回：
        包含 prompt 及生成内容的完整字符串
    """
    model.eval()

    ids = tokenizer.encode(prompt)
    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)  # (1, L)

    generated = list(ids)

    for _ in range(max_new_tokens):
        # 每一步将当前全部 token 送入模型（教学示例；生产环境应使用 KV-cache）
        logits = model(input_ids).logits  # (1, L, vocab_size)
        next_logits = logits[0, -1, :]    # 取最后一个位置的 logits

        # 温度缩放后采样
        probs = F.softmax(next_logits / temperature, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)
        input_ids = torch.cat(
            [input_ids, torch.tensor([[next_id]], dtype=torch.long, device=device)],
            dim=1,
        )

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# 5. 主程序
# ---------------------------------------------------------------------------

TRAINING_TEXT = """
To be, or not to be, that is the question:
Whether 'tis nobler in the mind to suffer
The slings and arrows of outrageous fortune,
Or to take arms against a sea of troubles
And by opposing end them. To die—to sleep,
No more; and by a sleep to say we end
The heartache and the thousand natural shocks
That flesh is heir to: 'tis a consummation
Devoutly to be wish'd. To die, to sleep;
To sleep, perchance to dream—ay, there's the rub:
For in that sleep of death what dreams may come,
When we have shuffled off this mortal coil,
Must give us pause.
"""


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备：{device}\n")

    # 1. 构建分词器
    tokenizer = CharTokenizer(TRAINING_TEXT)
    print(f"词表大小（字符数）：{tokenizer.vocab_size}")
    print(f"词表内容：{''.join(tokenizer.vocab)}\n")

    # 2. 编码训练文本
    token_ids = torch.tensor(tokenizer.encode(TRAINING_TEXT), dtype=torch.long)
    print(f"训练 token 总数：{len(token_ids)}\n")

    # 3. 构建模型
    model = build_model(tokenizer.vocab_size, device=device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量：{num_params:,}\n")

    # 4. 训练
    print("开始训练……")
    train(
        model,
        token_ids,
        seq_len=32,
        batch_size=16,
        num_steps=500,
        lr=1e-3,
        device=device,
    )
    print("\n训练完成！\n")

    # 5. 文本生成
    prompt = "To be, or not to be"
    print(f"提示文本（prompt）：{prompt!r}\n")
    result = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_new_tokens=200,
        temperature=0.8,
        device=device,
    )
    print("生成结果：")
    print(result)


if __name__ == "__main__":
    main()
