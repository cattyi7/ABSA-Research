import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# 强制使用非交互式后端
import matplotlib
matplotlib.use('Agg')

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from transformers import AutoTokenizer

# ---------- 加载模型 ----------
def load_model(checkpoint_path, language="english"):
    from src.architectures.roberta_aspect import RobertaAspectModel
    model = RobertaAspectModel.create_model(language)
    state = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model

# ---------- 预处理 ----------
def preprocess_sample(text, aspect, tokenizer, max_len=256):
    encoded = tokenizer(text, aspect, max_length=max_len, truncation=True, return_tensors="pt")
    return encoded["input_ids"], encoded["attention_mask"]

# ---------- 获取 attention 权重 ----------
def get_attention_weights(model, input_ids, attention_mask):
    with torch.no_grad():
        logits, attn_weights, _ = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attn_weight=True
        )
    return attn_weights.squeeze(0).cpu().numpy()

# ---------- 画热力图 ----------
def plot_attention_heatmap(tokens, weights, title, save_path):
    plt.figure(figsize=(max(0.4 * len(tokens), 6), 3))
    sns.heatmap([weights], xticklabels=tokens, yticklabels=["Attn"],
                cmap="Reds", cbar=True, linewidths=0.5, linecolor="grey")
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")

# ---------- token 转字符串 ----------
def token_ids_to_tokens(input_ids, tokenizer):
    return tokenizer.convert_ids_to_tokens(input_ids.squeeze(0).tolist())

# ---------- 主流程 ----------
if __name__ == "__main__":
    # 手动定义 5 条代表性样本
    samples = [
        {"text": "The battery life is excellent and lasts all day.", "aspect": "battery life"},
        {"text": "The screen is too dim and hard to read outdoors.", "aspect": "screen"},
        {"text": "The keyboard feels cheap and mushy when typing.", "aspect": "keyboard"},
        {"text": "I love the lightweight design and fast processor.", "aspect": "design"},
        {"text": "The speakers produce clear and loud sound.", "aspect": "speakers"},
    ]

    print("Step 1: 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    print("Tokenizer 加载完成")

    print("Step 2: 加载 baseline 模型...")
    model_base = load_model("models/roberta_aspect_english_base_seed42.pth", "english")
    print("Baseline 模型加载完成")

    print("Step 3: 加载 rationale 模型...")
    model_rationale = load_model("models/roberta_aspect_english_rationale_seed42.pth", "english")
    print("Rationale 模型加载完成")

    output_dir = Path("output/attention_plots")
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(samples):
        text, aspect = sample["text"], sample["aspect"]
        print(f"\n>>> 处理样本 {idx}: {text[:30]}... | Aspect: {aspect}")

        # 编码
        input_ids, attn_mask = preprocess_sample(text, aspect, tokenizer)
        tokens = token_ids_to_tokens(input_ids, tokenizer)

        # 获取两个模型的注意力权重
        weights_base = get_attention_weights(model_base, input_ids, attn_mask)
        weights_rationale = get_attention_weights(model_rationale, input_ids, attn_mask)

        # 截断到实际长度（排除 padding）
        actual_len = attn_mask.sum().item()
        tokens = tokens[:actual_len]
        weights_base = weights_base[:actual_len]
        weights_rationale = weights_rationale[:actual_len]

        # 保存对比图
        plot_attention_heatmap(
            tokens, weights_base,
            title=f"Baseline Attention\nAspect: {aspect}",
            save_path=output_dir / f"sample_{idx}_base.png"
        )
        plot_attention_heatmap(
            tokens, weights_rationale,
            title=f"Rationale Attention\nAspect: {aspect}",
            save_path=output_dir / f"sample_{idx}_rationale.png"
        )
        print(f"  Done: {idx}")

    print("\n🎉 全部完成！图片保存在 output/attention_plots/")