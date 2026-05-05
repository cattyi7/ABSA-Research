"""
独立脚本：搜索最优 rationale_lambda
用法：python search_lambda.py
"""
import sys
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ================= 路径设置 =================
sys.path.append("src")
sys.path.append(".")

from src.dataset.aspect_roberta_dataset import AspectRobertaDataset
from src.training.roberta_aspect_trainer import RobertaAspectTrainer
from datasets import Dataset

# ================= 固定参数 =================
LANGUAGE = "english"
BATCH_SIZE = 16
EPOCHS = 6
MAX_SEQ_LEN = 256
LR = 1.2e-5
PATIENCE = 2
SEED = 42

# ================= 数据加载函数（复用你的预处理逻辑）=================
def load_rationale_data():
    """加载含 rationale 的训练集和普通验证集"""
    # 训练集（含 rationale）
    rationale_path = "datasets/semeval_split/train_with_rationales.csv"
    if not os.path.exists(rationale_path):
        raise FileNotFoundError(f"未找到 {rationale_path}，请先运行 generate_rationales.py")
    
    rationale_df = pd.read_csv(rationale_path)
    rationale_df = rationale_df.rename(columns={
        "Sentence": "text",
        "Aspect Term": "aspect",
        "polarity": "label"
    })
    rationale_df["label"] = rationale_df["label"].map({"positive": 1, "negative": 0})
    rationale_df = rationale_df.dropna(subset=["text", "aspect", "label", "rationale"])
    rationale_df["label"] = rationale_df["label"].astype(int)
    train_dataset = Dataset.from_pandas(
        rationale_df[["text", "aspect", "label", "rationale"]],
        preserve_index=False
    )
    
    # 验证集（普通，不需要 rationale）
    val_df = pd.read_csv("datasets/semeval_split/val_original.csv")
    val_df = val_df.rename(columns={
        "Sentence": "text",
        "Aspect Term": "aspect",
        "polarity": "label"
    })
    val_df = val_df.dropna(subset=["text", "aspect", "label"])
    val_df["label"] = val_df["label"].map({"positive": 1, "negative": 0})
    val_df = val_df.dropna(subset=["label"])
    val_df["label"] = val_df["label"].astype(int)
    val_dataset = Dataset.from_pandas(
        val_df[["text", "aspect", "label"]],
        preserve_index=False
    )
    
    return train_dataset, val_dataset

# ================= 设置随机种子 =================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================= 主搜索函数 =================
def search_lambda():
    print("=" * 60)
    print("  Rationale Lambda 网格搜索")
    print("=" * 60)
    
    set_seed(SEED)
    
    # 加载数据
    print("\n加载数据...")
    train_dataset, val_dataset = load_rationale_data()
    print(f"训练集: {len(train_dataset)} | 验证集: {len(val_dataset)}")
    
    # 创建训练器（先创建一个临时实例，获取 tokenizer）
    print("\n初始化...")
    temp_trainer = RobertaAspectTrainer(
        language=LANGUAGE, lr=LR, patience=PATIENCE,
        use_rationale=True, use_gfm=False, rationale_lambda=0.1
    )
    
    tokenizer = temp_trainer.tokenizer_wrapper
    
    # 构建验证 DataLoader（复用同一个）
    val_ds = AspectRobertaDataset(val_dataset.to_list(), tokenizer, MAX_SEQ_LEN)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 待测试的 lambda 列表
    lambdas_to_test = [0.05, 0.1, 0.2, 0.5, 1.0]
    results = {}
    
    for lam in lambdas_to_test:
        print(f"\n{'='*50}")
        print(f"  测试 rationale_lambda = {lam}")
        print(f"{'='*50}")
        
        set_seed(SEED)  # 每个 lambda 用相同种子
        
        # 创建新的训练器
        trainer = RobertaAspectTrainer(
            language=LANGUAGE, lr=LR, patience=PATIENCE,
            use_rationale=True, use_gfm=False,
            rationale_lambda=lam
        )
        
        
        # 构建训练 DataLoader
        train_ds = AspectRobertaDataset(train_dataset.to_list(), tokenizer, MAX_SEQ_LEN)
        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        
        # 训练
        trainer.train(train_loader, val_loader, epochs=EPOCHS)
        
        # 记录最佳验证准确率
        best_val_acc = max(trainer.history["val_acc"])
        results[lam] = best_val_acc
        print(f"  ✅ λ={lam:.2f}  最佳验证 Acc: {best_val_acc:.4f}")
    
    # ================= 输出结果 =================
    print("\n\n" + "=" * 60)
    print("  Lambda 搜索结果（按验证 Acc 降序）")
    print("=" * 60)
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    for lam, acc in sorted_results:
        print(f"  λ={lam:.2f}  →  Val Acc = {acc:.4f}")
    
    best_lam, best_acc = sorted_results[0]
    print(f"\n  ✅ 最佳 rationale_lambda = {best_lam}")
    print(f"  对应验证 Acc = {best_acc:.4f}")
    print(f"\n  📝 请在 run_ABSA.py 中设置 USE_RATIONALE = True，" +
          f"并在 get_trainer 中传入 rationale_lambda={best_lam}")
    
    return best_lam, results

# ================= 入口 =================
if __name__ == "__main__":
    search_lambda()