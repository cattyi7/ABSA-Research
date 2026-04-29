# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class AspectBiLSTMDataset(Dataset):
    """
    BiLSTM 专用 Aspect 级情感分析数据集
    自动拼接：文本 + [SEP] + Aspect
    与原有模型、训练器完全兼容
    """
    def __init__(self, data, vocab, max_len=128):
        self.data = data
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        aspect = item["aspect"]
        label = item["label"]

        # 核心：拼接文本与评价维度
        combined_text = f"{text} [SEP] {aspect}"

        # 分词（简单按空格分词）
        tokens = combined_text.lower().split()

        # 转为词汇表 ID，不存在则用 <unk> 对应 1
        input_ids = [self.vocab.get(token, 1) for token in tokens]

        # 统一长度
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids += [0] * (self.max_len - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }