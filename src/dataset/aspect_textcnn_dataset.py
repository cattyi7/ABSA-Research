# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class AspectTextCNNDataset(Dataset):
    """
    TextCNN 专用 Aspect 数据集
    自动拼接：句子 + [SEP] + Aspect
    与原模型、训练器完全兼容
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

        # 核心：拼接文本与维度
        combined = f"{text} [SEP] {aspect}"

        # 分词 + 转 ID
        tokens = combined.lower().split()
        input_ids = [self.vocab.get(t, 1) for t in tokens]

        # 长度统一
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
        else:
            input_ids += [0] * (self.max_len - len(input_ids))

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(label, dtype=torch.long)
        }