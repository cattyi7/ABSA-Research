# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset

class AspectBertDataset(Dataset):
    """
    BERT 专用 Aspect 级情感分析数据集
    自动处理：文本 + Aspect 句子对输入
    完全兼容你的现有 BertTrainer
    """
    def __init__(self, data, tokenizer_wrapper, max_len=128):
        self.data = data
        self.tokenizer = tokenizer_wrapper.tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        aspect = item["aspect"]
        label = item["label"]

        #  BERT 原生句子对格式 → 直接做 ABSA
        inputs = self.tokenizer(
            text,
            aspect,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 去掉 batch 维度
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long)
        }