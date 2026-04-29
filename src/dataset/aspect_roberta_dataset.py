# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset
from utils.rationale_mask import create_rationale_mask


class   AspectRobertaDataset(Dataset):
    """
    RoBERTa 专用 Aspect 级情感分析数据集
    自动处理：文本 + Aspect 句子对
    与你的 RobertaAspectTrainer 完全对齐
    """
    def __init__(self, data, tokenizer_wrapper, max_len=256):
        self.data = data
        self.tokenizer = tokenizer_wrapper.tokenizer  # 直接用现有的分词器
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        aspect = item["aspect"]
        label = item["label"]
        rationale = item.get("rationale", "")

        # 核心：BERT/RoBERTa 原生句子对 → 完美支持 ABSA
        inputs = self.tokenizer(
            text,
            aspect,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # 压缩多余的 batch 维度
        input_ids = inputs["input_ids"].squeeze(0)
        attention_mask = inputs["attention_mask"].squeeze(0)

        rationale_mask = create_rationale_mask(
        self.tokenizer, text, rationale, self.max_len)


        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": torch.tensor(label, dtype=torch.long),
            "rationale_mask": rationale_mask
        }