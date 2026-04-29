# -*- coding: utf-8 -*-
"""
RoBERTa Aspect 级情感分类模型（真正完整版本）
接口与项目中 BERT / RoBERTa 完全对齐，可直接替换训练
输入格式：句子 + [SEP] + 评价维度（Aspect）
"""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional
from peft import inject_adapter_in_model, LoraConfig

from src.models.aspect_attention import AspectAttention

class RobertaAspectModel(nn.Module):
    """
    Aspect-Based Sentiment Analysis (ABSA) 专用模型
    针对属性维度进行情感分类
    """

    def __init__(self, config: Dict[str, Any]):
        super(RobertaAspectModel, self).__init__()

        # 配置项
        self.model_name = config['model_name']
        self.num_classes = config.get('num_classes', 2)
        self.dropout_rate = config.get('dropout', 0.1)
        self.freeze_roberta = config.get('freeze_roberta', False)

        # 加载预训练模型
        self.roberta = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.roberta.config.hidden_size

        lora_config = LoraConfig(
            r=4,
            lora_alpha=8,
            target_modules=["query", "value"],  # 对RoBERTa有效
            lora_dropout=0.0,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        inject_adapter_in_model(lora_config, self.roberta)
        
        for param in self.roberta.parameters():
            param.requires_grad = True
        
        # 冻结主干
        if self.freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False

        # 分类头
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)

        # Aspect Attention
        self.aspect_attention = AspectAttention(self.hidden_size)

        # 初始化
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask=None, output_attn_weight=False):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        hidden_states = outputs.last_hidden_state  # [B,T,H]

        # ====== NEW: aspect embedding ======
        aspect_mask = (input_ids == 102).float()   # SEP token（RoBERTa通常是102）
        aspect_hidden = torch.sum(hidden_states * aspect_mask.unsqueeze(-1), dim=1)
        aspect_hidden = aspect_hidden / (aspect_mask.sum(dim=1, keepdim=True) + 1e-8)

        # ====== aspect-guided attention ======
        attended_output, attn_weights = self.aspect_attention(
            hidden_states,
            aspect_hidden,
            attention_mask
        )

        attended_output = self.dropout(attended_output)
        logits = self.classifier(attended_output)

        if output_attn_weight:
            return logits, attn_weights   
        return logits

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """预测概率"""
        with torch.no_grad(): #不用算梯度，节省时间
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
        return probabilities

    def get_embeddings(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor = None
    ) -> torch.Tensor:
        """获取句向量"""
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.pooler_output

    @classmethod
    def create_model(cls, language: str = "english", config_name: str = "roberta_aspect"):
        from ..utils.config import Config
        model_config = Config.MODEL_CONFIGS[config_name].copy()

        if language == "chinese":
            model_config['model_name'] = "hfl/chinese-roberta-wwm-ext"
        else:
            model_config['model_name'] = "roberta-base"

        return cls(model_config)


class RobertaAspectTokenizerWrapper:
    """
    与项目完全对齐的分词器包装
    包含 encode、decode 全套方法
    """

    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name

    def encode_texts(
        self,
        texts: list,
        max_length: int = 256,
        padding: bool = True,
        truncation: bool = True
    ) -> Dict[str, torch.Tensor]:
        """单句编码（兼容接口）"""
        return self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )

    def encode_sentence_aspect_pair(
        self,
        sentences: list,
        aspects: list,
        max_length: int = 256
    ) -> Dict[str, torch.Tensor]:
        """
        【核心】Aspect 级专用编码
        输入：句子 + 评价维度
        格式：[CLS] sentence [SEP] aspect [SEP]
        """
        return self.tokenizer(
            sentences,
            aspects,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def decode_tokens(self, token_ids: torch.Tensor) -> list:
        """解码：token → 文本"""
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)