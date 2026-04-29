# -*- coding: utf-8 -*-
"""
RoBERTa情感分析模型
用途：基于预训练RoBERTa模型的文本分类，SOTA基线模型
RoBERTa = 优化版BERT（更大数据、移除NSP、更长训练）
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, Any, Optional

class RobertaSentimentModel(nn.Module):
    """
    RoBERTa情感分析模型类
    用途：使用预训练RoBERTa模型进行文本情感分类
    与BERT代码结构完全一致，仅底层模型不同
    """
    
    def __init__(self, config: Dict[str, Any]):
        super(RobertaSentimentModel, self).__init__()
        
        self.model_name = config['model_name']
        self.num_classes = config.get('num_classes', 2)
        self.dropout_rate = config.get('dropout', 0.1)
        self.freeze_roberta = config.get('freeze_roberta', False)
        
        # 加载预训练RoBERTa
        self.roberta = AutoModel.from_pretrained(self.model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # 冻结参数
        if self.freeze_roberta:
            for param in self.roberta.parameters():
                param.requires_grad = False
        
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, self.num_classes)
        
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, 
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        # RoBERTa 不使用 token_type_ids
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def predict(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, 
                token_type_ids: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            logits = self.forward(input_ids, attention_mask)
            probabilities = torch.softmax(logits, dim=1)
            return probabilities
    
    def get_embeddings(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            return outputs.pooler_output
    
    @classmethod
    def create_model(cls, language: str = "engilsh", config_name: str = "roberta"):
        from ..utils.config import Config
        
        model_config = Config.MODEL_CONFIGS[config_name].copy()
        
        if language == "chinese":
            model_config['model_name'] = "hfl/chinese-roberta-wwm-ext"
        else:
            model_config['model_name'] = "roberta-base"
        
        return cls(model_config)

class RobertaTokenizerWrapper:
    def __init__(self, model_name: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model_name = model_name
    
    def encode_texts(self, texts: list, max_length: int = 512, 
                    padding: bool = True, truncation: bool = True) -> Dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            return_tensors="pt"
        )
        return encoded
    
    def decode_tokens(self, token_ids: torch.Tensor) -> list:
        return self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)