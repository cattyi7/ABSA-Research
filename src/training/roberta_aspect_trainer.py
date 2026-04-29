# -*- coding: utf-8 -*-
"""
RoBERTa Aspect 级情感分类 训练器
完整可运行，与项目结构完全统一
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path
import os

from transformers import  get_cosine_schedule_with_warmup
from ..architectures.roberta_aspect import RobertaAspectModel, RobertaAspectTokenizerWrapper
from ..utils.config import Config
from src.losses.focal_loss import FocalLoss

class RobertaAspectTrainer:
    def __init__(self, language="english", num_classes=2, lr=8e-6, patience=4):
        self.language = language
        self.lr = lr
        self.patience = patience
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_type = "roberta_aspect"
        self.num_classes = num_classes

        # 模型 & 分词器
        self.model = RobertaAspectModel.create_model(language=language).to(self.device)
        model_name = "hfl/chinese-roberta-wwm-ext" if language == "chinese" else "roberta-base"
        self.tokenizer_wrapper = RobertaAspectTokenizerWrapper(model_name)
        
        # 损失函数
        self.criterion = FocalLoss(alpha=1, gamma=2)

        # 训练历史
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rates": []
        }

        print(f"✅ 初始化 Aspect-RoBERTa 训练器完成 | 设备: {self.device}")

    def train_epoch(self, train_loader, optimizer, scheduler):
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        USE_RATIONALE_ATTN = False  # 是否使用 rationale attention

        for batch in tqdm(train_loader, desc="训练中"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            rationale_mask = batch["rationale_mask"].to(self.device)

            optimizer.zero_grad()
            logits, attn_weights = self.model(
                input_ids, 
                attention_mask, 
                output_attn_weight=True  # 打开开关
            )
            # 1. 原始分类损失
            cls_loss = self.criterion(logits, labels)

            # 2. 【核心新增】Rationale 注意力监督损失
            attn_loss = 0.0
            if USE_RATIONALE_ATTN:
                # 直接监督 AspectAttention！
                attn_loss = -torch.mean(attn_weights * rationale_mask)

                # 3. 总损失（0.1 是稳定系数）
            total_loss_batch = cls_loss + 0.1 * attn_loss

            # 反向传播（用总损失）
            total_loss_batch.backward()

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) #梯度裁剪
            optimizer.step() #优化器更新权重
            scheduler.step() #学习率调度器更新学习率

            total_loss += total_loss_batch.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(train_loader)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def val_epoch(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="验证中"):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                logits = self.model(input_ids, attention_mask)
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        avg_loss = total_loss / len(val_loader)
        acc = accuracy_score(all_labels, all_preds)
        return avg_loss, acc

    def train(self, train_loader, val_loader, epochs=8, lr=8e-6, save_best=True):
        print(f"\n🚀 开始训练 Aspect-RoBERTa，共 {epochs} 轮 | 学习率: {lr}")

        # 优化器
        no_decay = ["bias", "LayerNorm.weight","lora"] # 不进行权重衰减的参数
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": 0.01,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.lr, eps=1e-8)

        # 学习率调度
        total_steps = len(train_loader) * epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        best_val_acc = 0.0
        patience = self.patience
        patience_counter = 0

        for epoch in range(epochs):
            print(f"\n========== Epoch {epoch + 1}/{epochs} ==========")
            train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
            val_loss, val_acc = self.val_epoch(val_loader)

            # 记录
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["learning_rates"].append(scheduler.get_last_lr()[0])

            print(f"📈 训练结果:")
            print(f"   - 训练损失: {train_loss:.4f}")
            print(f"   - 训练准确率: {train_acc:.4f} ({train_acc*100:.2f}%)")
            print(f"   - 验证损失: {val_loss:.4f}")
            print(f"   - 验证准确率: {val_acc:.4f} ({val_acc*100:.2f}%)")
            print(f"当前学习率: {scheduler.get_last_lr()[0]:.2e}")

            # 最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                if save_best:
                    self.save_model()
                patience_counter = 0
                print(f"✅ 最优模型更新 | 最佳验证精度: {best_val_acc:.4f}")
            else:
                patience_counter += 1
                print(f"⚠️ 早停计数: {patience_counter}/{patience}")
                if patience_counter >= patience:
                    print("🛑 触发早停，停止训练")
                    break

        # 训练结束
        print(f"\n🏁 训练完成！最佳验证精度: {best_val_acc:.4f}")
        
        return {"best_val_acc": best_val_acc, "history": self.history}

    def save_model(self):
        model_path = Config.get_model_path('roberta_aspect', self.language)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "language": self.language,
            "history": self.history,
            "model_type": self.model_type
        }
        torch.save(checkpoint, model_path)
        print(f"模型已保存到: {model_path}")
        return model_path



    def evaluate(self, test_loader):
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                logits = self.model(input_ids, attention_mask)
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        print("\n测试集评估结果:")
        print(classification_report(all_labels, all_preds, target_names=["负面", "正面"], digits=4))
       
        return accuracy_score(all_labels, all_preds)


    
    
    

    