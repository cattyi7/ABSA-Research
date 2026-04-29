# -*- coding: utf-8 -*-
"""
RoBERTa模型训练器
用途：专门负责RoBERTa模型的训练、验证和保存
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
from typing import Dict, List, Tuple, Any
from pathlib import Path
import json
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from ..utils.config import Config
from ..architectures.roberta import RobertaSentimentModel, RobertaTokenizerWrapper


class RobertaTrainer:
    """
    RoBERTa训练器类
    与BertTrainer完全对齐
    """
    
    def __init__(self, language: str = "english"):
        self.language = language
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.tokenizer = None
        self.model_type = "roberta"  # 用于输出路径
        
        # 训练历史
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        print(f"初始化RoBERTa训练器 ({language}), 设备: {self.device}")
    
    def create_model(self) -> nn.Module:
        self.model = RobertaSentimentModel.create_model(self.language)
        self.model.to(self.device)
        
        model_name = "hfl/chinese-roberta-wwm-ext" if self.language == "chinese" else "roberta-base"
        self.tokenizer = RobertaTokenizerWrapper(model_name)
        
        self.criterion = nn.CrossEntropyLoss()
        print(f"RoBERTa模型创建完成")
        return self.model
    
    def setup_optimizer(self, train_loader: DataLoader, epochs: int, 
                       warmup_ratio: float = 0.1) -> None:
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        
        self.optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=2e-5, 
            eps=1e-8
        )
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        print(f"优化器设置完成，总步数: {total_steps}, 预热步数: {warmup_steps}")
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        progress_bar = tqdm(train_loader, desc="训练中")
        
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_predictions += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
            
            current_acc = correct_predictions / total_predictions
            current_lr = self.scheduler.get_last_lr()[0]
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Acc': f'{current_acc:.4f}',
                'LR': f'{current_lr:.2e}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = correct_predictions / total_predictions
        self.history['learning_rates'].append(self.scheduler.get_last_lr()[0])
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct_predictions = 0
        total_predictions = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = correct_predictions / total_predictions
        
        return avg_loss, accuracy
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 3, save_best: bool = True) -> Dict[str, Any]:
        print(f"开始训练RoBERTa模型，共{epochs}轮...")
        self.setup_optimizer(train_loader, epochs)
        
        best_val_acc = 0.0
        best_model_state = None
        patience_counter = 0
        patience = Config.TRAINING_CONFIG.get('early_stopping_patience', 2)
        
        for epoch in range(epochs):
            print(f"\n=== Epoch {epoch + 1}/{epochs} ===")
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            print(f"训练损失: {train_loss:.4f}, 训练准确率: {train_acc:.4f}")
            print(f"验证损失: {val_loss:.4f}, 验证准确率: {val_acc:.4f}")
            print(f"当前学习率: {self.scheduler.get_last_lr()[0]:.2e}")
            
            if save_best and val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                patience_counter = 0
                print(f"★ 新的最佳模型！验证准确率: {val_acc:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停触发！已连续{patience}轮无改善")
                    break
        
        if save_best and best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"\n恢复最佳模型权重，验证准确率: {best_val_acc:.4f}")
        
        model_path = self.save_model()
        
        return {
            'model_type': 'roberta',
            'language': self.language,
            'epochs': epoch + 1,
            'best_val_accuracy': best_val_acc,
            'final_train_accuracy': self.history['train_acc'][-1],
            'final_val_accuracy': self.history['val_acc'][-1],
            'model_path': str(model_path),
            'history': self.history,
            'early_stopped': patience_counter >= patience
        }
    
    def save_model(self) -> Path:
        model_path = Config.get_model_path('roberta', self.language)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'model_config': Config.MODEL_CONFIGS['roberta'],
            'language': self.language,
            'history': self.history,
            'tokenizer_name': "hfl/chinese-roberta-wwm-ext" if self.language == "chinese" else "roberta-base"
        }
        
        torch.save(checkpoint, model_path)
        print(f"模型已保存到: {model_path}")
        return model_path
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, Any]:
        self.model.eval()
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_predictions)
        report = classification_report(all_labels, all_predictions, output_dict=True)
        
        # 自动画混淆矩阵
        self.plot_confusion_matrix(all_labels, all_predictions)
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'total_samples': len(all_labels)
        }
        
        print(f"测试准确率: {accuracy:.4f}")
        return results

    # ===================== 可视化功能 =====================
    def plot_training_history(self, save_path=None):
        from ..utils.config import Config
        history = self.history
        epochs = range(1, len(history['train_loss']) + 1)

        if save_path is None:
            save_path = Config.MODEL_OUTPUT_PATHS[self.model_type] / "training_curve.png"

        plt.figure(figsize=(12, 5))
        plt.subplot(1,2,1)
        plt.plot(epochs, history['train_loss'], 'b-', label='Train Loss')
        plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
        plt.title('Loss Curve')
        plt.legend()
        plt.grid(True)

        plt.subplot(1,2,2)
        plt.plot(epochs, history['train_acc'], 'b-', label='Train Acc')
        plt.plot(epochs, history['val_acc'], 'r-', label='Val Acc')
        plt.title('Accuracy Curve')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"训练曲线已保存: {save_path}")

    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        from ..utils.config import Config
        if save_path is None:
            save_path = Config.MODEL_OUTPUT_PATHS[self.model_type] / "confusion_matrix.png"

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Negative","Positive"],
                    yticklabels=["Negative","Positive"])
        plt.title("Confusion Matrix")
        plt.ylabel("True")
        plt.xlabel("Pred")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"混淆矩阵已保存: {save_path}")