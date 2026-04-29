# ABSA Research: RoBERTa + RAG + Rationale
> 细粒度情感分析研究项目 | Aspect-Based Sentiment Analysis

---

## 📌 项目简介
本项目面向**细粒度情感分析（Aspect-Based Sentiment Analysis, ABSA）**任务，基于 SemEval 公开数据集构建深度学习模型。在 RoBERTa 基础模型之上，实现了 **RAG 案例检索增强**与 **Rationale 关键词监督**，旨在提升模型的准确率、泛化能力与可解释性。

本项目专注于**算法研究与实验对比**，提供多模型（RoBERTa/BERT/BiLSTM/TextCNN）的统一训练、评估与对比框架，适合科研与学习场景。

---

## ✨ 核心创新点
1.  **RAG 案例检索增强**
    构建训练数据案例库，为低置信度样本引入相似案例检索，辅助模型决策，提升泛化能力。

2.  **Rationale 情感关键词监督**
    引入情感关键词（Rationale）监督信号，增强模型对关键文本的注意力权重，提升判别精度与可解释性。

3.  **多模型统一实验框架**
    支持一次性训练、评估、对比多种模型，自动记录 Accuracy、F1 等关键指标，便于性能分析。

---

## 🚀 快速开始
### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 配置 API Key（可选，仅使用 LLM 增强功能时需要）
在项目根目录新建 `.env` 文件：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 训练模型
```bash
python run_ABSA.py
```

### 4. 推理预测
```bash
python predict_demo.py
```

---

## 📊 实验结果
| 模型配置 | 测试集 Accuracy | 测试集 F1-Score |
| :--- | :---: | :---: |
| RoBERTa 基础模型 | - | - |
| RoBERTa + RAG | - | - |
| RoBERTa + RAG + Rationale | - | - |

---

## 📁 项目结构
```
.
├── src/                  # 核心研究代码
│   ├── architectures/   # 多模型基础架构实现
│   ├── dataset/          # 数据集加载与预处理
│   ├── losses/           # 损失函数（含 Focal Loss）
│   ├── models/           # 改进模型模块（Aspect Attention 等）
│   ├── scripts/          # 数据下载、加载脚本
│   ├── training/         # 训练、评估核心逻辑
│   └── utils/            # RAG检索、日志、工具函数
├── experiments/          # 实验指标JSON归档
├── run_ABSA.py          # 训练入口脚本
├── predict.py           # 推理预测脚本
├── requirements.txt     # 环境依赖配置
├── .gitignore           # 敏感/缓存文件忽略
└── README.md            # 项目说明文档
```

---

## 📚 数据集
本项目使用公开标准数据集：
- SemEval 2014 / 2015 / 2016 ABSA
- 餐厅领域、笔记本电脑领域标注数据

---

## 🔧 使用说明
本项目仅供科研与学习使用。

---

## 📄 License
MIT License