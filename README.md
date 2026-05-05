```markdown
# 方面级情感分析中对抗训练与注意力监督的交互效应研究

> **An Empirical Study on the Interaction between Adversarial Training and Attention Supervision for Aspect-Level Sentiment Analysis**

本项目是一个面向**方面级情感分析（Aspect-Based Sentiment Analysis, ABSA）** 的统一实验框架，基于 RoBERTa 预训练模型，结合 **LoRA 高效微调**与**方面注意力机制（Aspect Attention）**，系统探究 **FGM 对抗训练**与**大语言模型生成的理据（Rationale）注意力弱监督**在低资源场景下的交互效应。

实验在 **SemEval-2014 Task 4 Laptop** 数据集上完成，采用严格的按句子 ID 划分策略和多种子评估协议。框架同时支持 BERT、BiLSTM、TextCNN 作为对比基线。

---

## 📑 目录

- [主要特性](#主要特性)
- [核心实验结果](#核心实验结果)
- [关键发现](#关键发现)
- [环境依赖](#环境依赖)
- [项目结构](#项目结构)
- [数据准备](#数据准备)
- [快速开始](#快速开始)
- [实验配置说明](#实验配置说明)
- [可视化工具](#可视化工具)
- [致谢](#致谢)

---

## 主要特性

### 🔧 多模型支持
- **RoBERTa**（主力模型，配合 Aspect Attention + LoRA）
- **BERT**（预训练对比基线）
- **BiLSTM**（传统方法参照）
- **TextCNN**（传统方法参照）

所有模型通过统一的训练器接口调用，一键切换。

### 🧩 可插拔增强模块

| 模块 | 功能 | 开关变量 |
|------|------|----------|
| **FGM 对抗训练** | 嵌入层梯度扰动，提升鲁棒性 | `USE_FGM` |
| **Rationale 注意力监督** | LLM 生成关键词掩码，引导注意力聚焦 | `USE_RATIONALE` |
| **LLM 标签清洗** | 训练前用 DeepSeek 纠正错误标签 | `USE_TRAIN_LABEL_CLEAN` |
| **LLM 推理校正** | 低置信度样本触发大模型重判 | `USE_INFER_CORRECTION` |
| **数据增强** | LLM 改写训练样本扩充数据 | `USE_AUG` |

### 📊 严格的实验设计
- **按句子 ID 划分数据集**：杜绝同一句子不同方面词泄露到不同集合
- **多随机种子评估**：3 个种子（42, 123, 456），报告均值 ± 标准差
- **Bootstrap 置信区间**：1000 次有放回重采样，计算 95% CI
- **早停策略**：基于验证准确率，耐心值为 2

### 📈 可视化工具
- 训练损失/准确率曲线
- 混淆矩阵
- **注意力热力图对比**（Baseline vs Rationale）

### 🏗️ 模块化架构
- 模型架构、数据集类、训练器相互解耦
- 所有增强技术通过全局开关控制，支持一键消融实验
- 自动记录实验指标到 JSON，方便对比分析

---

## 核心实验结果

**数据集**：SemEval-2014 Task 4 Laptop（训练 1438 / 验证 180 / 测试 186）

**基础配置**：RoBERTa-base + LoRA（r=4, α=8）+ Aspect Attention + Focal Loss（γ=2）

| 配置 | Test Acc (%) | F1 (%) | 标准差 | 95% Bootstrap CI |
|------|-------------|--------|--------|-------------------|
| **Baseline** | 90.48 | 90.47 | ±0.26 | [88.10, 92.86] |
| **+ FGM** | **90.66** | **90.66** | **±0.00** | [88.28, 93.04] |
| **+ Rationale** (λ=0.5) | 90.48 | 90.47 | ±0.69 | [88.10, 93.04] |
| **+ FGM + Rationale** | **90.11** | 90.10 | ±0.45 | [87.55, 92.67] |

**其他骨干网络 Baseline 对比**：

| 模型 | Test Acc (%) |
|------|-------------|
| RoBERTa | 90.48 |
| BERT | 90.32 |
| TextCNN | 80.65 |
| BiLSTM | 77.96 |

---

## 关键发现

1. **FGM 是唯一带来测试集正向提升的增强技术**：+0.18%（90.48% → 90.66%），且三个种子标准差接近 0，展现出卓越的训练稳定性。同时，FGM 将训练-验证准确率差距从约 5% 缩减至约 2%，有效抑制了过拟合。

2. **Rationale 监督显著改善注意力可解释性**：注意力热力图清晰显示模型从“分散关注功能词”转变为“聚焦情感关键词”。验证准确率一度达到 94.19%，但该增益未泛化至测试集（90.48%），揭示了在小样本验证集上调参的风险。

3. **FGM + Rationale 联合训练出现拮抗效应**：测试准确率降至 90.11%，低于所有其他配置。我们提出**梯度冲突假说**解释——FGM 的嵌入扰动破坏了理性监督所依赖的词符级注意力目标，导致梯度信号相互抵消。

---

## 环境依赖

| 依赖 | 版本要求 |
|------|---------|
| Python | 3.9+ |
| PyTorch | 2.0+ |
| Transformers | 4.30+ |
| Datasets | 2.14+ |
| PEFT | 0.5+ |
| scikit-learn | 1.2+ |
| pandas | 2.0+ |
| matplotlib | 3.7+ |
| seaborn | 0.12+ |
| tqdm | 4.65+ |
| python-dotenv | 1.0+ |

**快速安装**：
```bash
pip install torch transformers datasets peft scikit-learn pandas matplotlib seaborn tqdm python-dotenv
```

**可选依赖**（仅当使用 LLM 功能时需要）：
- DeepSeek API Key（设置环境变量 `DEEPSEEK_API_KEY`）

---

## 项目结构

```
deep-learning-text-sentiment-analysis/
│
├── run_ABSA.py                      # 主实验脚本（核心入口）
├── search_lambda.py                 # Rationale λ 网格搜索脚本
├── generate_rationales.py           # 生成 Rationale 标注（需 LLM）
├── generate_augmented_data.py       # 生成增强训练数据（需 LLM）
├── generate_rationales.py           # Rationale 关键词提取
│
├── datasets/
│   └── semeval_split/
│       ├── train_original.csv       # 训练集（按 ID 划分）
│       ├── val_original.csv         # 验证集
│       ├── test_original.csv        # 测试集
│       ├── train_aug.csv            # 增强后的训练集（可选）
│       └── train_with_rationales.csv # 带理性标注的训练集（可选）
│
├── src/
│   ├── architectures/               # 模型定义
│   │   ├── roberta_aspect.py        # RoBERTa + Aspect Attention + LoRA
│   │   ├── bert.py                  # BERT 模型
│   │   ├── bilstm.py                # BiLSTM 模型
│   │   └── textcnn.py               # TextCNN 模型
│   │
│   ├── dataset/                     # 各模型对应的 Dataset 类
│   │   ├── aspect_roberta_dataset.py
│   │   ├── aspect_bert_dataset.py
│   │   ├── aspect_bilstm_dataset.py
│   │   └── aspect_textcnn_dataset.py
│   │
│   ├── training/                    # 训练器
│   │   ├── roberta_aspect_trainer.py  # RoBERTa 训练器（含 FGM、Rationale）
│   │   ├── bert_trainer.py
│   │   ├── bilstm_trainer.py
│   │   └── textcnn_trainer.py
│   │
│   ├── losses/                      # 损失函数
│   │   └── focal_loss.py            # Focal Loss 实现
│   │
│   ├── models/                      # 注意力模块
│   │   └── aspect_attention.py      # Aspect Attention 机制
│   │
│   ├── utils/                       # 工具模块
│   │   ├── config.py                # 全局配置管理
│   │   ├── llm_corrector.py         # DeepSeek 接口封装
│   │   ├── rationale_mask.py        # Rationale 掩码生成
│   │   ├── evaluator.py             # 评估器
│   │   ├── experiment_logger.py     # 实验记录器
│   │   ├── visualizer.py            # 多模型对比可视化
│   │   └── visualize_attention.py   # 注意力热力图可视化
│   │
│   └── scripts/                     # 数据加载
│       └── dataset_loader.py        # 数据集加载器
│
├── models/                          # 保存的模型权重
├── output/                          # 图表输出
│   ├── roberta_aspect/              # 训练曲线、混淆矩阵
│   ├── attention_plots/             # 注意力热力图
│   └── comparison/                  # 多模型对比图
│
├── experiments/                     # 实验结果
│   └── absa_results.json            # 所有实验指标汇总
│
├── rag_library/                     # RAG 案例库（已弃用）
├── .env                             # 环境变量（API Key）
└── README.md                        # 本文件
```

---

## 数据准备

### 1. 获取 SemEval-2014 Laptop 数据集
下载数据集后，按句子 ID 严格划分为三个文件，放入 `datasets/semeval_split/` 目录：

- `train_original.csv`（列：`id`, `Sentence`, `Aspect Term`, `polarity`）
- `val_original.csv`
- `test_original.csv`

> ⚠️ **重要**：必须按句子 ID 划分而非随机划分，防止同一句子的不同方面词泄露到多个集合中。

### 2. （可选）生成 Rationale 标注
如果需要使用理性注意力监督功能，运行：
```bash
python generate_rationales.py
```
确保已设置 `DEEPSEEK_API_KEY` 环境变量。生成的文件为 `train_with_rationales.csv`。

### 3. （可选）生成增强训练数据
```bash
python generate_augmented_data.py
```
增强后的数据保存为 `train_aug.csv`。

---

## 快速开始

### 1. 克隆仓库并安装依赖
```bash
git clone <your-repo-url>
cd deep-learning-text-sentiment-analysis
pip install -r requirements.txt
```

### 2. 配置 API Key（可选）
在项目根目录创建 `.env` 文件：
```
DEEPSEEK_API_KEY=your_api_key_here
```

### 3. 运行实验

#### 运行 Baseline（纯 RoBERTa）
在 `run_ABSA.py` 中设置：
```python
MODEL_CHOICE = "roberta"
USE_FGM = False
USE_RATIONALE = False
USE_AUG = False
USE_TRAIN_LABEL_CLEAN = False
USE_INFER_CORRECTION = False
```
```bash
python run_ABSA.py
```

#### 运行 Baseline + FGM
```python
USE_FGM = True
USE_RATIONALE = False
```

#### 运行 Baseline + Rationale
确保已生成 `train_with_rationales.csv`，然后：
```python
USE_FGM = False
USE_RATIONALE = True
```

#### 运行 FGM + Rationale 联合实验
```python
USE_FGM = True
USE_RATIONALE = True
```

### 4. 搜索最佳 Rationale λ（可选）
```bash
python search_lambda.py
```
脚本将自动测试 λ ∈ {0.05, 0.1, 0.2, 0.5, 1.0}，输出各 λ 对应的验证准确率及最优值。

### 5. 查看结果
- 训练曲线、混淆矩阵：`output/roberta_aspect/`
- 实验指标汇总：`experiments/absa_results.json`
- 多模型对比图：`output/comparison/`

### 6. 注意力可视化（可选）
```bash
python src/utils/visualize_attention.py
```
将生成 `output/attention_plots/` 目录，包含 Baseline 和 Rationale 模型的注意力热力图对比。

---

## 实验配置说明

所有可配置的全局变量均位于 `run_ABSA.py` 顶部：

| 变量 | 说明 | 可选值 |
|------|------|--------|
| `MODEL_CHOICE` | 选择骨干模型 | `"roberta"`, `"bert"`, `"bilstm"`, `"textcnn"` |
| `USE_FGM` | 是否使用 FGM 对抗训练 | `True` / `False` |
| `USE_RATIONALE` | 是否使用理性注意力监督 | `True` / `False` |
| `USE_AUG` | 是否使用数据增强 | `True` / `False` |
| `USE_TRAIN_LABEL_CLEAN` | 是否进行标签清洗 | `True` / `False` |
| `USE_INFER_CORRECTION` | 是否进行推理校正 | `True` / `False` |
| `LOW_CONF_THRESHOLD` | 低置信度阈值 | 0.0 - 1.0 |
| `MARGIN_THRESHOLD` | 预测边际阈值 | 0.0 - 1.0 |
| `PATIENCE` | 早停耐心值 | 正整数 |
| `DROPOUT` | Dropout 率 | 0.0 - 0.9 |
| `EPOCHS` | 最大训练轮数 | 正整数 |

---

## 可视化工具

### 训练曲线
自动生成训练/验证损失和准确率随 epoch 变化的对比曲线。


### 混淆矩阵
预测结果与真实标签的混淆矩阵。


### 注意力热力图对比
同一测试样本，对比 Baseline 模型（无 Rationale）与 Rationale 增强模型的注意力分布差异。


### 多模型对比图
所有已记录模型的 Acc/F1 柱状图、P/R/F1 折线图、雷达图。


---

## 致谢

- 本项目使用了 **SemEval-2014 Task 4** 公开数据集
- 理性标注生成借助 **DeepSeek** 大语言模型 API
- 模型骨干基于 HuggingFace **Transformers** 库和 **PEFT** 库

---



---

## 📄 许可证

MIT License

