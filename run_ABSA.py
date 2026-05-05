import sys
import os
import transformers.utils.import_utils
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None

os.environ["TRANSFORMERS_FORCE_SAFETENSORS"] = "1"
os.environ["TRANSFORMERS_DISABLE_LOAD_SAFETENSORS"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "1"  # transformers 离线模式
os.environ["HF_DATASETS_OFFLINE"] = "1"  # datasets 离线模式
os.environ["PYTHONUNBUFFERED"] = "1"  # 强制无缓冲
sys.stdout.reconfigure(line_buffering=True)  # 强制实时打印
import torch
if torch.cuda.is_available():
    torch.cuda.init()

from dotenv import load_dotenv
load_dotenv()
from src.utils.case_retrieval import CaseRetrieval

sys.path.append("src")  # 确保 src 在路径中
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from pathlib import Path

from src.utils.config import Config
from src.utils.llm_corrector import DeepSeekCorrector
from src.scripts.dataset_loader import DatasetLoader
from src.utils.experiment_logger import ExperimentLogger
from src.utils.evaluator import Evaluator
from src.utils.visualizer import ExperimentVisualizer

import random
import numpy as np

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=1000, alpha=0.95):
    """计算准确率的 95% Bootstrap 置信区间"""
    n = len(y_true)
    accs = []
    rng = np.random.default_rng(42)  # 固定随机种子，保证可复现
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)   # 有放回抽样
        acc = np.mean(y_pred[indices] == y_true[indices])
        accs.append(acc)
    lower = np.percentile(accs, (1 - alpha) / 2 * 100)
    upper = np.percentile(accs, (1 + alpha) / 2 * 100)
    return np.mean(accs), (lower, upper)

# ===================== 全局配置 =====================
LANGUAGE = "english"
TASK_TYPE = "aspect"
BATCH_SIZE = 16
EPOCHS = 6
MAX_SEQ_LEN = 256
LOW_CONF_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.2

# 统一超参
LR_ROBERTA = 1.2e-5
LR_BERT = 1.2e-5
LR_CNN = 2e-4
LR_LSTM = 1e-4
PATIENCE = 2
DROPOUT = 0.4

# 选择你要跑的模型 👇
MODEL_CHOICE = "roberta"    # choices: roberta / bert / bilstm / textcnn

USE_FGM = True
USE_AUG = False
USE_RATIONALE = True
USE_TRAIN_LABEL_CLEAN = False
USE_INFER_CORRECTION = False
USE_DEEPSEEK_CORRECTION = USE_TRAIN_LABEL_CLEAN or USE_INFER_CORRECTION
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# ================================================
suffix_parts = []

if USE_FGM:
    suffix_parts.append("fgm")
if USE_AUG:
    suffix_parts.append("aug")
if USE_RATIONALE:
    suffix_parts.append("rationale")
if USE_TRAIN_LABEL_CLEAN:
    suffix_parts.append("clean")
if USE_INFER_CORRECTION:
    suffix_parts.append("refine")

version_suffix = "_".join(suffix_parts) if suffix_parts else "base"

# 自动按模型生成保存路径
model_type_map = {
    "roberta": "roberta_aspect",
    "bert": "bert",
    "bilstm": "bilstm",
    "textcnn": "textcnn"
}
model_type = model_type_map[MODEL_CHOICE]
model_versioned_name = f"{model_type}_{LANGUAGE}_{version_suffix}"

# ===================== 数据集自动匹配 =====================
def get_dataset_and_loader(MODEL_CHOICE, data, tokenizer=None, vocab=None, max_len=128):
    if MODEL_CHOICE == "roberta":
        from src.dataset.aspect_roberta_dataset import AspectRobertaDataset
        dataset = AspectRobertaDataset(data, tokenizer,max_len)
    elif MODEL_CHOICE == "bert":
        from src.dataset.aspect_bert_dataset import AspectBertDataset
        dataset = AspectBertDataset(data, tokenizer, max_len)
    elif MODEL_CHOICE == "bilstm":
        from src.dataset.aspect_bilstm_dataset import AspectBiLSTMDataset
        dataset = AspectBiLSTMDataset(data, vocab, max_len)
    elif MODEL_CHOICE == "textcnn":
        from src.dataset.aspect_textcnn_dataset import AspectTextCNNDataset
        dataset = AspectTextCNNDataset(data, vocab, max_len)
    else:
        raise ValueError("不支持的模型")
    
    return dataset

# ===================== 训练器自动加载 =====================
def get_trainer(MODEL_CHOICE, language="english", vocab=None, lr=None, patience=None, use_rationale=None, rationale_lambda=0.5, use_gfm=None):
    if MODEL_CHOICE == "roberta":
        from src.training.roberta_aspect_trainer import RobertaAspectTrainer
        return RobertaAspectTrainer(language=language, lr=lr, patience=patience, use_rationale=use_rationale, rationale_lambda=rationale_lambda, use_gfm=use_gfm)
    elif MODEL_CHOICE == "bert":
        from src.training.bert_trainer import BertTrainer
        return BertTrainer(language=language, lr=lr, patience=patience)
    elif MODEL_CHOICE == "bilstm":
        from src.training.bilstm_trainer import BiLSTMTrainer
        return BiLSTMTrainer(language=language, vocab=vocab, lr=lr, patience=patience)
    elif MODEL_CHOICE == "textcnn":
        from src.training.textcnn_trainer import TextCNNTrainer
        return TextCNNTrainer(language=language, vocab=vocab, lr=lr, patience=patience)
    else:
        raise ValueError("不支持的模型")

# ===================== 推理 + LLM 修正 =====================
def inference_with_llm_refine(MODEL_CHOICE, model, tokenizer, test_data, corrector):
    model.eval()
    final_preds = []
    low_conf_count = 0
    device = next(model.parameters()).device

    from transformers import pipeline
    sentiment_model = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")


    with torch.no_grad():
        for item in tqdm(test_data, desc="推理流程"):
            text = item["text"]
            aspect = item["aspect"]
            

            # ===================== 模型编码输入 =====================
            if MODEL_CHOICE in ["roberta", "bert"]:
                encode = tokenizer.encode_sentence_aspect_pair([text], [aspect], MAX_SEQ_LEN)
                input_ids = encode["input_ids"].to(device)
                attention_mask = encode["attention_mask"].to(device)
                logits, _ = model(input_ids, attention_mask)
            else:
                combined = f"{text} [SEP] {aspect}"
                tokens = combined.lower().split()
                input_ids = [tokenizer.get(t, 1) for t in tokens]
                if len(input_ids) > MAX_SEQ_LEN:
                    input_ids = input_ids[:MAX_SEQ_LEN]
                else:
                    input_ids += [0] * (MAX_SEQ_LEN - len(input_ids))
                input_ids = torch.tensor([input_ids]).to(device)
                logits, _ = model(input_ids)

            # 置信度计算
            prob = F.softmax(logits, dim=-1).squeeze(0)
            top2 = torch.topk(prob, k=2)
            p1 = top2.values[0].item()
            p2 = top2.values[1].item()
            pred = top2.indices[0].item()
            confidence = p1
            margin = p1 - p2
            need_llm = (confidence < LOW_CONF_THRESHOLD) or (margin < MARGIN_THRESHOLD)

            # ===================== LLM 修正（只有开启 + 低置信才触发） =====================
            if USE_INFER_CORRECTION and need_llm:
                try:
                    low_conf_count += 1
                    refine = corrector.correct_sentiment(text, aspect, pred)
                    final_pred = corrector.label_mapping(refine)
                except:
                    final_pred = pred
            else:
                final_pred = pred

            final_preds.append(final_pred)

    print(f"低置信度难例总数: {low_conf_count}")
    return final_preds

# ===================== 统一可视化：单个模型内部图表 =====================
def plot_single_model_training_curve(trainer, model_type, model_versioned_name):
    save_dir = Config.MODEL_OUTPUT_PATHS[model_type]
    save_dir.mkdir(parents=True, exist_ok=True)

    history = trainer.history
    epochs_range = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history["train_loss"], label="Train Loss")
    plt.plot(epochs_range, history["val_loss"], label="Val Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history["train_acc"], label="Train Acc")
    plt.plot(epochs_range, history["val_acc"], label="Val Acc")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig(save_dir / f"training_curve_{model_versioned_name}.png", dpi=300)
    plt.close()


def plot_single_model_confusion_matrix(true_labels, pred_labels, model_type, model_versioned_name):
    save_dir = Config.MODEL_OUTPUT_PATHS[model_type]
    save_dir.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(true_labels, pred_labels)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title("Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    plt.savefig(save_dir / f"confusion_matrix_{model_versioned_name}.png", dpi=300)
    plt.close()


# ===================== 主程序 =====================
if __name__ == "__main__":
    
    #------ 实验种子列表 ------
    seeds = [42, 123, 456]
    all_test_metrics = []

    all_y_true_all_seeds = []
    all_y_pred_all_seeds = []

    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"  Running with seed: {seed}")
        print(f"{'='*60}")
        set_seed(seed)

        #loader = DatasetLoader(language=LANGUAGE, task_type=TASK_TYPE)
        #train_dataset, val_dataset, test_dataset = loader.get_or_download_data()

        train_df = pd.read_csv("datasets/semeval_split/train_original.csv")
        val_df   = pd.read_csv("datasets/semeval_split/val_original.csv")
        test_df  = pd.read_csv("datasets/semeval_split/test_original.csv")


        import pandas as pd
        from datasets import Dataset

        def preprocess_df(df):
            df = df.rename(columns={
                "Sentence": "text",
                "Aspect Term": "aspect",
                "polarity": "label"
            })
            df = df.dropna(subset=["text", "aspect", "label"])
            df["label"] = df["label"].map({"positive": 1, "negative": 0})
            df = df.dropna(subset=["label"])
            df["label"] = df["label"].astype(int)
            return df[["text", "aspect", "label"]]

        train_df = preprocess_df(train_df)
        val_df   = preprocess_df(val_df)
        test_df  = preprocess_df(test_df)

        # 转为 HuggingFace Dataset
        train_dataset = Dataset.from_pandas(train_df, preserve_index=False)
        val_dataset   = Dataset.from_pandas(val_df,   preserve_index=False)
        test_dataset  = Dataset.from_pandas(test_df,  preserve_index=False)

        #数据增强
        if USE_AUG:
            train_aug_df = pd.read_csv("./datasets/semeval_split/train_aug.csv")

            # 重命名列 → 匹配你内部格式
            train_aug_df = train_aug_df.rename(columns={
                "Sentence": "text",
                "Aspect Term": "aspect",
                "polarity": "label"
            })

            # 过滤、映射标签
            train_aug_df = train_aug_df.dropna(subset=["text", "aspect", "label"])
            # 先映射，再删除空标签，最后转int
            train_aug_df["label"] = train_aug_df["label"].map({"positive": 1, "negative": 0})
            # 先dropna，再astype
            train_aug_df = train_aug_df.dropna(subset=["label"])
            train_aug_df["label"] = train_aug_df["label"].astype(int)

            # 替换原来的 train_dataset 
            train_dataset = Dataset.from_pandas(train_aug_df[["text", "aspect", "label"]], preserve_index=False)

            print(f"✅ 已替换为【增强后训练集】，数量：{len(train_dataset)}")
    
        if USE_RATIONALE:
            rationale_path = "datasets/semeval_split/train_with_rationales.csv"
            if os.path.exists(rationale_path):
                rationale_df = pd.read_csv(rationale_path)
                rationale_df = rationale_df.rename(columns={
                    "Sentence": "text",
                    "Aspect Term": "aspect",
                    "polarity": "label"
                })
                rationale_df["label"] = rationale_df["label"].map({"positive": 1, "negative": 0})
                rationale_df = rationale_df.dropna(subset=["text", "aspect", "label", "rationale"])
                rationale_df["label"] = rationale_df["label"].astype(int)
                train_dataset = Dataset.from_pandas(
                    rationale_df[["text", "aspect", "label", "rationale"]],
                    preserve_index=False
                )
                print(f"✅ 已加载【含 rationale 训练集】，数量：{len(train_dataset)}")
            else:
                print(f"❌ 未找到 {rationale_path}，请先运行 generate_rationales.py！已关闭 USE_RATIONALE")
                USE_RATIONALE = False
            
        print(f"训练: {len(train_dataset)} | 验证: {len(val_dataset)} | 测试: {len(test_dataset)}")

    


        corrector = DeepSeekCorrector(DEEPSEEK_API_KEY)

        # 训练集标签清洗
        if USE_TRAIN_LABEL_CLEAN:
            print("\n🧠 DeepSeek 训练集标签清洗中...")
            train_list = train_dataset.to_list()
            for sample in tqdm(train_list, desc="清洗标签"):
                try:
                    new_lab = corrector.correct_sentiment(sample["text"], sample["aspect"], sample["label"])
                    sample["original_label"] = sample["label"]
                    sample["label"] = corrector.label_mapping(new_lab)
                except Exception as e:
                    print("LLM error:", e)
            from datasets import Dataset
            train_dataset = Dataset.from_list(train_list)

        # 自动构建词汇表(传统模型适配) - 只有 BiLSTM 和 TextCNN 需要
        vocab = None
        if MODEL_CHOICE in ["bilstm", "textcnn"]:
            all_texts = []
            for item in train_dataset.to_list():
                    comb = f"{item['text']} [SEP] {item['aspect']}"
                    all_texts.extend(comb.lower().split())
            vocab = {"<pad>":0, "<unk>":1}
            for w in all_texts:
                if w not in vocab:
                    vocab[w] = len(vocab)

        # 自动获取训练器
        trainer = get_trainer(
            MODEL_CHOICE, 
            language=LANGUAGE, 
            vocab=vocab,
            lr={
                "roberta": LR_ROBERTA,
                "bert": LR_BERT,
                "textcnn": LR_CNN,
                "bilstm": LR_LSTM
            }[MODEL_CHOICE],
            patience=PATIENCE,
            use_rationale=USE_RATIONALE,
            use_gfm=USE_FGM
        )
        if hasattr(trainer, "create_model"):
            trainer.create_model()

        tokenizer = trainer.tokenizer_wrapper if MODEL_CHOICE in ["roberta","bert"] else vocab

        # 自动构建数据集
        train_ds = get_dataset_and_loader(MODEL_CHOICE, train_dataset.to_list(), tokenizer, vocab, MAX_SEQ_LEN)
        val_ds = get_dataset_and_loader(MODEL_CHOICE, val_dataset.to_list(), tokenizer, vocab, MAX_SEQ_LEN)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

        # 训练
        print(f"\n🚀 开始训练 {MODEL_CHOICE} ...")
        trainer.train(train_loader, val_loader, epochs=EPOCHS)

    

        # 保存模型
        import shutil
        original_model_path = Config.get_model_path(model_type, LANGUAGE)
        versioned_model_path = Config.MODELS_DIR / f"{model_versioned_name}_seed{seed}.pth"
        shutil.copy(str(original_model_path), str(versioned_model_path))
        print(f"\n【最优模型已版本化保存】: {versioned_model_path}")
        # ===========================================================================

        # 推理 + LLM 修正
        test_list = test_dataset.to_list()
        final_preds = inference_with_llm_refine(MODEL_CHOICE, trainer.model, tokenizer, test_list, corrector)
        true_labels = [x["label"] for x in test_list]

        all_y_true_all_seeds.extend(true_labels)
        all_y_pred_all_seeds.extend(final_preds)

        # 输出结果
        evaluator = Evaluator()
        metrics = evaluator.evaluate(true_labels, final_preds)
        print(f"\nSeed {seed} 测试结果: {metrics}")

        report = classification_report(true_labels, final_preds, output_dict=True)
        acc = accuracy_score(true_labels, final_preds)
        f1 = report["weighted avg"]["f1-score"]
        precision = report["weighted avg"]["precision"]
        recall = report["weighted avg"]["recall"]

        test_metrics = {"acc": acc, "f1": f1, "precision": precision, "recall": recall}
        all_test_metrics.append(test_metrics)
        print(f"Seed {seed} Test Metrics: {test_metrics}")

        if seed == seeds[-1]:
            print("\n📊 生成单个模型训练曲线 & 混淆矩阵...")
            plot_single_model_training_curve(trainer, model_type, f"{model_versioned_name}_seed{seed}")
            plot_single_model_confusion_matrix(true_labels, final_preds, model_type, f"{model_versioned_name}_seed{seed}")
        
    
    # --- 汇总所有种子 ---
    print("\n" + "="*60)
    print("Final Summary over seeds:", seeds)
    avg = {k: np.mean([m[k] for m in all_test_metrics]) for k in all_test_metrics[0]}
    std = {k: np.std([m[k] for m in all_test_metrics]) for k in all_test_metrics[0]}
    for k in avg:
        print(f"{k}: {avg[k]:.4f} ± {std[k]:.4f}")

    # ---------- Bootstrap 置信区间 ----------
    if all_y_true_all_seeds and all_y_pred_all_seeds:
        y_true_all = np.array(all_y_true_all_seeds)
        y_pred_all = np.array(all_y_pred_all_seeds)
        acc_boot_mean, (acc_boot_lower, acc_boot_upper) = bootstrap_accuracy_ci(
            y_true_all, y_pred_all, n_bootstrap=1000
        )
        print(f"\nBootstrap 95% CI for Accuracy: {acc_boot_mean:.4f} ({acc_boot_lower:.4f}, {acc_boot_upper:.4f})")

    # 记录最终平均结果
    logger = ExperimentLogger()
    logger.log_model(model_versioned_name, {
        "acc": avg["acc"],
        "f1": avg["f1"],
        "precision": avg["precision"],
        "recall": avg["recall"],
        "std_acc": std["acc"],
        "std_f1": std["f1"],
        "bootstrap_acc_95ci_lower": acc_boot_lower,
        "bootstrap_acc_95ci_upper": acc_boot_upper
    })
    logger.save("absa_results.json")

    # 可视化对比图 (使用汇总后的平均指标)
    viz = ExperimentVisualizer("experiments/absa_results.json")
    viz.plot_main_metrics()
    viz.plot_prf()
    viz.plot_radar()
    