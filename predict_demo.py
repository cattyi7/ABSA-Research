import sys
sys.path.append("src")

from dotenv import load_dotenv
import os
load_dotenv()

import torch
import torch.nn.functional as F

from src.utils.config import Config
from src.utils.llm_corrector import DeepSeekCorrector  # 新增

import random
import numpy as np
import torch

#固定随机种子
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

LANGUAGE = "english"
MODEL_CHOICE = "bilstm"

USE_TRAIN_LABEL_CLEAN = False
USE_INFER_CORRECTION = True  # 开启推理纠偏
LOW_CONF_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.2
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
# ===============================================================

clean_suffix = "clean" if USE_TRAIN_LABEL_CLEAN else "noclean"
version_suffix = clean_suffix

# 模型类型映射
model_type_map = {
    "roberta": "roberta_aspect",
    "bert": "bert",
    "bilstm": "bilstm",
    "textcnn": "textcnn"
}
model_type = model_type_map[MODEL_CHOICE]
model_path = Config.get_model_path(model_type, LANGUAGE)

# 初始化LLM纠偏器
corrector = DeepSeekCorrector(DEEPSEEK_API_KEY)

# ===================== 加载正确的模型 =====================
def load_trained_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if MODEL_CHOICE == "roberta":
        from src.training.roberta_aspect_trainer import RobertaAspectTrainer
        trainer = RobertaAspectTrainer(language=LANGUAGE)
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = trainer.tokenizer_wrapper

    elif MODEL_CHOICE == "bert":
        from src.training.bert_trainer import BertTrainer
        trainer = BertTrainer(language=LANGUAGE)
        trainer.create_model()
        checkpoint = torch.load(model_path, map_location=device)
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = trainer.tokenizer_wrapper

    elif MODEL_CHOICE == "bilstm":
        from src.training.bilstm_trainer import BiLSTMTrainer
        checkpoint = torch.load(model_path, map_location=device)
        vocab = checkpoint["vocab"]
        trainer = BiLSTMTrainer(LANGUAGE, vocab=vocab)
        trainer.create_model()
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = vocab

    elif MODEL_CHOICE == "textcnn":
        from src.training.textcnn_trainer import TextCNNTrainer
        checkpoint = torch.load(model_path, map_location=device)
        vocab = checkpoint["vocab"]
        trainer = TextCNNTrainer(LANGUAGE, vocab=vocab)
        trainer.create_model()
        trainer.model.load_state_dict(checkpoint["model_state_dict"])
        tokenizer = vocab

    else:
        raise ValueError("不支持的模型")

    trainer.model.eval()
    return trainer, tokenizer

# ===================== 预测（带 LLM 纠偏） =====================
def predict(trainer, tokenizer, text, aspect):
    model = trainer.model
    device = next(model.parameters()).device
    max_len = 128

    with torch.no_grad():
        if MODEL_CHOICE in ["roberta", "bert"]:
            if MODEL_CHOICE == "roberta":
                encode = tokenizer.encode_sentence_aspect_pair([text], [aspect], max_len)
            else:
                encode = tokenizer.tokenizer(
                    text, aspect,
                    max_length=max_len,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt"
                )

            input_ids = encode["input_ids"].to(device)
            attention_mask = encode["attention_mask"].to(device)
            logits = model(input_ids, attention_mask)
        else:
            combined = f"{text} [SEP] {aspect}"
            tokens = combined.lower().split()
            input_ids = [tokenizer.get(t, 1) for t in tokens]
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
            else:
                input_ids += [0] * (max_len - len(input_ids))
            input_ids = torch.tensor([input_ids]).to(device)
            logits = model(input_ids)

    prob = F.softmax(logits, dim=-1).squeeze(0)

    top2 = torch.topk(prob,k=2)
    p1 = top2.values[0].item()
    p2 = top2.values[1].item()

    pred = top2.indices[0].item()

    confidence = p1
    margin = p1-p2

    need_llm = (confidence < LOW_CONF_THRESHOLD) or (margin < MARGIN_THRESHOLD)

    print(f"[INFO] pred={pred}, conf={confidence:.3f}, margin={margin:.3f}, llm={need_llm}")

    # ===================== LLM 纠偏逻辑 =====================
    if USE_INFER_CORRECTION and need_llm:
        print(f"⚠️  低置信度({confidence:.2f})，启用LLM纠偏...")
        try:
            refine_label = corrector.correct_sentiment(text, aspect, pred)
            final_pred = corrector.label_mapping(refine_label)
            pred = final_pred
            confidence = 1.0  # 纠偏后置信度设为1
        except:
            print("❌ LLM纠偏失败，使用原模型预测")

    label = "Positive" if pred == 1 else "Negative"
    print(f"最终预测：{label} | 置信度：{confidence:.2f}")
    return label, confidence

# ===================== 主程序 =====================
if __name__ == "__main__":
    print(f"加载模型：{model_type} | 版本：{version_suffix}")
    trainer, tokenizer = load_trained_model()
    print("模型加载成功！\n")

    while True:
        text = input("输入句子：")
        if text == "quit":
            break
        aspect = input("输入Aspect：")
        predict(trainer, tokenizer, text, aspect)
        print("-" * 50)