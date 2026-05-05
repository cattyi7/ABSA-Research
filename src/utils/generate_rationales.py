from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
from tqdm import tqdm
from llm_corrector import DeepSeekCorrector  

# 初始化LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
corrector = DeepSeekCorrector(DEEPSEEK_API_KEY)

def filter_rationale(rationale, sentence):
    import string
    sentence = sentence.lower().strip()
    words = [w.strip().lower() for w in rationale.split(",") if w.strip()]
    
    valid_words = []
    for w in words:
        # 去掉标点
        w_clean = w.translate(str.maketrans("", "", string.punctuation))
        # 必须在句子里 + 长度大于2 + 不是空
        if len(w_clean) > 2 and w_clean in sentence:
            valid_words.append(w_clean)
    
    return ", ".join(valid_words) if valid_words else ""


# 加载训练集
train_df = pd.read_csv("datasets/semeval_split/train_original.csv")  # 你自己的路径

rationales = []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    text = row["Sentence"]
    aspect = row["Aspect Term"]

    # ===================== Prompt =====================
    prompt = f"""
You are an expert at aspect-level sentiment analysis.
Given a sentence and a target aspect, extract ONLY the exact English words that determine the sentiment polarity.

Rules:
1. Extract ONLY words that appear in the sentence.
2. DO NOT create, paraphrase, or translate any word.
3. Output ONLY sentiment words (adjectives, verbs).
4. Separate multiple words with a comma.
5. Do NOT output any extra text, Chinese, or explanation.

Sentence: {text}
Aspect: {aspect}
Keywords:
""".strip()

    # ===================== 调用 _call_api =====================
    rationale = corrector._call_api(prompt, temperature=0.0)
    clean_rationale = filter_rationale(rationale, text)
    rationales.append(clean_rationale)

    

# 保存带 Rationale 的训练集
train_df["rationale"] = rationales
train_df.to_csv("datasets/semeval_split/train_with_rationales.csv", index=False, encoding="utf-8")

print("✅ 完成！已生成 train_with_rationales.csv")