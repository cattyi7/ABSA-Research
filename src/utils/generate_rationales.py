from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
from tqdm import tqdm
from llm_corrector import DeepSeekCorrector  # 用你现有的！

# 初始化LLM
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")
corrector = DeepSeekCorrector(DEEPSEEK_API_KEY)

# 加载训练集
train_df = pd.read_csv("datasets/semeval_split/train_aug.csv")  # 你自己的路径

rationales = []

for idx, row in tqdm(train_df.iterrows(), total=len(train_df)):
    text = row["Sentence"]
    aspect = row["Aspect Term"]

    # ===================== Prompt =====================
    prompt = f"""
你是专业的英文方面级情感分析助手。
任务：给定句子和评价方面，只提取决定情感的核心英文关键词/短语。
规则：
1. 只输出英文关键词，不要解释
2. 多个关键词用英文逗号 , 分隔
3. 不要输出任何多余内容、不要中文、不要解释

句子：{text}
评价方面：{aspect}
核心情感关键词：
""".strip()

    # ===================== 调用 _call_api =====================
    rationale = corrector._call_api(prompt, temperature=0.0)

    rationales.append(rationale)

# 保存带 Rationale 的训练集
train_df["rationale"] = rationales
train_df.to_csv("datasets/semeval_split/train_with_rationales.csv", index=False, encoding="utf-8")

print("✅ 完成！已生成 train_with_rationales.csv")