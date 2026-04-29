# -*- coding: utf-8 -*-
from dotenv import load_dotenv
import os
load_dotenv()

import pandas as pd
import time
from tqdm import tqdm
from llm_corrector import DeepSeekCorrector

# ===================== 配置 =====================
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")  
INPUT_CSV = "./datasets/semeval_split/train_original.csv"
OUTPUT_CSV = "./datasets/semeval_split/train_aug.csv"
AUG_NUM = 3  # 每条生成3条增强

PROMPT_TPL = """
你是英文ABSA数据增强专家，只输出增强后的英文句子，严格遵守规则：
1. 保持 Aspect 不变
2. 保持情感极性不变
3. 只改写句子，意思不变
4. 输出 {num} 条句子，一行一条
5. 不要任何解释、编号

原句：{sentence}
Aspect：{aspect}
极性：{polarity}
"""

def main():
    print("开始 LLM 数据增强...")
    corrector = DeepSeekCorrector(DEEPSEEK_API_KEY)
    df = pd.read_csv(INPUT_CSV)
    output_rows = []
    new_id = 1000000

    for _, row in tqdm(df.iterrows(), total=len(df)):
        sentence = row["Sentence"]
        aspect = row["Aspect Term"]
        polarity = row["polarity"]

        # 保存原始样本
        output_rows.append(row.to_dict())

        # 生成增强
        prompt = PROMPT_TPL.format(
            num=AUG_NUM,
            sentence=sentence,
            aspect=aspect,
            polarity=polarity
        )

        try:
            content = corrector.generate_augmented(prompt)
            aug_sents = [s.strip() for s in content.splitlines() if s.strip()]

            for s in aug_sents[:AUG_NUM]:
                new_row = {
                    "id": new_id,
                    "Sentence": s,
                    "Aspect Term": aspect,
                    "polarity": polarity,
                    "from": "",
                    "to": ""
                }
                output_rows.append(new_row)
                new_id += 1
        except Exception as e:
            print("增强出错:", e)

        time.sleep(0.7)

    df_out = pd.DataFrame(output_rows)
    df_out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"✅ 增强完成！原始：{len(df)} → 增强后：{len(df_out)}")

if __name__ == "__main__":
    main()