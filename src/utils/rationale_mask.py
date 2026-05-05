import torch
import string
import pandas as pd

def create_rationale_mask(tokenizer, text, rationale_str, max_seq_len=256):
    """
    把 LLM 给出的情感关键词 → 变成 token 级 0/1 掩码
    返回 shape: [max_seq_len]
    """
    if not rationale_str or pd.isna(rationale_str):
        return torch.zeros(max_seq_len, dtype=torch.float)

    # 1. 把 rationale 字符串切分成关键词列表
    keywords = [kw.strip().lower() for kw in rationale_str.split(",") if kw.strip()]

    # 2. 对原句子进行 RoBERTa 分词（得到真正的 token 序列）
    tokenized = tokenizer(
        text,
        max_length=max_seq_len,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True  # 关键：拿到 token 对应原文的位置
    )

    input_ids = tokenized["input_ids"]
    offset_mapping = tokenized["offset_mapping"]  # List[(start, end)]

    # 3. 初始化 mask（全 0）
    rationale_mask = [0] * len(input_ids)

    # 4. 遍历每个关键词，找到它在原文中的位置 → 映射到 token
    text_low = text.lower()

    for kw in keywords:
        if len(kw) < 2:
            continue  # 过滤太短的无效词

        # 在原文中找所有出现的位置
        start_idx = 0
        while True:
            pos = text_low.find(kw, start_idx)
            if pos == -1:
                break

            # 找到这个字符区间对应哪些 token
            for i, (token_start, token_end) in enumerate(offset_mapping):
                if token_start == 0 and token_end == 0:
                    continue  # 跳过特殊符号 <s> </s> <pad>
                # 判断是否重叠
                if not (token_end <= pos or token_start >= pos + len(kw)):
                    rationale_mask[i] = 1

            start_idx = pos + len(kw)

    # 5. 特殊位置强制不参与监督（<s>, </s>, padding）
    for i in range(len(input_ids)):
        if input_ids[i] in [tokenizer.cls_token_id, tokenizer.sep_token_id, tokenizer.pad_token_id]:
            rationale_mask[i] = 0

    # 6. 转 tensor
    rationale_mask = torch.tensor(rationale_mask, dtype=torch.float32)
    return rationale_mask