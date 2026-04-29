import torch
import string

def create_rationale_mask(tokenizer, text, rationale_words, max_seq_len=128):
    """
    把 LLM 给出的情感关键词 → 变成 token 级 0/1 掩码
    返回 shape: [max_seq_len]
    """
    if not rationale_words or pd.isna(rationale_words):
        return torch.zeros(max_seq_len, dtype=torch.float)

    # 1. 对文本分词
    tokens = tokenizer.tokenize(text)
    mask = [0] * max_seq_len

    # 2. 处理关键词（小写、去标点）
    keywords = [w.strip().lower().strip(string.punctuation) for w in rationale_words.split(",")]

    # 3. 匹配 token
    for idx, token in enumerate(tokens):
        if idx >= max_seq_len:
            break

        token_clean = token.lower().strip(string.punctuation)

        for kw in keywords:
            if kw in token_clean or token_clean in kw:
                mask[idx] = 1
                break

    return torch.tensor(mask, dtype=torch.float)