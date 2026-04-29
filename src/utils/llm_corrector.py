import requests
import json
import os

class DeepSeekCorrector:
    def __init__(self, api_token):
        self.cache = {}
        self.api_token = api_token
        self.url = "https://api.deepseek.com/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json"
        }
        self.cache_path = "llm_cache.json"

    # --------------------------
    # 统一的API调用接口
    # --------------------------
    def _call_api(self, prompt, temperature=0.1):
        data = {
            "model": "deepseek-chat",
            "messages": [{"role": "user", "content": prompt.strip()}],
            "temperature": temperature
        }
        try:
            rsp = requests.post(
                self.url,
                headers=self.headers,
                json=data,
                timeout=15
            )
            rsp.raise_for_status()
            return rsp.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print("LLM API Error:", e)
            return ""

    # --------------------------
    # 标签清洗
    # --------------------------
    def correct_sentiment(self, text, aspect, original_label):
        key = f"{text}|||{aspect}|||{original_label}"

        # 内存缓存
        if key in self.cache:
            return self.cache[key]

        # 加载文件缓存
        if os.path.exists(self.cache_path) and len(self.cache) == 0:
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except:
                self.cache = {}

        if key in self.cache:
            return self.cache[key]

        # 构造prompt
        prompt = f"""
你是情感分析专家，只输出【正面】或【负面】，不要其他内容。
文本：{text}
评价维度：{aspect}
情感标签：{original_label}
请纠正标签是否正确。
        """

        # 调用统一API
        result = self._call_api(prompt, temperature=0.1)

        # 缓存保存
        if result:
            self.cache[key] = result
            try:
                with open(self.cache_path, "w", encoding="utf-8") as f:
                    json.dump(self.cache, f, ensure_ascii=False, indent=2)
            except:
                pass

        return result if result else original_label

    # --------------------------
    # 标签映射
    # --------------------------
    def label_mapping(self, label):
        if "正面" in label:
            return 1
        elif "负面" in label:
            return 0
        return 0

    # --------------------------
    # 数据增强专用
    # --------------------------
    def generate_augmented(self, prompt):
        return self._call_api(prompt, temperature=0.6)
    
    