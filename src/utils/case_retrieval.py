from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os

class CaseRetrieval:
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.embedder = SentenceTransformer(model_name).to(device)
        self.cases = []
        self.corpus_embeds = None
        self.device = device

    # ===================== 构建并保存到本地 =====================
    def build_case_library(self, train_list, save_path="rag_library/case_library.pkl"):
        self.cases = train_list
        corpus_texts = [f"{item['text']} [SEP] {item['aspect']}" for item in train_list]
        
        print("→ 首次构建案例库向量，这可能需要一些时间...")
        self.corpus_embeds = self.embedder.encode(corpus_texts, convert_to_tensor=True)

        # 保存到本地
        with open(save_path, 'wb') as f:
            pickle.dump((self.cases, self.corpus_embeds), f)
        print(f"✅ 案例库已保存到 {save_path}，下次直接加载！")

    # ===================== 从本地加载 =====================
    def load_case_library(self, load_path="rag_library/case_library.pkl"):
        if not os.path.exists(load_path):
            return False
            
        with open(load_path, 'rb') as f:
            self.cases, self.corpus_embeds = pickle.load(f)
        print(f"✅ 从本地加载案例库：{len(self.cases)} 条 | 秒级完成！")
        return True

    # ===================== 推理检索 =====================
    def retrieve_top_k(self, text, aspect, top_k=2):
        query = f"{text} [SEP] {aspect}"
        query_embed = self.embedder.encode(query, convert_to_tensor=True)
        
        scores = util.cos_sim(query_embed, self.corpus_embeds)[0]
        top_indices = torch.topk(scores, k=top_k).indices.cpu().numpy()
        return [self.cases[i] for i in top_indices]