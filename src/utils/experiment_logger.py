import json
from pathlib import Path
from datetime import datetime


class ExperimentLogger:
    """
    统一实验记录器
    """

    def __init__(self, save_dir="experiments"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.file_path = self.save_dir / "absa_results.json"

        # ===================== 修复点：如果文件已存在，就读取旧数据 =====================
        if self.file_path.exists():
            with open(self.file_path, "r", encoding="utf-8") as f:
                self.result = json.load(f)
        else:
            self.result = {
                "timestamp": str(datetime.now()),
                "models": {}
            }

    def log_model(self, model_name, metrics):
        """
        metrics = {
            "acc": ...,
            "f1": ...,
            "loss": ...,
            "precision": ...,
            "recall": ...
        }
        """
        self.result["models"][model_name] = metrics

    def save(self, filename="results.json"):
        path = self.save_dir / filename

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.result, f, indent=4, ensure_ascii=False)

        print(f"✅ 实验结果已保存: {path}")