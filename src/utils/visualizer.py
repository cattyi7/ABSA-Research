import json
import matplotlib.pyplot as plt
import numpy as np

from pathlib import Path



class ExperimentVisualizer:
    """
    多模型对比可视化
    """

    def __init__(self, json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        self.models = self.data["models"]
        self.save_dir = Path("output/comparison")
        self.save_dir.mkdir(parents=True, exist_ok=True)

    # =========================
    # 1. Accuracy / F1 对比柱状图
    # =========================
    def plot_main_metrics(self, save_path="metrics_comparison.png"):
        names = list(self.models.keys())
        acc = [self.models[m]["acc"] for m in names]
        f1 = [self.models[m]["f1"] for m in names]

        x = np.arange(len(names))
        width = 0.35

        plt.figure(figsize=(8,5))
        plt.bar(x - width/2, acc, width, label="Accuracy")
        plt.bar(x + width/2, f1, width, label="F1-score")

        plt.xticks(x, names, rotation=20)
        plt.ylim(0, 1)
        plt.title("Model Comparison (Acc / F1)")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / save_path, dpi=300)
        plt.close()

    # =========================
    # 2. Precision / Recall / F1
    # =========================
    def plot_prf(self, save_path="prf_comparison.png"):
        names = list(self.models.keys())

        precision = [self.models[m]["precision"] for m in names]
        recall = [self.models[m]["recall"] for m in names]
        f1 = [self.models[m]["f1"] for m in names]

        x = np.arange(len(names))

        plt.figure(figsize=(9,5))
        plt.plot(x, precision, marker="o", label="Precision")
        plt.plot(x, recall, marker="o", label="Recall")
        plt.plot(x, f1, marker="o", label="F1")

        plt.xticks(x, names, rotation=20)
        plt.ylim(0, 1)
        plt.title("PRF Comparison")
        plt.legend()

        plt.tight_layout()
        plt.savefig(self.save_dir / save_path, dpi=300)
        plt.close()

    # =========================
    # 3. 雷达图
    # =========================
    def plot_radar(self, save_path="radar.png"):
        import matplotlib.pyplot as plt
        import numpy as np

        labels = ["acc", "precision", "recall", "f1"]
        num_vars = len(labels)

        angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        plt.figure(figsize=(6,6))
        ax = plt.subplot(111, polar=True)

        for model_name, metrics in self.models.items():
            values = [metrics[l] for l in labels]
            values += values[:1]

            ax.plot(angles, values, label=model_name)
            ax.fill(angles, values, alpha=0.1)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(labels)

        plt.legend(loc="upper right")
        plt.title("Model Performance Radar")

        plt.tight_layout()
        plt.savefig(self.save_dir / save_path, dpi=300)
        plt.close()