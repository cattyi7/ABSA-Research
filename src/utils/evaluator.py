from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


class Evaluator:
    """
    统一评估器
    """

    def __init__(self, average="weighted"):
        self.average = average

    def evaluate(self, y_true, y_pred):
        """
        返回统一 metrics dict
        """

        acc = accuracy_score(y_true, y_pred)

        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average=self.average,
            zero_division=0
        )

        return {
            "acc": float(acc),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1)
        }

    def detailed_report(self, y_true, y_pred):
        """
        分类详细报告
        """
        return classification_report(y_true, y_pred, digits=4)