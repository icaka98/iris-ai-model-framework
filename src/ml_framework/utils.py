import os
from typing import Any, Dict, Tuple

import evaluate
import numpy as np

STORAGE_PATH = os.path.join("..", "..", "storage")

ID_2_LABEL = {0: "ph", 1: "math", 2: "cs", 3: "bio", 4: "soc", 5: "chem"}
LABEL_2_ID = {v: k for k, v in ID_2_LABEL.items()}


def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
    metric_1 = evaluate.load("accuracy")
    metric_2 = evaluate.load("precision")
    metric_3 = evaluate.load("recall")
    metric_4 = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    accuracy = metric_1.compute(predictions=predictions, references=labels)["accuracy"]
    precision = metric_2.compute(
        predictions=predictions, references=labels, average="weighted"
    )["precision"]
    recall = metric_3.compute(
        predictions=predictions, references=labels, average="weighted"
    )["recall"]
    f1 = metric_4.compute(
        predictions=predictions, references=labels, average="weighted"
    )["f1"]

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
