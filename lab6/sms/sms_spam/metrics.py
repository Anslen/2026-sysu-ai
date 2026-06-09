from __future__ import annotations


def confusion_matrix(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> list[list[int]]:
    # Rows are true labels and columns are predicted labels.
    index = {label: i for i, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1
    return matrix


def accuracy_score(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def precision_recall_f1(
    y_true: list[str], y_pred: list[str], positive_label: str
) -> tuple[float, float, float]:
    tp = fp = fn = 0
    for t, p in zip(y_true, y_pred):
        if p == positive_label and t == positive_label:
            tp += 1
        elif p == positive_label and t != positive_label:
            fp += 1
        elif p != positive_label and t == positive_label:
            fn += 1
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
    return precision, recall, f1
