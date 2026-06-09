from __future__ import annotations

from collections import Counter
from pathlib import Path
from random import Random

from .data import load_sms_dataset
from .metrics import accuracy_score, confusion_matrix, precision_recall_f1
from .model import BagOfWordsVectorizer, MultinomialNaiveBayes
from .text import tokenize


def stratified_train_test_split(
    texts: list[str],
    labels: list[str],
    test_size: float = 0.2,
    seed: int = 42,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Split texts/labels with class proportions preserved in both folds.

    Indices are grouped by label, shuffled within each group, and the first
    ``test_size`` fraction of every group is allocated to the test fold so
    that ham/spam ratios match the full dataset.
    """
    rng = Random(seed)
    by_label: dict[str, list[int]] = {}
    for i, lab in enumerate(labels):
        by_label.setdefault(lab, []).append(i)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for lab, idxs in by_label.items():
        rng.shuffle(idxs)
        k = int(round(len(idxs) * test_size))
        test_idx.extend(idxs[:k])
        train_idx.extend(idxs[k:])

    # Re-shuffle the merged folds so adjacent samples come from mixed classes.
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    x_train = [texts[i] for i in train_idx]
    y_train = [labels[i] for i in train_idx]
    x_test = [texts[i] for i in test_idx]
    y_test = [labels[i] for i in test_idx]
    return x_train, x_test, y_train, y_test


def _print_confusion_matrix(matrix: list[list[int]], labels: list[str]) -> None:
    """Pretty-print a confusion matrix with header rows/cols."""
    width = max(max(len(str(v)) for row in matrix for v in row), 6)
    header = "true \\ pred  " + "  ".join(f"{lab:>{width}}" for lab in labels)
    print(header)
    for lab, row in zip(labels, matrix):
        cells = "  ".join(f"{v:>{width}}" for v in row)
        print(f"{lab:>11}  {cells}")


def _print_classification_report(
    y_true: list[str], y_pred: list[str], labels: list[str]
) -> None:
    """Sklearn-style per-class precision/recall/F1 plus macro/weighted averages."""
    n_total = len(y_true)
    counts = Counter(y_true)
    print(
        f"{'':>12}  {'precision':>10}  {'recall':>8}  {'f1-score':>9}  {'support':>8}"
    )
    p_sum = r_sum = f_sum = 0.0
    pw_sum = rw_sum = fw_sum = 0.0
    for lab in labels:
        p, r, f1 = precision_recall_f1(y_true, y_pred, positive_label=lab)
        support = counts[lab]
        p_sum += p
        r_sum += r
        f_sum += f1
        pw_sum += p * support
        rw_sum += r * support
        fw_sum += f1 * support
        print(f"{lab:>12}  {p:>10.4f}  {r:>8.4f}  {f1:>9.4f}  {support:>8d}")
    acc = accuracy_score(y_true, y_pred)
    n_labels = len(labels)
    print(f"{'accuracy':>12}  {'':>10}  {'':>8}  {acc:>9.4f}  {n_total:>8d}")
    print(
        f"{'macro avg':>12}  {p_sum / n_labels:>10.4f}  "
        f"{r_sum / n_labels:>8.4f}  {f_sum / n_labels:>9.4f}  {n_total:>8d}"
    )
    print(
        f"{'weighted avg':>12}  {pw_sum / n_total:>10.4f}  "
        f"{rw_sum / n_total:>8.4f}  {fw_sum / n_total:>9.4f}  {n_total:>8d}"
    )


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    dataset = load_sms_dataset(root / "spam.csv")

    full_counts = Counter(dataset.labels)
    n_total = len(dataset.texts)
    print("=" * 60)
    print(" 实验任务一：基于朴素贝叶斯的 SMS 垃圾邮件分类")
    print("=" * 60)
    print(f"[1/5] 加载数据: {root / 'spam.csv'}")
    print(
        f"      总样本数: {n_total} | "
        f"ham={full_counts['ham']} ({full_counts['ham'] / n_total:.2%}) | "
        f"spam={full_counts['spam']} ({full_counts['spam'] / n_total:.2%})"
    )

    # Build the full pipeline: stratified split -> tokenize -> vectorize -> train -> evaluate.
    print("[2/5] 分层抽样划分 训练 / 测试 (80% / 20%, seed=42)")
    x_train_raw, x_test_raw, y_train, y_test = stratified_train_test_split(
        dataset.texts, dataset.labels, test_size=0.2, seed=42
    )
    tr_counts = Counter(y_train)
    te_counts = Counter(y_test)
    print(
        f"      训练集: {len(x_train_raw)} 条 | "
        f"ham={tr_counts['ham']} | spam={tr_counts['spam']}"
    )
    print(
        f"      测试集: {len(x_test_raw)} 条 | "
        f"ham={te_counts['ham']} | spam={te_counts['spam']}"
    )

    print("[3/5] 词袋向量化 (tokenize + 训练集 fit)")
    x_train_tok = [tokenize(text) for text in x_train_raw]
    x_test_tok = [tokenize(text) for text in x_test_raw]
    vectorizer = BagOfWordsVectorizer()
    x_train = vectorizer.fit_transform(x_train_tok)
    x_test = vectorizer.transform(x_test_tok)
    print(f"      词表大小: {len(vectorizer.vocab_)}")

    print("[4/5] 训练多项式朴素贝叶斯 (alpha=1.0, Laplace smoothing)")
    model = MultinomialNaiveBayes(alpha=1.0)
    model.fit(x_train, y_train, vocab_size=len(vectorizer.vocab_))
    y_pred = model.predict(x_test)

    print("[5/5] 测试集评估")
    labels = ["ham", "spam"]
    matrix = confusion_matrix(y_test, y_pred, labels)
    print()
    print("混淆矩阵 (行=真实, 列=预测):")
    _print_confusion_matrix(matrix, labels)
    tn, fp = matrix[0]
    fn, tp = matrix[1]
    print(f"      真负 TN={tn} | 假正 FP={fp} | 假负 FN={fn} | 真正 TP={tp}")

    print()
    print("分类报告 (测试集):")
    _print_classification_report(y_test, y_pred, labels)
    return 0
