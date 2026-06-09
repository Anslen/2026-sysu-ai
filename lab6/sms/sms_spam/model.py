from __future__ import annotations

import math
from collections import Counter, defaultdict


class BagOfWordsVectorizer:
    def __init__(self) -> None:
        self.vocab_: dict[str, int] = {}

    def fit(self, texts: list[list[str]]) -> "BagOfWordsVectorizer":
        vocab = {}
        for tokens in texts:
            for token in tokens:
                if token not in vocab:
                    # Assign each new token a stable integer index.
                    vocab[token] = len(vocab)
        self.vocab_ = vocab
        return self

    def transform(self, texts: list[list[str]]) -> list[dict[int, int]]:
        vectors: list[dict[int, int]] = []
        for tokens in texts:
            counts = Counter(tokens)
            # Keep only vocabulary words and store them as sparse count vectors.
            vectors.append(
                {
                    self.vocab_[tok]: cnt
                    for tok, cnt in counts.items()
                    if tok in self.vocab_
                }
            )
        return vectors

    def fit_transform(self, texts: list[list[str]]) -> list[dict[int, int]]:
        return self.fit(texts).transform(texts)


class MultinomialNaiveBayes:
    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha
        self.classes_: list[str] = []
        self.class_log_prior_: dict[str, float] = {}
        self.feature_log_prob_: dict[str, dict[int, float]] = {}
        self.default_log_prob_: dict[str, float] = {}
        self.vocab_size_: int = 0

    def fit(
        self, X: list[dict[int, int]], y: list[str], vocab_size: int
    ) -> "MultinomialNaiveBayes":
        self.vocab_size_ = vocab_size
        class_counts = Counter(y)
        total_docs = len(y)
        # Class priors are computed from document frequency.
        self.classes_ = sorted(class_counts)
        self.class_log_prior_ = {
            c: math.log(class_counts[c] / total_docs) for c in self.classes_
        }

        token_totals: dict[str, Counter[int]] = defaultdict(Counter)
        total_per_class = Counter()
        for feats, label in zip(X, y):
            for idx, cnt in feats.items():
                token_totals[label][idx] += cnt
                total_per_class[label] += cnt

        for c in self.classes_:
            # Laplace smoothing avoids zero probability for unseen tokens.
            denom = total_per_class[c] + self.alpha * vocab_size
            self.default_log_prob_[c] = math.log(self.alpha / denom)
            self.feature_log_prob_[c] = {
                idx: math.log((token_totals[c][idx] + self.alpha) / denom)
                for idx in token_totals[c]
            }

        return self

    def predict_one(self, feats: dict[int, int]) -> str:
        best_class = None
        best_score = float("-inf")
        for c in self.classes_:
            # Work in log space to avoid numerical underflow.
            score = self.class_log_prior_[c]
            default = self.default_log_prob_[c]
            feature_log_prob = self.feature_log_prob_[c]
            for idx, cnt in feats.items():
                score += cnt * feature_log_prob.get(idx, default)
            if score > best_score:
                best_score = score
                best_class = c
        if best_class is None:
            raise RuntimeError("分类失败")
        return best_class

    def predict(self, X: list[dict[int, int]]) -> list[str]:
        return [self.predict_one(feats) for feats in X]
