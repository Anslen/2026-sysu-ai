from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Dataset:
    texts: list[str]
    labels: list[str]


def load_sms_dataset(path: str | Path) -> Dataset:
    texts: list[str] = []
    labels: list[str] = []

    # The dataset is stored in latin-1 and the useful fields are in the first two columns.
    with Path(path).open("r", encoding="latin-1", newline="") as fh:
        reader = csv.reader(fh)
        header = next(reader, None)
        if not header:
            raise ValueError("CSV 文件为空")

        for row in reader:
            if len(row) < 2:
                continue
            # Column 0 is the label and column 1 is the SMS content.
            label = row[0].strip()
            text = row[1].strip()
            if label and text:
                labels.append(label)
                texts.append(text)

    if not texts:
        raise ValueError("没有读取到有效样本")

    return Dataset(texts=texts, labels=labels)
