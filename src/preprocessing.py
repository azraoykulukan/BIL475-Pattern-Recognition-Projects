import numpy as np


def ensure_class_labels(y, max_classes: int = 20):

    y = np.asarray(y).reshape(-1)
    uniq = np.unique(y)

    if len(uniq) > max_classes:
        raise ValueError(
            f"Sınıflandırma etiketi çok fazla görünüyor ({len(uniq)}). "
            f"y yanlış seçilmiş olabilir."
        )

    # float ama aslında 0/1 gibi discrete ise int'e çevir
    if np.issubdtype(y.dtype, np.floating):
        if np.all(np.isclose(y, np.round(y))):
            y = np.round(y).astype(int)
    else:
        y = y.astype(int)

    return y


def smape(y_true, y_pred, eps: float = 1e-8) -> float:

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    denom = np.abs(y_true) + np.abs(y_pred) + eps
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)
