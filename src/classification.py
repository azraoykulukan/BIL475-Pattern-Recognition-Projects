import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

from .preprocessing import ensure_class_labels


def run_classification_cv(X, y, k_folds: int, random_state: int):
    """
    Models:
      - SVM (RBF) + scaler
      - kNN (k=5) + scaler

    Returns:
      cls_summary_df, cls_folds_df, best_dict(for confusion matrix)
    """
    y = ensure_class_labels(y)

    skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    models = {
        "SVM (RBF)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ]),
        "kNN (k=5)": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier(n_neighbors=5))
        ]),
    }

    fold_rows = []
    summary_rows = []

    best = {"model": None, "fold": None, "f1": -1, "y_true": None, "y_pred": None}

    for model_name, model in models.items():
        accs, f1s = [], []
        fold = 0

        for tr_idx, te_idx in skf.split(X, y):
            fold += 1
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            acc = float(accuracy_score(yte, pred))
            f1 = float(f1_score(yte, pred, average="binary"))

            accs.append(acc)
            f1s.append(f1)

            fold_rows.append({
                "Task": "Classification",
                "Model": model_name,
                "Fold": fold,
                "ACC": acc,
                "F1": f1
            })

            if f1 > best["f1"]:
                best = {"model": model_name, "fold": fold, "f1": f1, "y_true": yte, "y_pred": pred}

        summary_rows.append({
            "Task": "Classification",
            "Model": model_name,
            "ACC_mean": float(np.mean(accs)), "ACC_std": float(np.std(accs)),
            "F1_mean": float(np.mean(f1s)),  "F1_std": float(np.std(f1s)),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), best


def save_confusion_matrix(best: dict, out_path: str):
    cm = confusion_matrix(best["y_true"], best["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)

    fig, ax = plt.subplots()
    disp.plot(ax=ax, values_format="d")
    ax.set_title(f"Confusion Matrix | {best['model']} | fold={best['fold']}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
