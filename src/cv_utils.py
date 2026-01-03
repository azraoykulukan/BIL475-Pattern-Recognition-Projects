import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def make_dirs(results_dir: str):
    fig_dir = os.path.join(results_dir, "figures")
    tab_dir = os.path.join(results_dir, "tables")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    return fig_dir, tab_dir


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def pick_random_1k(y_true, y_pred, seed: int, sample_n: int = 1000):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    n = len(y_true)
    if n <= sample_n:
        return y_true, y_pred

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=sample_n, replace=False)
    return y_true[idx], y_pred[idx]


def save_class_distribution(y, out_path: str, title: str = "Class Distribution"):
    y = np.asarray(y).reshape(-1)
    vals, cnts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots()
    ax.bar([str(v) for v in vals], cnts)
    ax.set_title(title)
    ax.set_xlabel("Class")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_hist_1d(x, out_path: str, title: str, bins: int = 30):
    x = np.asarray(x).reshape(-1)
    fig, ax = plt.subplots()
    ax.hist(x, bins=bins)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
