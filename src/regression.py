import numpy as np
import pandas as pd

from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt

from .preprocessing import smape
from .cv_utils import pick_random_1k


def get_xgb_regressor(random_state: int):
    """
    XGBoost varsa onu kullanır.
    Yoksa GradientBoostingRegressor'a düşer.
    """
    try:
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            n_jobs=-1
        ), "XGBoostRegressor"
    except Exception:
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(random_state=random_state), "GradientBoostingRegressor (fallback)"


def run_regression_cv(X, y, k_folds: int, random_state: int):
    """
    Models:
      - ANN (MLPRegressor) + scaler
      - XGBoostRegressor (or fallback)

    Returns:
      reg_summary_df, reg_folds_df, best_dict(for x=y plot)
    """
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)

    xgb, xgb_name = get_xgb_regressor(random_state)

    models = {
        "ANN (MLPRegressor)": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                max_iter=800,
                random_state=random_state
            ))
        ]),
        xgb_name: xgb
    }

    fold_rows = []
    summary_rows = []

    best = {"model": None, "fold": None, "mae": float("inf"), "y_true": None, "y_pred": None}

    for model_name, model in models.items():
        maes, smapes = [], []
        fold = 0

        for tr_idx, te_idx in kf.split(X):
            fold += 1
            Xtr, Xte = X[tr_idx], X[te_idx]
            ytr, yte = y[tr_idx], y[te_idx]

            model.fit(Xtr, ytr)
            pred = model.predict(Xte)

            mae = float(np.mean(np.abs(yte - pred)))
            s = float(smape(yte, pred))

            maes.append(mae)
            smapes.append(s)

            fold_rows.append({
                "Task": "Regression",
                "Model": model_name,
                "Fold": fold,
                "MAE": mae,
                "SMAPE": s
            })

            if mae < best["mae"]:
                best = {"model": model_name, "fold": fold, "mae": mae, "y_true": yte, "y_pred": pred}

        summary_rows.append({
            "Task": "Regression",
            "Model": model_name,
            "MAE_mean": float(np.mean(maes)),   "MAE_std": float(np.std(maes)),
            "SMAPE_mean": float(np.mean(smapes)),"SMAPE_std": float(np.std(smapes)),
        })

    return pd.DataFrame(summary_rows), pd.DataFrame(fold_rows), best


def save_xy_plot(best: dict, out_path: str, random_state: int, sample_n: int = 1000):
    y_true = np.asarray(best["y_true"]).reshape(-1)
    y_pred = np.asarray(best["y_pred"]).reshape(-1)

    y_true, y_pred = pick_random_1k(y_true, y_pred, seed=random_state, sample_n=sample_n)

    mn = float(min(y_true.min(), y_pred.min()))
    mx = float(max(y_true.max(), y_pred.max()))

    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6)
    ax.plot([mn, mx], [mn, mx], linewidth=2)
    ax.set_title(f"x=y Plot | {best['model']} | fold={best['fold']}")
    ax.set_xlabel("True (y)")
    ax.set_ylabel("Predicted (ŷ)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def save_residual_plot(best: dict, out_path: str, random_state: int, sample_n: int = 1000):
    y_true = np.asarray(best["y_true"]).reshape(-1)
    y_pred = np.asarray(best["y_pred"]).reshape(-1)

    y_true, y_pred = pick_random_1k(y_true, y_pred, seed=random_state, sample_n=sample_n)
    resid = y_true - y_pred

    fig, ax = plt.subplots()
    ax.scatter(y_true, resid, s=10, alpha=0.6)
    ax.axhline(0, linewidth=2)
    ax.set_title("Residual Plot (y - ŷ)")
    ax.set_xlabel("True (y)")
    ax.set_ylabel("Residual")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
