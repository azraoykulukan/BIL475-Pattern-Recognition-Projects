import os
import warnings
import numpy as np

from src.load_data import load_xy
from src.cv_utils import make_dirs, save_df, save_class_distribution, save_hist_1d
from src.classification import run_classification_cv, save_confusion_matrix
from src.regression import run_regression_cv, save_xy_plot, save_residual_plot

warnings.filterwarnings("ignore")

# =========================
# Paths
# =========================
BANKNOTE_MAT_PATH = os.path.join("data", "data_banknote_authentication.mat")
GASTURBINE_MAT_PATH = os.path.join("data", "Gas_Turbine_Co_NoX_2015.mat")

# =========================
# Keys (senin datasetlerin için kesin)
# =========================
BANKNOTE_X_KEY = "feat"
BANKNOTE_Y_KEY = "lbl"

GASTURBINE_X_KEY = "feat"
GASTURBINE_Y_KEY = "lbl1"   # istersen lbl2 yap

# =========================
# Settings
# =========================
RANDOM_STATE = 42
K_FOLDS = 3
RESULTS_DIR = "results"


def main():
    fig_dir, tab_dir = make_dirs(RESULTS_DIR)

    # --- Load
    Xc, yc, _ = load_xy(BANKNOTE_MAT_PATH, BANKNOTE_X_KEY, BANKNOTE_Y_KEY)
    Xr, yr, _ = load_xy(GASTURBINE_MAT_PATH, GASTURBINE_X_KEY, GASTURBINE_Y_KEY)

    print("Loaded datasets:")
    print(f"  Banknote: X={BANKNOTE_X_KEY} {Xc.shape}, y={BANKNOTE_Y_KEY} {yc.shape}, unique_y={np.unique(yc)[:10]}")
    print(f"  GasTurb: X={GASTURBINE_X_KEY} {Xr.shape}, y={GASTURBINE_Y_KEY} {yr.shape}")

    # --- OPTIONAL EDA (sunumun veri tanıtımı slaytı için iyi)
    save_class_distribution(yc, os.path.join(fig_dir, "banknote_class_dist.png"), "Banknote Class Distribution")
    save_hist_1d(Xc[:, 0], os.path.join(fig_dir, "banknote_feat0_hist.png"), "Banknote feat[0] histogram")

    save_hist_1d(yr, os.path.join(fig_dir, "gasturb_target_hist.png"), "Gas Turbine target histogram (lbl1)")
    save_hist_1d(Xr[:, 0], os.path.join(fig_dir, "gasturb_feat0_hist.png"), "Gas Turbine feat[0] histogram")

    # --- Run CV
    cls_summary, cls_folds, cls_best = run_classification_cv(Xc, yc, k_folds=K_FOLDS, random_state=RANDOM_STATE)
    reg_summary, reg_folds, reg_best = run_regression_cv(Xr, yr, k_folds=K_FOLDS, random_state=RANDOM_STATE)

    # --- Save tables
    save_df(cls_summary, os.path.join(tab_dir, "classification_summary.csv"))
    save_df(cls_folds, os.path.join(tab_dir, "classification_folds.csv"))
    save_df(reg_summary, os.path.join(tab_dir, "regression_summary.csv"))
    save_df(reg_folds, os.path.join(tab_dir, "regression_folds.csv"))

    # --- Required figures
    save_confusion_matrix(cls_best, os.path.join(fig_dir, "confusion_matrix.png"))
    save_xy_plot(reg_best, os.path.join(fig_dir, "xy_plot.png"), random_state=RANDOM_STATE, sample_n=1000)

    # --- Extra (sunum yorumunu güçlendirir)
    save_residual_plot(reg_best, os.path.join(fig_dir, "residual_plot.png"), random_state=RANDOM_STATE, sample_n=1000)

    # --- Print
    print("\n=== Classification (k=3) ===")
    print(cls_summary.to_string(index=False))

    print("\n=== Regression (k=3) ===")
    print(reg_summary.to_string(index=False))

    print("\nSaved outputs:")
    print(f"  Figures -> {fig_dir}")
    print(f"  Tables  -> {tab_dir}")


if __name__ == "__main__":
    main()
