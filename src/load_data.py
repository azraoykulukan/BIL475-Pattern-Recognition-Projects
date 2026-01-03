import os
import numpy as np
import scipy.io as sio


def load_xy(mat_path: str, x_key: str, y_key: str):
    """
    .mat dosyasından X ve y'yi çeker.
    Senin veri setlerinde keys:
      - Banknote: feat, lbl
      - Gas Turbine: feat, lbl1, lbl2
    """
    if not os.path.exists(mat_path):
        raise FileNotFoundError(f"Dosya bulunamadı: {mat_path}")

    d = sio.loadmat(mat_path)
    d = {k: v for k, v in d.items() if not k.startswith("__")}

    if x_key not in d or y_key not in d:
        raise KeyError(f"Anahtar bulunamadı. Mevcut keys: {list(d.keys())}")

    X = np.asarray(d[x_key])
    y = np.asarray(d[y_key]).reshape(-1)
    return X, y, d


def mat_overview(d: dict):
    """Debug için: .mat key ve shape yazdırmak istersen."""
    out = []
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            out.append(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            out.append(f"{k}: type={type(v)}")
    return "\n".join(out)
