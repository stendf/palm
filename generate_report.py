# generate_report.py
from pathlib import Path
import cv2
import numpy as np


def make_summary_image(
    out_dir: Path,
    original_name: str,
    pred_name: str,
    radar_name: str,
    summary_name: str = "summary.png"
) -> str:
    """
    Crea un'immagine riassuntiva affiancando:
    - originale
    - overlay predizione
    - radar
    """
    orig_path = out_dir / original_name
    pred_path = out_dir / pred_name
    radar_path = out_dir / radar_name

    orig = cv2.imread(str(orig_path))
    pred = cv2.imread(str(pred_path))
    radar = cv2.imread(str(radar_path))

    imgs = [img for img in [orig, pred, radar] if img is not None]
    if not imgs:
        return ""

    # porta tutto alla stessa altezza
    h_min = min(img.shape[0] for img in imgs)
    resized = []
    for img in imgs:
        ratio = h_min / img.shape[0]
        new_w = int(img.shape[1] * ratio)
        resized.append(cv2.resize(img, (new_w, h_min)))

    summary = np.hstack(resized)
    out_path = out_dir / summary_name
    cv2.imwrite(str(out_path), summary)
    return summary_name
