# analyze_lines.py
import json
import math
from pathlib import Path
from typing import Dict, Any, Tuple

import cv2
import numpy as np
import matplotlib.pyplot as plt


CLASS_NAMES = ["fate", "head", "heart", "life"]


# ---------------------------
# Utility geometriche
# ---------------------------
def _largest_component(mask_bin: np.ndarray) -> np.ndarray:
    """Mantiene solo la componente con area massima (per evitare rumore)."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin, connectivity=8)
    if num_labels <= 1:
        return mask_bin

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_idx = 1 + np.argmax(areas)
    return (labels == max_idx).astype(np.uint8)


def _compute_line_geometry(mask_bin: np.ndarray) -> Dict[str, float]:
    """
    Calcola area, lunghezza approssimata, spessore, orientazione, curvatura.
    mask_bin: (H,W) 0/1
    """
    h, w = mask_bin.shape
    area = float(mask_bin.sum())
    if area < 10:  # troppo piccola, consideriamo assente
        return {
            "present": False,
            "area": 0.0,
            "length": 0.0,
            "thickness": 0.0,
            "orientation_deg": 0.0,
            "curvature": 0.0,
        }

    # Prendiamo solo la componente principale
    comp = _largest_component(mask_bin)

    ys, xs = np.where(comp > 0)
    coords = np.stack([xs, ys], axis=1)  # (N,2)

    if coords.shape[0] < 10:
        return {
            "present": False,
            "area": area,
            "length": 0.0,
            "thickness": 0.0,
            "orientation_deg": 0.0,
            "curvature": 0.0,
        }

    # Lunghezza approssimata = distanza massima tra due pixel della componente
    max_len = 0.0
    for i in range(0, coords.shape[0], max(1, coords.shape[0] // 200)):
        dx = coords[:, 0] - coords[i, 0]
        dy = coords[:, 1] - coords[i, 1]
        dist2 = dx * dx + dy * dy
        max_len = max(max_len, float(dist2.max()))
    length = math.sqrt(max_len)

    # Spessore medio ≈ area / lunghezza
    thickness = float(area / (length + 1e-6))

    # PCA per orientazione principale
    mean = coords.mean(axis=0)
    centered = coords - mean
    cov = centered.T @ centered / (coords.shape[0] + 1e-6)
    eigvals, eigvecs = np.linalg.eig(cov)
    main_vec = eigvecs[:, np.argmax(eigvals)]
    angle_rad = math.atan2(main_vec[1], main_vec[0])
    angle_deg = math.degrees(angle_rad)

    # Curvatura approssimata = deviazione media dal fit lineare principale
    # Proiezione dei punti sulla retta principale
    proj = centered @ main_vec
    recon = np.outer(proj, main_vec)  # punti ricostruiti sulla retta
    residuals = centered - recon
    curvature = float(np.sqrt((residuals ** 2).sum(axis=1).mean()))

    return {
        "present": True,
        "area": float(area),
        "length": float(length),
        "thickness": float(thickness),
        "orientation_deg": float(angle_deg),
        "curvature": float(curvature),
    }


def _categorize_z(z: float, low_th: float = 0.5, high_th: float = 1.0, labels=("bassa", "nella media", "alta")) -> str:
    """
    Trasforma uno z-score in etichetta qualitativa.
    """
    if not np.isfinite(z):
        return "n.d."

    if z < -high_th:
        return f"molto {labels[0]}"
    if z < -low_th:
        return labels[0]
    if z > high_th:
        return f"molto {labels[2]}"
    if z > low_th:
        return labels[2]
    return labels[1]


def _compare_with_stats(
    cls: str,
    geom: Dict[str, float],
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Confronta le metriche della linea con le statistiche del dataset (se presenti).
    stats ci aspettiamo qualcosa tipo:
    stats["classes"]["fate"]["length_mean"], etc.
    ma rendiamo tutto robusto a chiavi mancanti.
    """
    out = {}

    classes_stats = stats.get("classes", {})
    cls_stats = classes_stats.get(cls, {})

    def _z(value_key: str, mean_key: str, std_key: str) -> Tuple[float, float, str]:
        val = float(geom.get(value_key, 0.0))
        mean = float(cls_stats.get(mean_key, 0.0))
        std = float(cls_stats.get(std_key, 0.0) or 1.0)
        if std <= 0:
            return val, 0.0, "n.d."
        z = (val - mean) / std
        return val, z, _categorize_z(z)

    # lunghezza
    length, z_len, cat_len = _z("length", "length_mean", "length_std")
    # curvatura
    curvature, z_curv, cat_curv = _z("curvature", "curvature_mean", "curvature_std")
    # spessore
    thickness, z_thick, cat_thick = _z("thickness", "thickness_mean", "thickness_std")

    out["length"] = {
        "value": length,
        "z_score": z_len,
        "category": cat_len,  # corta / media / lunga
    }
    out["curvature"] = {
        "value": curvature,
        "z_score": z_curv,
        "category": cat_curv,  # poco / media / molto curva
    }
    out["thickness"] = {
        "value": thickness,
        "z_score": z_thick,
        "category": cat_thick,
    }

    return out


def _compute_intersections(masks_bin: np.ndarray) -> Dict[str, Any]:
    """
    Calcola intersezioni tra classi diverse usando leggero dilate.
    masks_bin: (H, W, C) 0/1
    """
    H, W, C = masks_bin.shape
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    inter_dict: Dict[str, Any] = {}

    for i in range(C):
        for j in range(i + 1, C):
            mi = masks_bin[:, :, i]
            mj = masks_bin[:, :, j]
            if mi.sum() == 0 or mj.sum() == 0:
                count = 0
            else:
                di = cv2.dilate(mi, kernel)
                dj = cv2.dilate(mj, kernel)
                inter = (di & dj).astype(np.uint8)
                count = int(inter.sum())

            key = f"{CLASS_NAMES[i]}-{CLASS_NAMES[j]}"
            inter_dict[key] = {
                "pixel_overlap": count,
            }

    return inter_dict


def _make_radar_plot(per_class_geom: Dict[str, Dict[str, float]], out_dir: Path) -> str:
    """
    Crea un radar chart semplice basato sulla lunghezza normalizzata (0-1) per classe.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # valori di lunghezza per classe
    values = []
    labels = []
    for cls in CLASS_NAMES:
        geom = per_class_geom.get(cls, {})
        length = float(geom.get("length", 0.0))
        values.append(length)
        labels.append(cls)

    max_val = max(values) if any(v > 0 for v in values) else 1.0
    norm_values = [v / max_val for v in values]
    norm_values.append(norm_values[0])  # chiusura
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, norm_values, "o-", linewidth=2)
    ax.fill(angles, norm_values, alpha=0.25)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels)
    ax.set_title("Profilo lunghezze linee (normalizzate)", pad=20)
    ax.set_ylim(0, 1.0)

    out_path = out_dir / "radar.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

    return out_path.name


def _build_overall_summary(per_class: Dict[str, Any]) -> Dict[str, Any]:
    """
    Costruisce una sintesi testuale “interpretativa” ad alto livello.
    """
    summary = {}

    for cls in CLASS_NAMES:
        cls_info = per_class.get(cls, {})
        geom = cls_info.get("geometry", {})
        comp = cls_info.get("comparison", {})

        if not geom.get("present", False):
            summary[cls] = f"La linea {cls} risulta poco o per nulla evidente."
            continue

        # lunghezza
        len_cat = comp.get("length", {}).get("category", "n.d.")
        curv_cat = comp.get("curvature", {}).get("category", "n.d.")
        thick_cat = comp.get("thickness", {}).get("category", "n.d.")
        angle = geom.get("orientation_deg", 0.0)

        parts = []
        # lunghezza
        if "corta" in len_cat:
            parts.append("piuttosto corta")
        elif "molto lunga" in len_cat:
            parts.append("molto lunga")
        elif "lunga" in len_cat:
            parts.append("abbastanza lunga")
        else:
            parts.append("di lunghezza nella media")

        # curvatura
        if "molto" in curv_cat and "alta" in curv_cat:
            parts.append("molto ondulata/mossa")
        elif "alta" in curv_cat:
            parts.append("abbastanza ondulata")
        elif "bassa" in curv_cat:
            parts.append("piuttosto dritta")
        else:
            parts.append("con curvatura nella norma")

        # spessore
        if "molto" in thick_cat and "alta" in thick_cat:
            parts.append("molto marcata (spessa)")
        elif "alta" in thick_cat:
            parts.append("ben marcata")
        elif "bassa" in thick_cat:
            parts.append("sottile")
        else:
            parts.append("di spessore medio")

        parts.append(f"orientata circa a {angle:.0f}°.")

        summary[cls] = f"La linea {cls} è " + ", ".join(parts)

    return summary


# ---------------------------
# MAIN ENTRY
# ---------------------------
def analyze_prediction(
    pred_masks_path: Path,
    stats_json_path: Path,
    out_dir: Path,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Legge le maschere predette (npy) e le confronta col file di statistiche.
    Ritorna:
      {
        "per_class": {...},
        "intersections": {...},
        "overall_summary": {...},
        "radar_image": "radar.png"
      }
    """
    pred_masks_path = Path(pred_masks_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    masks = np.load(pred_masks_path)

    # 🔧 Fix automatico shape maschere
    # Se arrivano come (C,H,W), trasponiamo a (H,W,C)
    if masks.ndim == 3 and masks.shape[0] == 4:
        masks = np.transpose(masks, (1, 2, 0))

    # Ora ci aspettiamo (H,W,4)
    if masks.ndim != 3 or masks.shape[2] != 4:
        raise ValueError(f"Maschere con shape inattesa: {masks.shape}, attese (H,W,4)")

    if masks.ndim != 3 or masks.shape[2] != len(CLASS_NAMES):
        raise ValueError(f"Maschere con shape inattesa: {masks.shape}, attese (H,W,4)")

    # binarizza
    masks_bin = (masks > threshold).astype(np.uint8)  # (H,W,4)

    # carica statistiche dataset (robusto a strutture leggermente diverse)
    with open(stats_json_path, "r") as f:
        stats = json.load(f)

    per_class: Dict[str, Any] = {}

    # Geometria + confronto per classe
    for idx, cls in enumerate(CLASS_NAMES):
        geom = _compute_line_geometry(masks_bin[:, :, idx])
        if geom["present"]:
            comp = _compare_with_stats(cls, geom, stats)
        else:
            comp = {}

        per_class[cls] = {
            "geometry": geom,
            "comparison": comp,
        }

    # Intersezioni tra classi
    intersections = _compute_intersections(masks_bin)

    # Radar
    geom_only = {cls: per_class[cls]["geometry"] for cls in CLASS_NAMES}
    radar_name = _make_radar_plot(geom_only, out_dir)

    # Sintesi interpretativa
    overall_summary = _build_overall_summary(per_class)

    return {
        "per_class": per_class,
        "intersections": intersections,
        "overall_summary": overall_summary,
        "radar_image": radar_name,
    }


if __name__ == "__main__":
    # Test manuale (non usato da run_pipeline)
    import sys
    if len(sys.argv) < 3:
        print(json.dumps({"error": "Usage: python analyze_lines.py <pred_masks.npy> <stats.json>"}))
        raise SystemExit(1)

    pred_p = Path(sys.argv[1])
    stats_p = Path(sys.argv[2])
    out = Path("debug_analysis")

    res = analyze_prediction(pred_p, stats_p, out)
    print(json.dumps(res["overall_summary"], indent=2, ensure_ascii=False))
