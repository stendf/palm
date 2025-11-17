# run_pipeline.py
import json
import shutil
from datetime import datetime
from pathlib import Path

from predict_lines import load_model, predict_lines
from analyze_lines import analyze_prediction
from generate_report import make_summary_image


# ===============================
# CONFIG (Render-friendly paths)
# ===============================
MODEL_PATH = Path("models/best_unetpp_resnet50_resume_20251116_193959_dice0.799.pth")
STATS_PATH = Path("data/palm_line_stats.json")
RESULTS_ROOT = Path("results")

# cache modello
_model_cache = None


def load_cached_model(device="cpu"):
    global _model_cache
    if _model_cache is None:
        _model_cache = load_model(MODEL_PATH, device=device)
    return _model_cache


# ===============================
# FUNZIONE API (usata da server.py)
# ===============================
def run_pipeline_api(img_path: Path):
    """
    Esegue la pipeline completa e restituisce il JSON (senza print).
    Usata da Flask su Render.
    """
    img_path = Path(img_path)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_ROOT / f"run_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    original_name = img_path.name
    shutil.copy(str(img_path), str(run_dir / original_name))

    model = load_cached_model(device="cpu")

    pred_info = predict_lines(
        model=model,
        input_image_path=img_path,
        out_dir=run_dir,
        device="cpu",
        threshold=0.5
    )

    analysis = analyze_prediction(
        pred_masks_path=run_dir / pred_info["pred_masks"],
        stats_json_path=STATS_PATH,
        out_dir=run_dir
    )

    summary_name = make_summary_image(
        out_dir=run_dir,
        original_name=original_name,
        pred_name=pred_info["pred_overlay"],
        radar_name=analysis["radar_image"],
        summary_name="summary.png"
    )

    full_json = {
        "run_dir": str(run_dir),
        "images": {
            "original": original_name,
            "prediction_overlay": pred_info["pred_overlay"],
            "radar": analysis["radar_image"],
            "summary": summary_name
        },
        "stats": {
            "per_class": analysis["per_class"],
            "intersections": analysis["intersections"],
            "overall_summary": analysis["overall_summary"]
        }
    }

    # salva anche in disco come prima
    with open(run_dir / "palm_line_eval.json", "w") as f:
        json.dump(full_json, f, indent=2, ensure_ascii=False)

    return full_json


# ===============================
# CLI VERSION (PER USO MANUALE)
# ===============================
def main():
    import sys

    if len(sys.argv) < 2:
        print(json.dumps({"error": "Usage: python run_pipeline.py <input_image>"}))
        return

    img_path = Path(sys.argv[1])

    result = run_pipeline_api(img_path)

    # versione ridotta per il terminale
    min_json = {
        "run_dir": result["run_dir"],
        "images": result["images"]
    }

    print(json.dumps(min_json))


if __name__ == "__main__":
    main()
