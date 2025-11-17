# predict_lines.py
import cv2
import torch
import numpy as np
from pathlib import Path
import segmentation_models_pytorch as smp
from skimage.morphology import skeletonize
from collections import deque

IMG_SIZE = 1024
N_CLASSES = 4
CLASS_NAMES = ["fate", "head", "heart", "life"]
CLASS_COLORS = [
    (255,   0,   0),   # fate  -> rosso
    (  0, 255,   0),   # head  -> verde
    (  0,   0, 255),   # heart -> blu
    (255, 255,   0),   # life  -> giallo
]


# -------------------------------------------------
# 1) Estrarre SOLO il percorso più lungo per classe
# -------------------------------------------------
def _longest_path_on_skeleton(comp: np.ndarray):
    sk = skeletonize(comp > 0).astype(np.uint8)
    coords = np.argwhere(sk > 0)  # (N, 2)

    if coords.shape[0] < 2:
        return sk, int(coords.shape[0])

    idx_map = {(int(r), int(c)): i for i, (r, c) in enumerate(coords)}
    n = len(coords)

    neighbors = [[] for _ in range(n)]
    for i, (r, c) in enumerate(coords):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = int(r + dr), int(c + dc)
                j = idx_map.get((rr, cc))
                if j is not None:
                    neighbors[i].append(j)

    def bfs(start_idx):
        dist = [-1] * n
        parent = [-1] * n
        q = deque([start_idx])
        dist[start_idx] = 0
        while q:
            u = q.popleft()
            for v in neighbors[u]:
                if dist[v] == -1:
                    dist[v] = dist[u] + 1
                    parent[v] = u
                    q.append(v)
        far = max(range(n), key=lambda i: dist[i])
        return far, dist, parent

    u0 = 0
    u, _, _ = bfs(u0)
    v, dist, parent = bfs(u)

    path_nodes = []
    cur = v
    while cur != -1:
        path_nodes.append(cur)
        if cur == u:
            break
        cur = parent[cur]

    path_mask = np.zeros_like(comp, dtype=np.uint8)
    for idx in path_nodes:
        r, c = coords[idx]
        path_mask[int(r), int(c)] = 1

    return path_mask, len(path_nodes)


def _keep_single_longest_line_per_class(masks_hw4: np.ndarray) -> np.ndarray:
    H, W, C = masks_hw4.shape
    out = np.zeros_like(masks_hw4, dtype=np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    for c in range(C):
        m = masks_hw4[:, :, c].astype(np.uint8)
        if m.sum() == 0:
            continue

        num_labels, labels = cv2.connectedComponents(m)
        if num_labels <= 1:
            path_mask, _ = _longest_path_on_skeleton(m)
            out[:, :, c] = cv2.dilate(path_mask, kernel, iterations=1)
            continue

        best_len = -1
        best_path = None

        for lab in range(1, num_labels):
            comp = (labels == lab).astype(np.uint8)
            if comp.sum() < 10:
                continue
            path_mask, plen = _longest_path_on_skeleton(comp)
            if plen > best_len:
                best_len = plen
                best_path = path_mask

        if best_path is not None:
            out[:, :, c] = cv2.dilate(best_path, kernel, iterations=1)

    return out


# -------------------------------------------------
# 2) Modello UNet++ + ResNet50
# -------------------------------------------------
def build_model(device: str = "cpu") -> torch.nn.Module:
    model = smp.UnetPlusPlus(
        encoder_name="resnet50",
        encoder_weights=None,
        in_channels=3,
        classes=N_CLASSES,
        activation=None,
    )
    model.to(device)
    model.eval()
    return model


def load_model(model_path: Path, device: str = "cpu") -> torch.nn.Module:
    model = build_model(device=device)
    state = torch.load(str(model_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


# -------------------------------------------------
# 3) Preprocess immagine
# -------------------------------------------------
def _preprocess_image(image_path: Path, device: str = "cpu"):
    img_bgr = cv2.imread(str(image_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Immagine non trovata: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))

    inp = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
    inp = torch.from_numpy(inp).unsqueeze(0).to(device)

    return inp, img_resized, img_rgb


# -------------------------------------------------
# 4) Overlay a colori
# -------------------------------------------------
def _make_overlay(base_rgb: np.ndarray, masks_hw4: np.ndarray) -> np.ndarray:
    overlay = base_rgb.astype(np.float32) / 255.0

    for c in range(N_CLASSES):
        mask = masks_hw4[:, :, c]
        if mask.sum() == 0:
            continue

        color = np.array(CLASS_COLORS[c], dtype=np.float32) / 255.0
        color_mask = np.zeros_like(overlay)

        for i in range(3):
            color_mask[..., i] = mask * color[i]

        overlay = cv2.addWeighted(overlay, 1.0, color_mask, 0.7, 0)

    return (overlay * 255).clip(0, 255).astype(np.uint8)


# -------------------------------------------------
# 5) Predizione completa
# -------------------------------------------------
def predict_lines(
    model: torch.nn.Module,
    input_image_path: Path,
    out_dir: Path,
    device: str = "cpu",
    threshold: float = 0.5,
) -> dict:

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inp, img_resized, img_orig = _preprocess_image(input_image_path, device=device)

    with torch.no_grad():
        logits = model(inp)                 # [1,4,H,W]
        probs = torch.sigmoid(logits)[0]    # [4,H,W]
        probs_np = probs.cpu().numpy()

    masks_bin = (probs_np > threshold).astype(np.uint8)
    masks_hw4 = np.transpose(masks_bin, (1, 2, 0))

    # 🔥 una sola linea per classe
    masks_hw4 = _keep_single_longest_line_per_class(masks_hw4)

    # 🔥 RESIZE alle dimensioni ORIGINALI dell'immagine (FIX DISTORSIONE)
    orig_h, orig_w = img_orig.shape[:2]
    masks_hw4 = np.stack([
        cv2.resize(masks_hw4[:, :, c].astype(np.uint8), (orig_w, orig_h))
        for c in range(4)
    ], axis=2)

    # salva maschere
    masks_path = out_dir / "pred_masks.npy"
    np.save(masks_path, masks_hw4)

    # overlay senza distorsione
    overlay = _make_overlay(img_orig, masks_hw4)
    overlay_path = out_dir / "pred_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return {
        "pred_masks": masks_path.name,
        "pred_overlay": overlay_path.name,
    }
