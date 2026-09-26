import os
import math
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from lightglue import SuperPoint


# ============================================================
# 0. 執行設定
# ============================================================
#
# 修正後務必：
#   1. BUILD_TILE_CACHE = True   → 重建「只標最後一碼」的語意 index
#   2. TRAIN_SEMANTIC_MODEL = True → 用與 test2.py 對齊的 TinyCNN 重訓
#
# A  TinyCNN 只看最後一碼（線上推論對齊，結構以此檔為準）
# E  Cache 語意只標個位數框，0 / 6 不進入 semantic index
# ============================================================

BUILD_TILE_CACHE = False
TRAIN_SEMANTIC_MODEL = True

DEVICE = "cpu"
NUM_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CORES)

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# 1. 路徑
# ============================================================

IMAGE_PATH = BASE_DIR / "world_image" / "parking_lot.jpg"
MASK_PATH = BASE_DIR / "world_image" / "auto_vehicle_mask.png"
CACHE_PATH = BASE_DIR / "world_image" / "tile_feature_cache.npz"
VIS_DIR = BASE_DIR / "world_image" / "tile_features_vis"

SEMANTIC_MODEL_DIR = BASE_DIR / "semantic_model"
SEMANTIC_MODEL_PATH = SEMANTIC_MODEL_DIR / "semantic_classifier.pth"
SEMANTIC_META_PATH = SEMANTIC_MODEL_DIR / "semantic_classifier_meta.npz"
SEMANTIC_PREVIEW_DIR = SEMANTIC_MODEL_DIR / "synthetic_preview"
SEMANTIC_BASE_DIGIT_DIR = SEMANTIC_MODEL_DIR / "base_digits"
DIGIT_REGION_PREVIEW_PATH = SEMANTIC_MODEL_DIR / "map_digit_regions.jpg"

SEMANTIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. 地圖 / Tile
# ============================================================

MAP_WIDTH_PX = 2985
MAP_HEIGHT_PX = 1730

ANGLES = list(range(0, 360, 30))

GRID_ROWS = 8
GRID_COLS = 8
TILE_OVERLAP = 0.30

MAX_KPTS_PER_TILE = 800
DETECTION_THRESHOLD = 0.005
NMS_RADIUS = 3


# ============================================================
# 3. 車格
# ============================================================

PARKING_NUMBER_REGIONS = {
    "063": (1337, 479, 1615, 564),
    "064": (1078, 479, 1337, 564),
    "065": (809, 479, 1078, 564),
    "066": (546, 479, 809, 564),
    "067": (281, 479, 546, 564),
    "068": (9, 479, 278, 564)
}

PARKING_MATCH_REGIONS = {
    "063": (1337, 430, 1615, 650),
    "064": (1078, 430, 1337, 650),
    "065": (809, 430, 1078, 650),
    "066": (546, 430, 809, 650),
    "067": (281, 430, 546, 650),
    "068": (9, 455, 278, 610)
}

PARKING_NUMBERS = ["063", "064", "065", "066", "067", "068"]

# 最後一碼框較小，2 個 keypoint 即可標進語意 index
MIN_SEMANTIC_KPTS_PER_TILE = 2


# ============================================================
# 4. 航線
# ============================================================

ROUTES = {
    "1": {"name": "68號車格航線", "waypoints": [(2130, 200), (2130, 550), (60, 550)]},
    "2": {"name": "70號車格航線", "waypoints": [(2130, 200), (2130, 550), (1270, 550), (1270, 1350)]}
}

ROUTE_FEATURE_MAX_DISTANCE_PX = 500.0
PROGRESS_LOW_PERCENTILE = 5
PROGRESS_HIGH_PERCENTILE = 95


# ============================================================
# 5. Digit TinyCNN & 旋轉數據增強 / 自適應幾何切分
# ============================================================

SEMANTIC_CLASSES = ["3", "4", "5", "6", "7", "8"]

PARKING_TO_DIGIT = {
    "063": "3",
    "064": "4",
    "065": "5",
    "066": "6",
    "067": "7",
    "068": "8"
}

DIGIT_IMAGE_SIZE = 96

DIGIT_SAMPLES_PER_CLASS = 1200
SEMANTIC_BATCH_SIZE = 32
SEMANTIC_EPOCHS = 50
SEMANTIC_LEARNING_RATE = 1e-3
SEMANTIC_VAL_RATIO = 0.20
SEMANTIC_EARLY_STOP_PATIENCE = 8
SEMANTIC_SEED = 42

SAVE_SYNTHETIC_PREVIEW = True
PREVIEW_COUNT_PER_CLASS = 24

random.seed(SEMANTIC_SEED)
np.random.seed(SEMANTIC_SEED)
torch.manual_seed(SEMANTIC_SEED)


# ------------------------------------------------------------
# 5.1 旋轉數據增強 (針對 TinyCNN 訓練與預處理)
# ------------------------------------------------------------
def augment_digit_rotation(img_gray):
    """
    對輸入的單字元灰階影像進行 0°, 90°, 180°, 270° 的隨機直角旋轉增強，
    確保 TinyCNN 具備旋轉不變性。
    """
    rot_code = random.choice([None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE])
    if rot_code is not None:
        return cv2.rotate(img_gray, rot_code)
    return img_gray


# ------------------------------------------------------------
# 5.2 車格幾何區域自適應切分
# ------------------------------------------------------------
def fallback_last_digit_regions(parking_regions):
    """
    根據車格區域長寬比自動判定排列方向：
    - 若高 > 寬 (垂直排列)：切分 Y 軸下方 40% 作為最後一碼區域
    - 若寬 >= 高 (水平排列)：切分 X 軸右側 40% 作為最後一碼區域
    """
    regions = {}
    for number, (x1, y1, x2, y2) in parking_regions.items():
        w, h = x2 - x1, y2 - y1
        if h > w:  # 垂直排列：取 Y 軸下方 40%
            regions[number] = (x1, int(y1 + 0.60 * h), x2, y2)
        else:      # 水平排列：取 X 軸右側 40%
            regions[number] = (int(x1 + 0.60 * w), y1, x2, y2)
    return regions


# ------------------------------------------------------------
# 5.3 TinyCNN 模型架構
# ------------------------------------------------------------
class TinyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super(TinyCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 7 * 7, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# SemanticTinyCNN 別名以維持與舊版本的相容性
SemanticTinyCNN = TinyCNN


# ============================================================
# 6. 車輛遮罩
# ============================================================

def load_vehicle_mask(force_recreate=True):
    if force_recreate and MASK_PATH.exists():
        MASK_PATH.unlink()

    if MASK_PATH.exists():
        return cv2.imread(str(MASK_PATH), cv2.IMREAD_GRAYSCALE)

    img = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"找不到母圖：{IMAGE_PATH}")

    h, w = img.shape
    mask = np.zeros((h, w), dtype=np.uint8)

    regions = [
        (0.06, 0.00, 0.36, 0.25), (0.44, 0.00, 0.53, 0.25),
        (0.26, 0.80, 0.36, 1.00), (0.45, 0.67, 0.55, 0.98),
        (0.63, 0.68, 0.72, 0.98), (0.72, 0.66, 0.81, 0.98),
        (0.82, 0.68, 0.91, 0.98), (0.91, 0.68, 0.99, 0.98)
    ]

    for x1, y1, x2, y2 in regions:
        cv2.rectangle(mask, (int(w * x1), int(h * y1)), (int(w * x2), int(h * y2)), 255, -1)

    MASK_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(MASK_PATH), mask)

    return mask


# ============================================================
# 7. 地圖旋轉
# ============================================================

def rotate_full(img, vehicle_mask, angle):
    h, w = img.shape[:2]

    if angle == 0:
        valid_mask = np.ones((h, w), dtype=np.uint8) * 255
        return img, vehicle_mask, valid_mask, np.eye(3, dtype=np.float32)

    cx, cy = w / 2.0, h / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)

    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    new_w = int(h * sin_v + w * cos_v)
    new_h = int(h * cos_v + w * sin_v)

    M[0, 2] += new_w / 2.0 - cx
    M[1, 2] += new_h / 2.0 - cy

    rot_img = cv2.warpAffine(img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=0)
    rot_vmask = cv2.warpAffine(vehicle_mask, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0) if vehicle_mask is not None else None
    valid_mask = cv2.warpAffine(np.ones((h, w), dtype=np.uint8) * 255, M, (new_w, new_h), flags=cv2.INTER_NEAREST, borderValue=0)

    M_homo = np.eye(3, dtype=np.float32)
    M_homo[:2] = M

    return rot_img, rot_vmask, valid_mask, np.linalg.inv(M_homo).astype(np.float32)


# ============================================================
# 8. SuperPoint
# ============================================================

def extract_superpoint(img_patch, extractor):
    tensor = torch.from_numpy(img_patch).float()[None, None].to(DEVICE) / 255.0

    with torch.inference_mode():
        pred = extractor({"image": tensor})

    kpts = pred["keypoints"][0].cpu().numpy().astype(np.float32)
    descs = pred["descriptors"][0].cpu().numpy().astype(np.float32)
    scores = pred["keypoint_scores"][0].cpu().numpy().astype(np.float32) if "keypoint_scores" in pred else np.ones(len(kpts), dtype=np.float32)

    return kpts, descs, scores


def local_to_global_0deg(kpts_local, x1, y1, M_inv):
    kpts_rot = kpts_local.copy()
    kpts_rot[:, 0] += x1
    kpts_rot[:, 1] += y1

    homo = np.hstack([kpts_rot, np.ones((len(kpts_rot), 1), dtype=np.float32)])
    global_homo = (M_inv @ homo.T).T
    kpts_0deg = global_homo[:, :2] / global_homo[:, 2:3]

    return kpts_rot.astype(np.float32), kpts_0deg.astype(np.float32)


# ============================================================
# 9. Cache Semantic（E：只標最後一碼）
# ============================================================

def point_in_box(x, y, box):
    x1, y1, x2, y2 = box
    return (x >= x1) and (x <= x2) and (y >= y1) and (y <= y2)


def compute_feature_semantic_labels(kpts_0deg, last_digit_regions):
    """
    只把落在「個位數」框內的點標成 063~068。
    前兩碼 0 / 6 的點保持 0，不進入 semantic_{number}_aXXX index。
    """
    labels = np.zeros(len(kpts_0deg), dtype=np.int16)

    for number, box in last_digit_regions.items():
        x1, y1, x2, y2 = box
        mask = (
            (kpts_0deg[:, 0] >= x1) & (kpts_0deg[:, 0] <= x2) &
            (kpts_0deg[:, 1] >= y1) & (kpts_0deg[:, 1] <= y2)
        )
        labels[mask] = int(number)

    return labels


def compute_prefix_digit_mask(kpts_0deg, prefix_digit_regions):
    mask = np.zeros(len(kpts_0deg), dtype=bool)

    for x1, y1, x2, y2 in prefix_digit_regions:
        mask |= (
            (kpts_0deg[:, 0] >= x1) & (kpts_0deg[:, 0] <= x2) &
            (kpts_0deg[:, 1] >= y1) & (kpts_0deg[:, 1] <= y2)
        )

    return mask


def get_tile_parking_numbers(labels):
    return [number for number in PARKING_NUMBERS if np.sum(labels == int(number)) >= MIN_SEMANTIC_KPTS_PER_TILE]


# ============================================================
# 10. Route
# ============================================================

def project_point_to_route(point, waypoints):
    xP, yP = point
    best_dist, best_progress, cumulative = float("inf"), 0.0, 0.0

    for i in range(len(waypoints) - 1):
        xA, yA = waypoints[i]
        xB, yB = waypoints[i + 1]

        dx, dy = xB - xA, yB - yA
        seg_len = math.hypot(dx, dy)

        if seg_len < 1e-8:
            continue

        t = ((xP - xA) * dx + (yP - yA) * dy) / (seg_len * seg_len)
        t = float(np.clip(t, 0.0, 1.0))

        qx, qy = xA + t * dx, yA + t * dy
        dist = math.hypot(xP - qx, yP - qy)

        if dist < best_dist:
            best_dist = dist
            best_progress = cumulative + t * seg_len

        cumulative += seg_len

    return float(best_progress), float(best_dist)


def compute_feature_route_data(kpts_0deg, waypoints):
    progresses = np.zeros(len(kpts_0deg), dtype=np.float32)
    distances = np.zeros(len(kpts_0deg), dtype=np.float32)

    for i, point in enumerate(kpts_0deg):
        progresses[i], distances[i] = project_point_to_route(point, waypoints)

    return progresses, distances


def get_tile_progress_range(progresses, distances):
    valid = distances <= ROUTE_FEATURE_MAX_DISTANCE_PX

    if not np.any(valid):
        return -1.0, -1.0

    values = progresses[valid]
    return float(np.percentile(values, PROGRESS_LOW_PERCENTILE)), float(np.percentile(values, PROGRESS_HIGH_PERCENTILE))


# ============================================================
# 11. Tile Cache
# ============================================================

def build_tile_feature_cache():
    print("\n🚀 建立SuperPoint Tile Cache（語意只標最後一碼）")

    img_raw = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if img_raw is None:
        raise FileNotFoundError(f"找不到母圖：{IMAGE_PATH}")

    map_h, map_w = img_raw.shape[:2]

    if (map_w, map_h) != (MAP_WIDTH_PX, MAP_HEIGHT_PX):
        raise ValueError(f"母圖尺寸錯誤：{map_w}×{map_h}")

    last_digit_regions, prefix_digit_regions = build_map_digit_regions(img_raw)
    save_digit_region_preview(img_raw, last_digit_regions, prefix_digit_regions)
    save_digit_region_meta(last_digit_regions, prefix_digit_regions)

    vehicle_mask = load_vehicle_mask(force_recreate=True)
    extractor = SuperPoint(max_num_keypoints=MAX_KPTS_PER_TILE, detection_threshold=DETECTION_THRESHOLD, nms_radius=NMS_RADIUS).eval().to(DEVICE)

    cache = {}
    semantic_index = {number: {angle: [] for angle in ANGLES} for number in PARKING_NUMBERS}
    total_tiles = 0

    for angle in ANGLES:
        print(f"⚙️ Angle {angle:03d}°")

        rot_img, rot_vmask, valid_mask, M_inv = rotate_full(img_raw, vehicle_mask, angle)
        rot_h, rot_w = rot_img.shape[:2]

        valid_eroded = cv2.erode(valid_mask, np.ones((7, 7), np.uint8), iterations=2)
        rot_vmask_dilated = cv2.dilate(rot_vmask, np.ones((11, 11), np.uint8), iterations=2) if rot_vmask is not None else None

        rot_masked = rot_img.copy()

        if rot_vmask_dilated is not None:
            rot_masked[rot_vmask_dilated > 128] = 0

        tile_w, tile_h = rot_w / GRID_COLS, rot_h / GRID_ROWS
        pad_w, pad_h = tile_w * TILE_OVERLAP, tile_h * TILE_OVERLAP

        valid_tile_ids = []

        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                tile_id = r * GRID_COLS + c + 1

                x1 = max(0, int(c * tile_w - pad_w))
                y1 = max(0, int(r * tile_h - pad_h))
                x2 = min(rot_w, int((c + 1) * tile_w + pad_w))
                y2 = min(rot_h, int((r + 1) * tile_h + pad_h))

                patch = rot_masked[y1:y2, x1:x2]

                if patch.size == 0 or cv2.countNonZero(patch) == 0:
                    continue

                kpts_local, descs, scores = extract_superpoint(patch, extractor)

                if len(kpts_local) < 4:
                    continue

                gx = np.rint(kpts_local[:, 0] + x1).astype(np.int32)
                gy = np.rint(kpts_local[:, 1] + y1).astype(np.int32)

                inside = (gx >= 0) & (gx < rot_w) & (gy >= 0) & (gy < rot_h)
                keep = np.zeros(len(kpts_local), dtype=bool)
                valid_ids = np.flatnonzero(inside)

                if len(valid_ids):
                    valid_keep = valid_eroded[gy[valid_ids], gx[valid_ids]] > 128

                    if rot_vmask_dilated is not None:
                        valid_keep &= rot_vmask_dilated[gy[valid_ids], gx[valid_ids]] == 0

                    keep[valid_ids] = valid_keep

                kpts_local, descs, scores = kpts_local[keep], descs[keep], scores[keep]

                if len(kpts_local) < 4:
                    continue

                kpts_rot, kpts_0deg = local_to_global_0deg(kpts_local, x1, y1, M_inv)

                valid = (
                    (kpts_0deg[:, 0] >= 0) & (kpts_0deg[:, 0] <= MAP_WIDTH_PX) &
                    (kpts_0deg[:, 1] >= 0) & (kpts_0deg[:, 1] <= MAP_HEIGHT_PX)
                )

                kpts_local, kpts_rot, kpts_0deg = kpts_local[valid], kpts_rot[valid], kpts_0deg[valid]
                descs, scores = descs[valid], scores[valid]

                if len(kpts_local) < 4:
                    continue

                semantic_labels = compute_feature_semantic_labels(kpts_0deg, last_digit_regions)
                prefix_mask = compute_prefix_digit_mask(kpts_0deg, prefix_digit_regions)
                tile_numbers = get_tile_parking_numbers(semantic_labels)

                route1_progress, route1_distance = compute_feature_route_data(kpts_0deg, ROUTES["1"]["waypoints"])
                route2_progress, route2_distance = compute_feature_route_data(kpts_0deg, ROUTES["2"]["waypoints"])

                prefix = f"a{angle:03d}_t{tile_id:02d}"

                cache[f"{prefix}_kpts_local"] = kpts_local
                cache[f"{prefix}_kpts_rot"] = kpts_rot
                cache[f"{prefix}_kpts_0deg"] = kpts_0deg
                cache[f"{prefix}_descs"] = descs
                cache[f"{prefix}_scores"] = scores
                cache[f"{prefix}_size"] = np.array([patch.shape[1], patch.shape[0]], dtype=np.float32)
                cache[f"{prefix}_semantic_labels"] = semantic_labels
                cache[f"{prefix}_prefix_digit_mask"] = prefix_mask.astype(np.uint8)
                cache[f"{prefix}_parking_numbers"] = np.array(tile_numbers, dtype="<U3")

                cache[f"{prefix}_route1_progress"] = route1_progress
                cache[f"{prefix}_route1_distance"] = route1_distance
                cache[f"{prefix}_route1_range"] = np.array(get_tile_progress_range(route1_progress, route1_distance), dtype=np.float32)

                cache[f"{prefix}_route2_progress"] = route2_progress
                cache[f"{prefix}_route2_distance"] = route2_distance
                cache[f"{prefix}_route2_range"] = np.array(get_tile_progress_range(route2_progress, route2_distance), dtype=np.float32)

                valid_tile_ids.append(tile_id)
                total_tiles += 1

                for number in tile_numbers:
                    semantic_index[number][angle].append(tile_id)

        cache[f"angle_{angle:03d}_tile_ids"] = np.array(valid_tile_ids, dtype=np.int32)

    for number in PARKING_NUMBERS:
        for angle in ANGLES:
            cache[f"semantic_{number}_a{angle:03d}"] = np.array(semantic_index[number][angle], dtype=np.int32)

    cache["parking_numbers"] = np.array(PARKING_NUMBERS, dtype="<U3")
    cache["angles"] = np.array(ANGLES, dtype=np.int32)
    cache["map_size"] = np.array([MAP_WIDTH_PX, MAP_HEIGHT_PX], dtype=np.int32)
    cache["last_digit_only_semantic"] = np.array([1], dtype=np.int32)
    cache["digit_image_size"] = np.array([DIGIT_IMAGE_SIZE], dtype=np.int32)

    for number, box in last_digit_regions.items():
        cache[f"last_digit_region_{number}"] = np.array(box, dtype=np.int32)

    if prefix_digit_regions:
        cache["prefix_digit_regions"] = np.array(prefix_digit_regions, dtype=np.int32)
    else:
        cache["prefix_digit_regions"] = np.zeros((0, 4), dtype=np.int32)

    np.savez_compressed(CACHE_PATH, **cache)

    print(f"✅ Tile Cache建立完成｜總Tile={total_tiles}｜語意=最後一碼 only")


# ============================================================
# 12. 從完整063~068找三個字
# ============================================================

def make_digit_binary(gray):
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    enhanced = clahe.apply(gray)

    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))

    return binary


def find_three_digit_boxes(gray):
    h, w = gray.shape

    binary = make_digit_binary(gray)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, 8)

    components = []

    for i in range(1, count):
        x, y, bw, bh, area = stats[i]

        if bh < max(8, int(h * 0.25)) or bh > int(h * 0.98):
            continue

        if bw < 3 or bw > int(w * 0.35):
            continue

        if area < 18:
            continue

        aspect = bw / max(float(bh), 1.0)

        if aspect < 0.08 or aspect > 1.25:
            continue

        components.append((x, y, bw, bh, area))

    components.sort(key=lambda box: box[0])

    best_boxes = None
    best_score = -1.0

    for i in range(len(components)):
        for j in range(i + 1, len(components)):
            for k in range(j + 1, len(components)):
                boxes = [components[i], components[j], components[k]]

                heights = np.array([b[3] for b in boxes], dtype=np.float32)
                centers_y = np.array([b[1] + b[3] / 2.0 for b in boxes], dtype=np.float32)

                mean_h = float(np.mean(heights))
                height_similarity = float(np.min(heights) / max(np.max(heights), 1.0))
                alignment = max(0.0, 1.0 - float(np.ptp(centers_y)) / max(mean_h * 0.50, 1.0))

                gap1 = boxes[1][0] - (boxes[0][0] + boxes[0][2])
                gap2 = boxes[2][0] - (boxes[1][0] + boxes[1][2])

                if gap1 < -mean_h * 0.20 or gap2 < -mean_h * 0.20:
                    continue

                gap_similarity = min(abs(gap1) + 1.0, abs(gap2) + 1.0) / max(abs(gap1) + 1.0, abs(gap2) + 1.0)

                total_x1 = boxes[0][0]
                total_x2 = boxes[2][0] + boxes[2][2]
                total_width = total_x2 - total_x1

                if total_width < w * 0.20:
                    continue

                score = 0.50 * height_similarity + 0.35 * alignment + 0.15 * gap_similarity

                if score > best_score:
                    best_score = score
                    best_boxes = boxes

    return best_boxes, best_score, binary


# ============================================================
# 13. 取最後一碼
# ============================================================

def crop_box_with_margin(gray, box, margin_ratio=0.28):
    x, y, w, h, _ = box

    margin = int(round(max(w, h) * margin_ratio))

    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(gray.shape[1], x + w + margin)
    y2 = min(gray.shape[0], y + h + margin)

    return gray[y1:y2, x1:x2].copy()


def extract_last_digit_from_map(source_map, parking_number):
    x1, y1, x2, y2 = PARKING_NUMBER_REGIONS[parking_number]

    number_crop = source_map[y1:y2, x1:x2].copy()

    boxes, score, binary = find_three_digit_boxes(number_crop)

    if boxes is not None:
        digit_crop = crop_box_with_margin(number_crop, boxes[-1], 0.30)
        return digit_crop, number_crop, binary, boxes, score

    # 偵測失敗時使用自適應幾何切分 fallback
    fallback_map = fallback_last_digit_regions({parking_number: (0, 0, number_crop.shape[1], number_crop.shape[0])})
    fx1, fy1, fx2, fy2 = fallback_map[parking_number]
    digit_crop = number_crop[fy1:fy2, fx1:fx2].copy()

    return digit_crop, number_crop, binary, None, 0.0


def box_to_global(box, origin_x, origin_y, crop_w, crop_h, margin_ratio=0.12):
    x, y, w, h, _ = box
    margin = int(round(max(w, h) * margin_ratio))

    x1 = origin_x + max(0, x - margin)
    y1 = origin_y + max(0, y - margin)
    x2 = origin_x + min(crop_w, x + w + margin)
    y2 = origin_y + min(crop_h, y + h + margin)

    return (int(x1), int(y1), int(x2), int(y2))


def build_map_digit_regions(source_map):
    """
    E：在母圖 0° 座標系切出每個車格的
      - last_digit_regions[number] = 個位數框
      - prefix_digit_regions = 所有 0 / 6（前兩碼）框
    """
    last_digit_regions = {}
    prefix_digit_regions = []

    print("\n🔢 建立地圖最後一碼 / 前兩碼框")

    # 使用自適應幾何切分做為全域 fallback
    fallback_regions = fallback_last_digit_regions(PARKING_NUMBER_REGIONS)

    for parking_number in PARKING_NUMBERS:
        rx1, ry1, rx2, ry2 = PARKING_NUMBER_REGIONS[parking_number]
        crop = source_map[ry1:ry2, rx1:rx2]
        crop_h, crop_w = crop.shape[:2]

        boxes, score, _binary = find_three_digit_boxes(crop)

        if boxes is not None:
            last_digit_regions[parking_number] = box_to_global(
                boxes[-1], rx1, ry1, crop_w, crop_h, 0.18
            )
            for box in boxes[:2]:
                prefix_digit_regions.append(
                    box_to_global(box, rx1, ry1, crop_w, crop_h, 0.10)
                )
            print(
                f" ├─ {parking_number} 最後一碼框={last_digit_regions[parking_number]}｜"
                f"三字偵測成功｜Score={score:.3f}"
            )
        else:
            # 採用自適應幾何區域切分
            last_digit_regions[parking_number] = fallback_regions[parking_number]
            fx1, fy1, _, _ = fallback_regions[parking_number]
            prefix_digit_regions.append((rx1, ry1, fx1, ry2))
            print(f" ├─ {parking_number} 三字偵測失敗，改用自適應幾何區域切分")

    return last_digit_regions, prefix_digit_regions


def save_digit_region_preview(source_map, last_digit_regions, prefix_digit_regions):
    vis = cv2.cvtColor(source_map, cv2.COLOR_GRAY2BGR)

    for x1, y1, x2, y2 in prefix_digit_regions:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 220), 2)

    for number, (x1, y1, x2, y2) in last_digit_regions.items():
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(
            vis, number, (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 0), 2
        )

    SEMANTIC_MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(DIGIT_REGION_PREVIEW_PATH), vis)
    print(f"✅ 數字框預覽：{DIGIT_REGION_PREVIEW_PATH}")


def save_digit_region_meta(last_digit_regions, prefix_digit_regions):
    payload = {
        "classes": np.array(SEMANTIC_CLASSES),
        "digit_image_size": np.array([DIGIT_IMAGE_SIZE], dtype=np.int32),
        "last_digit_only_semantic": np.array([1], dtype=np.int32),
        "prefix_digit_regions": (
            np.array(prefix_digit_regions, dtype=np.int32)
            if prefix_digit_regions else
            np.zeros((0, 4), dtype=np.int32)
        )
    }

    for number, box in last_digit_regions.items():
        payload[f"last_digit_region_{number}"] = np.array(box, dtype=np.int32)

    np.savez(SEMANTIC_META_PATH, **payload)
    print(f"✅ 數字框 meta：{SEMANTIC_META_PATH}")


# ============================================================
# 14. Digit轉96×96
# ============================================================

def prepare_digit_image(img, target_size=DIGIT_IMAGE_SIZE):
    if img is None or img.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    h, w = img.shape[:2]

    side = max(h, w)
    background = int(np.median(img))

    canvas = np.full((side, side), background, dtype=np.uint8)

    x = (side - w) // 2
    y = (side - h) // 2

    canvas[y:y + h, x:x + w] = img

    interpolation = cv2.INTER_AREA if side > target_size else cv2.INTER_CUBIC

    return cv2.resize(canvas, (target_size, target_size), interpolation=interpolation)


# ============================================================
# 15. 建立基礎Digit
# ============================================================

def build_base_digit_crops(source_map):
    if SEMANTIC_BASE_DIGIT_DIR.exists():
        shutil.rmtree(SEMANTIC_BASE_DIGIT_DIR)

    SEMANTIC_BASE_DIGIT_DIR.mkdir(parents=True, exist_ok=True)

    base_digits = {}

    print("\n🔎 從063~068擷取最後一碼")

    for parking_number in PARKING_NUMBERS:
        digit = PARKING_TO_DIGIT[parking_number]

        digit_crop, full_crop, binary, boxes, score = extract_last_digit_from_map(source_map, parking_number)

        base_digits[digit] = digit_crop

        folder = SEMANTIC_BASE_DIGIT_DIR / digit
        folder.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(folder / f"{parking_number}_full.jpg"), full_crop)
        cv2.imwrite(str(folder / f"{parking_number}_binary.jpg"), binary)
        cv2.imwrite(str(folder / f"{parking_number}_digit.jpg"), prepare_digit_image(digit_crop))

        print(f" ├─ {parking_number} → Digit {digit}｜三字偵測={'成功' if boxes is not None else 'Fallback'}｜Score={score:.3f}")

    return base_digits


# ============================================================
# 16. Digit資料增強 (含旋轉與仿射等幾何增強)
# ============================================================

def augment_digit_image(base_img, rng):
    img = prepare_digit_image(base_img)

    # 套用隨機直角旋轉數據增強 (0°, 90°, 180°, 270°)
    if rng.random() < 0.25:
        img = augment_digit_rotation(img)

    angle = float(rng.uniform(-10.0, 10.0))
    scale = float(rng.uniform(0.78, 1.15))
    tx = float(rng.uniform(-0.08, 0.08) * DIGIT_IMAGE_SIZE)
    ty = float(rng.uniform(-0.08, 0.08) * DIGIT_IMAGE_SIZE)

    center = (DIGIT_IMAGE_SIZE / 2.0, DIGIT_IMAGE_SIZE / 2.0)

    M = cv2.getRotationMatrix2D(center, angle, scale)
    M[0, 2] += tx
    M[1, 2] += ty

    background = int(np.median(img))

    img = cv2.warpAffine(
        img, M, (DIGIT_IMAGE_SIZE, DIGIT_IMAGE_SIZE),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=background
    )

    alpha = float(rng.uniform(0.72, 1.28))
    beta = float(rng.uniform(-28, 28))

    img = np.clip(img.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)

    if rng.random() < 0.22:
        img = cv2.GaussianBlur(img, (3, 3), 0)

    if rng.random() < 0.10:
        k = int(rng.choice([3, 5]))
        kernel = np.zeros((k, k), dtype=np.float32)

        if rng.random() < 0.5:
            kernel[k // 2, :] = 1.0 / k
        else:
            kernel[:, k // 2] = 1.0 / k

        img = cv2.filter2D(img, -1, kernel)

    if rng.random() < 0.18:
        sigma = float(rng.uniform(1.0, 7.0))
        noise = rng.normal(0, sigma, img.shape).astype(np.float32)
        img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    if rng.random() < 0.12:
        kernel = np.ones((2, 2), np.uint8)

        if rng.random() < 0.5:
            img = cv2.dilate(img, kernel, iterations=1)
        else:
            img = cv2.erode(img, kernel, iterations=1)

    if rng.random() < 0.12:
        quality = int(rng.integers(55, 92))
        ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])

        if ok:
            img = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)

    return img


def semantic_image_to_tensor(img):
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    return torch.from_numpy(img).unsqueeze(0)


# ============================================================
# 17. Dataset
# ============================================================

class SyntheticDigitDataset(Dataset):
    def __init__(self, base_digits, samples, training=True):
        self.base_digits = base_digits
        self.samples = samples
        self.training = training

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        class_index, base_seed = self.samples[index]
        class_name = SEMANTIC_CLASSES[class_index]

        seed = base_seed + random.randint(0, 1_000_000) if self.training else base_seed
        rng = np.random.default_rng(seed)

        img = augment_digit_image(self.base_digits[class_name], rng)

        return semantic_image_to_tensor(img), torch.tensor(class_index, dtype=torch.long)


# ============================================================
# 18. Train / Validation
# ============================================================

def build_synthetic_samples():
    train_samples, val_samples = [], []

    print("\n📂 建立Digit資料")

    for class_index, class_name in enumerate(SEMANTIC_CLASSES):
        train_count = int(DIGIT_SAMPLES_PER_CLASS * (1.0 - SEMANTIC_VAL_RATIO))
        val_count = DIGIT_SAMPLES_PER_CLASS - train_count

        for i in range(train_count):
            train_samples.append((class_index, SEMANTIC_SEED + class_index * 100000 + i))

        for i in range(val_count):
            val_samples.append((class_index, SEMANTIC_SEED + 9_000_000 + class_index * 100000 + i))

        print(f" ├─ Digit {class_name}：Train={train_count}｜Val={val_count}")

    random.shuffle(train_samples)
    random.shuffle(val_samples)

    return train_samples, val_samples


def save_synthetic_preview(base_digits):
    if not SAVE_SYNTHETIC_PREVIEW:
        return

    if SEMANTIC_PREVIEW_DIR.exists():
        shutil.rmtree(SEMANTIC_PREVIEW_DIR)

    SEMANTIC_PREVIEW_DIR.mkdir(parents=True, exist_ok=True)

    print("\n🖼️ 產生Digit Preview")

    for class_index, class_name in enumerate(SEMANTIC_CLASSES):
        folder = SEMANTIC_PREVIEW_DIR / class_name
        folder.mkdir(parents=True, exist_ok=True)

        for i in range(PREVIEW_COUNT_PER_CLASS):
            rng = np.random.default_rng(SEMANTIC_SEED + class_index * 1000 + i)
            img = augment_digit_image(base_digits[class_name], rng)

            cv2.imwrite(str(folder / f"digit_{class_name}_{i + 1:02d}.jpg"), img)

    print(f"✅ Preview完成：{SEMANTIC_PREVIEW_DIR}")


# ============================================================
# 19. Validation
# ============================================================

def evaluate_semantic_model(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    with torch.inference_mode():
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            logits = model(images)
            loss = criterion(logits, labels)

            total_loss += loss.item() * images.size(0)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_count += images.size(0)

    return total_loss / max(total_count, 1), total_correct / max(total_count, 1)


# ============================================================
# 20. 離線索引建立主流程
# ============================================================

def build_offline_index(map_image_path, parking_regions):
    """
    讀取離線地圖，建立各車格整體區域與最後一碼 ROI 索引資訊
    """
    map_img = cv2.imread(str(map_image_path), cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"無法載入地圖影像: {map_image_path}")

    last_digit_regions = fallback_last_digit_regions(parking_regions)
    indexed_data = {}

    for num, (x1, y1, x2, y2) in last_digit_regions.items():
        crop = map_img[y1:y2, x1:x2]
        indexed_data[num] = {
            "full_box": parking_regions[num],
            "last_digit_roi": (x1, y1, x2, y2),
            "crop": crop
        }

    print(f"[Offline Indexing] 成功建置 {len(indexed_data)} 組車格索引資訊。")
    return indexed_data


# ============================================================
# 21. 訓練
# ============================================================

def train_semantic_classifier():
    print("\n" + "=" * 75)
    print("🧠 Digit TinyCNN")
    print("🎯 Classes = 3 / 4 / 5 / 6 / 7 / 8  （只吃最後一碼 crop）")
    print("=" * 75)

    source_map = cv2.imread(str(IMAGE_PATH), cv2.IMREAD_GRAYSCALE)

    if source_map is None:
        raise FileNotFoundError(f"找不到母圖：{IMAGE_PATH}")

    last_digit_regions, prefix_digit_regions = build_map_digit_regions(source_map)
    save_digit_region_preview(source_map, last_digit_regions, prefix_digit_regions)

    base_digits = build_base_digit_crops(source_map)
    save_synthetic_preview(base_digits)

    train_samples, val_samples = build_synthetic_samples()

    train_dataset = SyntheticDigitDataset(base_digits, train_samples, training=True)
    val_dataset = SyntheticDigitDataset(base_digits, val_samples, training=False)

    train_loader = DataLoader(train_dataset, batch_size=SEMANTIC_BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=SEMANTIC_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TinyCNN(len(SEMANTIC_CLASSES)).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=SEMANTIC_LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    best_acc = -1.0
    patience = 0

    print(f"\n📊 Train={len(train_dataset)}｜Validation={len(val_dataset)}")

    for epoch in range(1, SEMANTIC_EPOCHS + 1):
        model.train()

        total_loss = 0.0
        correct = 0
        count = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += images.size(0)

        train_loss = total_loss / max(count, 1)
        train_acc = correct / max(count, 1)

        val_loss, val_acc = evaluate_semantic_model(model, val_loader, criterion)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch:03d}｜Train Loss={train_loss:.4f}｜Train Acc={train_acc * 100:.1f}%｜"
            f"Val Loss={val_loss:.4f}｜Val Acc={val_acc * 100:.1f}%"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            patience = 0

            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": SEMANTIC_CLASSES,
                "digit_image_size": DIGIT_IMAGE_SIZE,
                "image_size": DIGIT_IMAGE_SIZE,
                "model_type": "DigitTinyCNN_v2_last_digit",
                "parking_to_digit": PARKING_TO_DIGIT,
                "pool": "7x7",
                "architecture": "TinyCNN"
            }, SEMANTIC_MODEL_PATH)

            save_digit_region_meta(last_digit_regions, prefix_digit_regions)

            print(f"   ✅ 儲存最佳模型｜Val={best_acc * 100:.1f}%")

        else:
            patience += 1

        if patience >= SEMANTIC_EARLY_STOP_PATIENCE:
            print("🛑 Validation長時間沒有改善，提前停止")
            break

    print("\n✅ Digit TinyCNN訓練完成")
    print(f"🎯 最佳Synthetic Validation：{best_acc * 100:.2f}%")
    print(f"📦 模型：{SEMANTIC_MODEL_PATH}")


# ============================================================
# 22. 主程式
# ============================================================

def main():
    print("\n" + "=" * 75)
    print("🚀 UAV VPS 離線資料建立系統")
    print("🔧 v2：語意只標最後一碼｜TinyCNN 與 test2.py 結構對齊")
    print("=" * 75)

    # 測試邊界切分範例
    sample_regions = {
        "101": (50, 100, 120, 320),  # 垂直排列 (h=220 > w=70)
        "102": (200, 100, 420, 160)   # 水平排列 (w=220 > h=60)
    }
    adaptive_regions = fallback_last_digit_regions(sample_regions)
    print("自適應切分測試結果:", adaptive_regions)

    print(f"Build Tile Cache：{BUILD_TILE_CACHE}")
    print(f"Train Digit TinyCNN：{TRAIN_SEMANTIC_MODEL}")
    print(f"設備：{DEVICE}")

    if not BUILD_TILE_CACHE:
        print("⚠️ 修正後建議至少跑一次 BUILD_TILE_CACHE=True，否則線上語意 index 仍含 0/6")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(f"找不到母圖：{IMAGE_PATH}")

    if BUILD_TILE_CACHE:
        build_tile_feature_cache()
    else:
        print("\n⏭️ 跳過Tile Cache建立")
        print(f"{'✅' if CACHE_PATH.exists() else '⚠️'} Cache：{CACHE_PATH}")

    if TRAIN_SEMANTIC_MODEL:
        train_semantic_classifier()
    else:
        print("\n⏭️ 跳過Digit TinyCNN訓練")

    print("\n" + "=" * 75)
    print("✅ 離線工作完成")
    print(f"📦 Tile Cache：{CACHE_PATH}")
    print(f"🧠 Digit TinyCNN：{SEMANTIC_MODEL_PATH}")
    print(f"🔢 Base Digit：{SEMANTIC_BASE_DIGIT_DIR}")
    print(f"🖼️ Preview：{SEMANTIC_PREVIEW_DIR}")
    print(f"🟩 數字框：{DIGIT_REGION_PREVIEW_PATH}")
    print("=" * 75)


if __name__ == "__main__":
    main()