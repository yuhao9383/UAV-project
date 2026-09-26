import os
import glob
import math
import time
import socket
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np
import torch
import torch.nn as nn
from lightglue import SuperPoint, LightGlue


# ============================================================
# 0. 系統設定 & 網路傳送設定
# ============================================================

DEVICE = torch.device("cpu")
NUM_CORES = os.cpu_count() or 4
torch.set_num_threads(NUM_CORES)

UDP_IP = "192.168.4.1"
UDP_PORT = 5007
ENABLE_UDP_SEND = True

BASE_DIR = Path(__file__).resolve().parent

MAP_IMAGE_PATH = BASE_DIR / "world_image" / "parking_lot.jpg"
CACHE_PATH = BASE_DIR / "world_image" / "tile_feature_cache.npz"
SEMANTIC_MODEL_PATH = BASE_DIR / "semantic_model" / "semantic_classifier.pth"
SEMANTIC_META_PATH = BASE_DIR / "semantic_model" / "semantic_classifier_meta.npz"

DATA_DIR = Path(r"C:\Users\howar\Desktop\photo_test\data\s13")

OUTPUT_IMAGE_DIR = BASE_DIR / "output_results" / "output_image"
MATCH_DATA_DIR = BASE_DIR / "output_results" / "match_data"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
MATCH_DATA_DIR.mkdir(parents=True, exist_ok=True)

FINAL_IMAGE_PATH = OUTPUT_IMAGE_DIR / "final_image.jpg"

# 全域常駐地圖快取 (將於 main 初始化時載入)
GLOBAL_MAP_IMG = None


# ============================================================
# 1. 地圖
# ============================================================

MAP_WIDTH_PX = 2985
MAP_HEIGHT_PX = 1730

X_GSD_M_PER_PX = 0.009296399
Y_GSD_M_PER_PX = 0.008953377

ANGLES = list(range(0, 360, 30))


# ============================================================
# 2. 航線
# ============================================================

ROUTES = {
    "1": {
        "name": "68號車格航線",
        "waypoints": [(2130, 200), (2130, 550), (60, 550)]
    },
    "2": {
        "name": "70號車格航線",
        "waypoints": [(2130, 200), (2130, 550), (1270, 550), (1270, 1350)]
    }
}


# ============================================================
# 3. 語意區域
# ============================================================

PARKING_NUMBERS = ["063", "064", "065", "066", "067", "068"]

PARKING_TO_DIGIT = {
    "063": "3",
    "064": "4",
    "065": "5",
    "066": "6",
    "067": "7",
    "068": "8"
}

DIGIT_TO_PARKING = {digit: number for number, digit in PARKING_TO_DIGIT.items()}

PARKING_MATCH_REGIONS = {
    "063": (1337, 455, 1615, 610),
    "064": (1078, 455, 1337, 610),
    "065": (809, 455, 1078, 610),
    "066": (546, 455, 809, 610),
    "067": (281, 455, 546, 610),
    "068": (9, 455, 278, 610)
}

PARKING_NUMBER_REGIONS = {
    "063": (1337, 479, 1615, 564),
    "064": (1078, 479, 1337, 564),
    "065": (809, 479, 1078, 564),
    "066": (546, 479, 809, 564),
    "067": (281, 479, 546, 564),
    "068": (9, 479, 278, 564)
}

SEMANTIC_HIGH_CONF = 0.85
SEMANTIC_MED_CONF = 0.65
MIN_SEMANTIC_INLIER_RATIO = 0.25
MIN_SEMANTIC_INLIER_RATIO_HIGH = 0.60

ENABLE_SEMANTIC_FALLBACK = True


# ============================================================
# 3b. 數字主導 / 最後一碼
# ============================================================

DIGIT_IMAGE_SIZE = 96
DIGIT_AREA_DOMINATED_RATIO = 0.25
MIN_NON_DIGIT_INLIER_RATIO = 0.35
MIN_DIGIT_DETECT_SCORE = 0.45

# 支援的旋轉角度設定 (0°, 90°, 180°, 270°)
DIGIT_DETECT_ANGLES = [0, 90, 180, 270]
UNIQUE_LAST_DIGITS = {"3", "4", "5", "7", "8"}


# ============================================================
# 4. SuperPoint / LightGlue、二階段粗篩 與 早期終止門檻
# ============================================================

ONLINE_MAX_KEYPOINTS = 512

MIN_MATCHES = 7
MIN_INLIERS = 6
MIN_INLIER_RATIO = 0.35
MIN_INLIER_RATIO_DIGIT_DOMINATED = 0.45

RANSAC_REPROJ_THRESH = 5.0

MIN_FINAL_SPREAD_X = 0.08
MIN_FINAL_SPREAD_Y = 0.08

MAX_ROUTE_DISTANCE_PX = 120.0

MIN_FILTERED_MAP_FEATURES = 30
MIN_FILTERED_MAP_FEATURES_DIGIT = 15
MAX_ONLINE_MAP_FEATURES = 300

# 二階段粗篩設定 (Two-Stage Coarse Filter)
TOP_K_COARSE_TILES = 10       # 第一階段點積粗篩僅保留分數最高的 Top-10 個 Tile 送入 LightGlue
COARSE_SIM_THRESH = 0.80    # Cosine 相似度門檻

# 早期終止設定 (Early Stopping)
EARLY_STOP_INLIERS = 12
EARLY_STOP_SCORE = 10.0


# ============================================================
# 5. 新版搜尋範圍
# ============================================================

TILE_PROGRESS_MARGIN_PX = 100.0
FEATURE_PROGRESS_MARGIN_PX = 100.0
ROUTE_FEATURE_MAX_DISTANCE_PX = 250.0


# ============================================================
# 6. Progress
# ============================================================

INITIAL_PROGRESS_BASE_PX = 50.0
INITIAL_EXTRA_PER_FAIL_FRAME_PX = 120.0
INITIAL_MAX_PROGRESS_PX = 1100.0

BASE_FORWARD_PROGRESS_PX = 160.0
EXTRA_FORWARD_PER_FAIL_FRAME_PX = 200.0
MAX_BACKWARD_PROGRESS_PX = 0.0

PROGRESS_RESET_AFTER_FAIL_FRAMES = 5

MAX_NORMAL_PROGRESS_JUMP_PX = 350.0
MAX_SEMANTIC_PROGRESS_JUMP_PX = 500.0

PENDING_CONFIRM_COUNT = 2
PENDING_PROGRESS_TOLERANCE_PX = 180.0


# ============================================================
# 7. Angle
# ============================================================

ANGLE_RESET_AFTER_FAIL_FRAMES = 5


# ============================================================
# 8. 候選歧義
# ============================================================

SAME_LOCATION_RADIUS_PX = 50.0
MIN_DISTINCT_SCORE_RATIO = 1.80


# ============================================================
# 9. Homography中心支撐檢查
# ============================================================

CENTER_SUPPORT_MARGIN_X = 0.15
CENTER_SUPPORT_MARGIN_Y = 0.15
MIN_CENTER_SUPPORT_SCORE = 0.35
USE_CONVEX_HULL_CENTER_CHECK = True


# ============================================================
# 10. 位置誤差限制
# ============================================================

ENABLE_ERROR_BOUNDARY = True
POSITION_ERROR_BOUNDARY_M = 0.5


# ============================================================
# 11. 輸出
# ============================================================

SAVE_FINAL_IMAGE = True
SAVE_MATCH_DATA = True

MATCH_CROP_MARGIN_PX = 60
MATCH_CROP_MIN_SIZE_PX = 300


# ============================================================
# 12. 全域資料
# ============================================================

_TILE_CACHE = {}
_SEMANTIC_INDEX = {}
_LAST_DIGIT_REGIONS = {}
_PREFIX_DIGIT_REGIONS = []
_CACHE_LAST_DIGIT_ONLY = False

SUCCESS_POSITIONS = []
CORRECTED_FRAME_ERRORS = []


# ============================================================
# UDP 網路封包發射工具
# ============================================================

def send_udp_telemetry(sock, ip, port, data):
    try:
        msg = json.dumps(data).encode("utf-8")
        sock.sendto(msg, (ip, port))
        return True
    except Exception as e:
        print(f"⚠️ UDP 發射傳送失敗: {e}")
        return False


# ============================================================
# 13. TinyCNN
# ============================================================

class SemanticTinyCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
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
        return self.classifier(x)


# ============================================================
# 14. TinyCNN載入
# ============================================================

def _as_int_size(value, default=DIGIT_IMAGE_SIZE):
    if value is None:
        return default
    if isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return default
        return int(value[0])
    return int(value)


def load_semantic_classifier():
    if not SEMANTIC_MODEL_PATH.exists():
        raise FileNotFoundError(f"找不到TinyCNN模型：{SEMANTIC_MODEL_PATH}")

    checkpoint = torch.load(SEMANTIC_MODEL_PATH, map_location="cpu")
    classes = list(checkpoint["classes"])
    image_size = _as_int_size(
        checkpoint.get("digit_image_size", checkpoint.get("image_size", DIGIT_IMAGE_SIZE))
    )

    model = SemanticTinyCNN(len(classes))

    try:
        model.load_state_dict(checkpoint["model_state_dict"])
    except RuntimeError as exc:
        raise RuntimeError("TinyCNN 權重與結構不一致。") from exc

    model.eval()
    dummy = torch.zeros(1, 1, image_size, image_size)

    with torch.inference_mode():
        for _ in range(3):
            model(dummy)

    print(
        f"✅ TinyCNN完成｜Classes={classes}｜Input={image_size}×{image_size}｜"
        f"type={checkpoint.get('model_type', 'unknown')}"
    )
    return model, classes, image_size


def prepare_digit_image(img, target_size=DIGIT_IMAGE_SIZE):
    if img is None or img.size == 0:
        return np.zeros((target_size, target_size), dtype=np.uint8)

    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    h, w = img.shape[:2]
    side = max(h, w)
    background = int(np.median(img))

    canvas = np.full((side, side), background, dtype=np.uint8)
    x = (side - w) // 2
    y = (side - h) // 2
    canvas[y:y + h, x:x + w] = img

    interpolation = cv2.INTER_AREA if side > target_size else cv2.INTER_CUBIC
    return cv2.resize(canvas, (target_size, target_size), interpolation=interpolation)


def classify_semantic(gray, model, classes, image_size):
    img = prepare_digit_image(gray, image_size)
    img = img.astype(np.float32) / 255.0
    img = (img - 0.5) / 0.5

    tensor = torch.from_numpy(img)[None, None]

    with torch.inference_mode():
        probs = torch.softmax(model(tensor), dim=1)[0]

    top2_prob, top2_idx = torch.topk(probs, k=min(2, len(classes)))

    index = int(top2_idx[0])
    semantic = str(classes[index])
    confidence = float(top2_prob[0])

    second = str(classes[int(top2_idx[1])]) if len(top2_idx) > 1 else "-"
    second_conf = float(top2_prob[1]) if len(top2_idx) > 1 else 0.0

    return semantic, confidence, second, second_conf


def parking_from_digit(digit):
    return DIGIT_TO_PARKING.get(str(digit), "NONE")


def get_allowed_semantics(semantic, confidence):
    if semantic in (None, "NONE") or confidence < SEMANTIC_MED_CONF:
        return None

    if semantic not in PARKING_NUMBERS:
        digit_as_parking = parking_from_digit(semantic)
        if digit_as_parking == "NONE":
            return None
        semantic = digit_as_parking

    if confidence >= SEMANTIC_HIGH_CONF:
        return {semantic}

    idx = PARKING_NUMBERS.index(semantic)
    allowed = set()

    for j in (idx - 1, idx, idx + 1):
        if 0 <= j < len(PARKING_NUMBERS):
            allowed.add(PARKING_NUMBERS[j])

    return allowed


# ============================================================
# 14b. UAV 三字偵測 + 最後一碼 (雙軸自適應與多角度檢測模組)
# ============================================================

def rotate_image(image, angle):

    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    elif angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    elif angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def find_three_digit_boxes(gray):
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    components = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if 8 <= w <= 100 and 8 <= h <= 100:
            components.append((x, y, w, h))

    if len(components) < 3:
        return None

    xs = [b[0] for b in components]
    ys = [b[1] for b in components]
    is_vertical = (max(ys) - min(ys)) > (max(xs) - min(xs))

    components.sort(key=lambda box: box[1] if is_vertical else box[0])

    best_boxes = None
    best_score = float('inf')

    for i in range(len(components) - 2):
        boxes = [components[i], components[i + 1], components[i + 2]]

        if is_vertical:
            gap1 = boxes[1][1] - (boxes[0][1] + boxes[0][3])
            gap2 = boxes[2][1] - (boxes[1][1] + boxes[1][3])
            align_err = abs(boxes[0][0] - boxes[1][0]) + abs(boxes[1][0] - boxes[2][0])
        else:
            gap1 = boxes[1][0] - (boxes[0][0] + boxes[0][2])
            gap2 = boxes[2][0] - (boxes[1][0] + boxes[1][2])
            align_err = abs(boxes[0][1] - boxes[1][1]) + abs(boxes[1][1] - boxes[2][1])

        gap_diff = abs(gap1 - gap2)
        score = gap_diff + (align_err * 0.5)

        if score < best_score:
            best_score = score
            best_boxes = boxes

    if best_boxes is None:
        return None

    return best_boxes, is_vertical


def pick_last_digit_crop(gray, boxes):
    if not boxes or len(boxes) < 3:
        return None

    x, y, w, h = boxes[2]
    
    pad = 3
    h_img, w_img = gray.shape
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w_img, x + w + pad)
    y2 = min(h_img, y + h + pad)

    return gray[y1:y2, x1:x2]


def match_parking_number(frame_gray, tiny_cnn_model):
    for angle in DIGIT_DETECT_ANGLES:
        rotated_img = rotate_image(frame_gray, angle)
        detection_res = find_three_digit_boxes(rotated_img)

        if detection_res is None:
            continue

        boxes, is_vertical = detection_res
        last_crop = pick_last_digit_crop(rotated_img, boxes)

        if last_crop is not None and last_crop.size > 0:
            crop_resized = cv2.resize(last_crop, (28, 28))
            tensor_in = torch.tensor(crop_resized, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

            with torch.no_grad():
                output = tiny_cnn_model(tensor_in)
                pred_label = torch.argmax(output, dim=1).item()

            return {
                "detected_angle": angle,
                "is_vertical": is_vertical,
                "boxes": boxes,
                "predicted_last_digit": pred_label,
                "last_digit_crop": last_crop
            }

    return None


def transform_box_to_original_ortho(box, angle, img_w, img_h):
    x, y, bw, bh = box[:4]
    if angle == 0:
        ox, oy, ow, oh = x, y, bw, bh
    elif angle == 90:
        ox = y
        oy = img_h - (x + bw)
        ow = bh
        oh = bw
    elif angle == 180:
        ox = img_w - (x + bw)
        oy = img_h - (y + bh)
        ow = bw
        oh = bh
    elif angle == 270:
        ox = img_w - (y + bh)
        oy = x
        ow = bh
        oh = bw
    else:
        ox, oy, ow, oh = x, y, bw, bh

    ox = int(np.clip(ox, 0, img_w - 1))
    oy = int(np.clip(oy, 0, img_h - 1))
    ow = max(1, min(int(ow), img_w - ox))
    oh = max(1, min(int(oh), img_h - oy))
    return (ox, oy, ow, oh, int(box[4]) if len(box) > 4 else 0)


def crop_box_with_margin(gray, box, margin_ratio=0.28):
    x, y, w, h = box[:4]
    margin = int(round(max(w, h) * margin_ratio))
    x1 = max(0, int(x - margin))
    y1 = max(0, int(y - margin))
    x2 = min(gray.shape[1], int(x + w + margin))
    y2 = min(gray.shape[0], int(y + h + margin))
    return gray[y1:y2, x1:x2].copy()


def digit_boxes_area_ratio(boxes, img_w, img_h):
    if not boxes:
        return 0.0
    xs1 = min(b[0] for b in boxes)
    ys1 = min(b[1] for b in boxes)
    xs2 = max(b[0] + b[2] for b in boxes)
    ys2 = max(b[1] + b[3] for b in boxes)
    area = max(0, xs2 - xs1) * max(0, ys2 - ys1)
    return float(area) / max(float(img_w * img_h), 1.0)


def pick_last_digit_index(preds):
    n = len(preds)
    if n == 0:
        return None
    if n == 1:
        return 0

    end_indices = [0, n - 1]

    unique = []
    for i in end_indices:
        if preds[i]["digit"] in UNIQUE_LAST_DIGITS and preds[i]["conf"] >= SEMANTIC_MED_CONF:
            unique.append(i)

    if len(unique) == 1:
        return unique[0]

    if len(unique) >= 2:
        unique.sort(key=lambda i: preds[i]["conf"], reverse=True)
        return unique[0]

    six_ends = [i for i in end_indices if preds[i]["digit"] == "6"]
    if six_ends:
        six_ends.sort(key=lambda i: preds[i]["conf"], reverse=True)
        return six_ends[0]

    return n - 1


def detect_uav_digits(gray):
    img_h, img_w = gray.shape[:2]
    best = None

    for angle in DIGIT_DETECT_ANGLES:
        rot = rotate_image(gray, angle)
        detection_res = find_three_digit_boxes(rot)

        if detection_res is None:
            continue

        boxes, is_vertical = detection_res
        orig_boxes = [transform_box_to_original_ortho(b, angle, img_w, img_h) for b in boxes]
        ratio = digit_boxes_area_ratio(orig_boxes, img_w, img_h)

        best = {
            "score": 1.0,
            "angle": int(angle),
            "rot_boxes": boxes,
            "boxes_orig": orig_boxes,
            "is_vertical": is_vertical,
            "rot": rot,
            "digit_area_ratio": ratio,
            "digit_dominated": ratio >= DIGIT_AREA_DOMINATED_RATIO
        }
        break

    return best


def classify_uav_last_digit(gray, digit_info, model, classes, image_size):
    empty = ("NONE", 0.0, "-", 0.0, None)

    if digit_info is None:
        return empty

    rot = digit_info["rot"]
    rot_boxes = digit_info["rot_boxes"]

    last_crop = pick_last_digit_crop(rot, rot_boxes)

    preds = []
    for box in rot_boxes:
        crop = crop_box_with_margin(rot, box, 0.30)
        digit, conf, second, second_conf = classify_semantic(crop, model, classes, image_size)
        preds.append({
            "digit": digit,
            "conf": conf,
            "second": second,
            "second_conf": second_conf
        })

    last_index = 2 if len(preds) >= 3 else pick_last_digit_index(preds)
    if last_index is None:
        return empty

    digit_info["last_index"] = last_index
    digit_info["preds"] = preds
    digit_info["last_digit_crop"] = last_crop

    chosen = preds[last_index]
    parking = parking_from_digit(chosen["digit"])
    second_parking = parking_from_digit(chosen["second"])

    return parking, chosen["conf"], second_parking, chosen["second_conf"], digit_info


def mask_non_discriminative_digits(gray, digit_info):
    if digit_info is None or not digit_info.get("boxes_orig"):
        return gray

    last_index = digit_info.get("last_index")
    if last_index is None:
        return gray

    masked = gray.copy()
    bg = int(np.median(gray))

    for i, box in enumerate(digit_info["boxes_orig"]):
        if i == last_index:
            continue
        x, y, w, h = box[:4]
        pad = int(round(0.10 * max(w, h)))
        x1 = max(0, int(x) - pad)
        y1 = max(0, int(y) - pad)
        x2 = min(gray.shape[1], int(x + w) + pad)
        y2 = min(gray.shape[0], int(y + h) + pad)
        masked[y1:y2, x1:x2] = bg

    return masked


# ============================================================
# 15. Tile Cache (已優化：預先轉換為 PyTorch Tensor)
# ============================================================

def fallback_last_digit_regions():
    regions = {}
    for number, (x1, y1, x2, y2) in PARKING_NUMBER_REGIONS.items():
        width = x2 - x1
        regions[number] = (int(x1 + 0.60 * width), y1, x2, y2)
    return regions


def fallback_prefix_digit_regions():
    regions = []
    for number, (x1, y1, x2, y2) in PARKING_NUMBER_REGIONS.items():
        width = x2 - x1
        regions.append((x1, y1, int(x1 + 0.60 * width), y2))
    return regions


def load_digit_geometry_from_npz(data):
    last_digit_regions = {}
    files = set(data.files)

    for number in PARKING_NUMBERS:
        key = f"last_digit_region_{number}"
        if key in files:
            box = data[key].astype(np.int32).tolist()
            last_digit_regions[number] = tuple(int(v) for v in box[:4])

    prefix = []
    if "prefix_digit_regions" in files:
        arr = data["prefix_digit_regions"]
        if arr is not None and len(arr) > 0:
            prefix = [tuple(int(v) for v in row[:4]) for row in arr.astype(np.int32)]

    last_digit_only = False
    if "last_digit_only_semantic" in files:
        last_digit_only = bool(int(np.array(data["last_digit_only_semantic"]).reshape(-1)[0]))

    return last_digit_regions, prefix, last_digit_only


def load_tile_cache():
    global _TILE_CACHE, _SEMANTIC_INDEX
    global _LAST_DIGIT_REGIONS, _PREFIX_DIGIT_REGIONS, _CACHE_LAST_DIGIT_ONLY

    if not CACHE_PATH.exists():
        raise FileNotFoundError(f"找不到Tile Cache：{CACHE_PATH}")

    print("⚡ 載入Tile Cache...")

    data = np.load(CACHE_PATH, allow_pickle=True)

    _TILE_CACHE = {}
    _SEMANTIC_INDEX = {number: {} for number in PARKING_NUMBERS}

    last_digit_regions, prefix, last_digit_only = load_digit_geometry_from_npz(data)

    if not last_digit_regions and SEMANTIC_META_PATH.exists():
        meta = np.load(SEMANTIC_META_PATH, allow_pickle=True)
        last_digit_regions, prefix, last_digit_only = load_digit_geometry_from_npz(meta)

    if not last_digit_regions:
        last_digit_regions = fallback_last_digit_regions()
        prefix = fallback_prefix_digit_regions()
        print("⚠️ Cache 沒有最後一碼框，改用車格號碼右40%。")

    _LAST_DIGIT_REGIONS = last_digit_regions
    _PREFIX_DIGIT_REGIONS = prefix
    _CACHE_LAST_DIGIT_ONLY = last_digit_only

    for angle in ANGLES:
        ids_key = f"angle_{angle:03d}_tile_ids"

        if ids_key not in data.files:
            continue

        _TILE_CACHE[angle] = {}

        for raw_id in data[ids_key]:
            tile_id = int(raw_id)
            prefix_key = f"a{angle:03d}_t{tile_id:02d}"

            required = [
                f"{prefix_key}_kpts_local",
                f"{prefix_key}_kpts_0deg",
                f"{prefix_key}_descs",
                f"{prefix_key}_size"
            ]

            if not all(key in data.files for key in required):
                continue

            raw_size = data[f"{prefix_key}_size"].astype(np.float32)

            tile = {
                "kpts_local": torch.from_numpy(data[f"{prefix_key}_kpts_local"].astype(np.float32)).to(DEVICE),
                "kpts_0deg": data[f"{prefix_key}_kpts_0deg"].astype(np.float32),
                "descs": torch.from_numpy(data[f"{prefix_key}_descs"].astype(np.float32)).to(DEVICE),
                "size": torch.tensor([[float(raw_size[0]), float(raw_size[1])]], dtype=torch.float32, device=DEVICE)
            }

            scores_key = f"{prefix_key}_scores"
            if scores_key in data.files:
                tile["scores"] = data[scores_key].astype(np.float32)
            else:
                tile["scores"] = np.ones(len(tile["kpts_0deg"]), dtype=np.float32)

            prefix_mask_key = f"{prefix_key}_prefix_digit_mask"
            if prefix_mask_key in data.files:
                tile["prefix_digit_mask"] = data[prefix_mask_key].astype(bool)
            else:
                tile["prefix_digit_mask"] = None

            for route_id in ["1", "2"]:
                for suffix in ["progress", "distance", "range"]:
                    key = f"{prefix_key}_route{route_id}_{suffix}"
                    if key in data.files:
                        tile[f"route{route_id}_{suffix}"] = data[key].astype(np.float32)

            _TILE_CACHE[angle][tile_id] = tile

    for number in PARKING_NUMBERS:
        for angle in ANGLES:
            key = f"semantic_{number}_a{angle:03d}"
            _SEMANTIC_INDEX[number][angle] = data[key].astype(int).tolist() if key in data.files else []

    total_tiles = sum(len(v) for v in _TILE_CACHE.values())
    print(f"✅ Tile Cache載入完成｜總Tile={total_tiles}")


# ============================================================
# 16. 航線
# ============================================================

def select_user_route():
    print("\n[1] 68號車格航線")
    print("[2] 70號車格航線")

    while True:
        choice = input("👉 請輸入1或2：").strip()

        if choice in ROUTES:
            route = ROUTES[choice]
            print(f"✅ 已載入：{route['name']}")
            return choice, route["waypoints"], route["name"]

        print("⚠️ 請輸入1或2")


def get_route_total_length(waypoints):
    return sum(
        math.hypot(
            waypoints[i + 1][0] - waypoints[i][0],
            waypoints[i + 1][1] - waypoints[i][1]
        )
        for i in range(len(waypoints) - 1)
    )


def project_point_to_route(point, waypoints):
    x, y = point

    best_dist = float("inf")
    best_progress = 0.0
    best_projection = waypoints[0]
    cumulative = 0.0

    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]

        dx, dy = bx - ax, by - ay
        length = math.hypot(dx, dy)

        if length < 1e-8:
            continue

        t = ((x - ax) * dx + (y - ay) * dy) / (length * length)
        t = float(np.clip(t, 0.0, 1.0))

        qx, qy = ax + t * dx, ay + t * dy
        distance = math.hypot(x - qx, y - qy)
        progress = cumulative + t * length

        if distance < best_dist:
            best_dist = distance
            best_progress = progress
            best_projection = (qx, qy)

        cumulative += length

    return {
        "distance": float(best_dist),
        "progress": float(best_progress),
        "projection": best_projection
    }


# ============================================================
# 17. Progress
# ============================================================

def get_progress_window(last_progress, fail_frames, total_length):
    if last_progress is None:
        upper = min(
            INITIAL_PROGRESS_BASE_PX + fail_frames * INITIAL_EXTRA_PER_FAIL_FRAME_PX,
            INITIAL_MAX_PROGRESS_PX,
            total_length
        )
        return 0.0, upper, "起點初始化"

    if fail_frames >= PROGRESS_RESET_AFTER_FAIL_FRAMES:
        return 0.0, total_length, "Progress已解除"

    lower = max(0.0, last_progress - MAX_BACKWARD_PROGRESS_PX)

    upper = min(
        total_length,
        last_progress + BASE_FORWARD_PROGRESS_PX + fail_frames * EXTRA_FORWARD_PER_FAIL_FRAME_PX
    )

    state = "正常追蹤" if fail_frames == 0 else "失敗後擴張"

    return lower, upper, state


# ============================================================
# 18. Tile篩選
# ============================================================

def tile_progress_overlaps(tile, route_id, p_min, p_max):
    key = f"route{route_id}_range"

    if key not in tile:
        return True

    tile_min, tile_max = map(float, tile[key])

    if tile_min < 0 or tile_max < 0:
        return False

    return not (
        tile_max < p_min - TILE_PROGRESS_MARGIN_PX or
        tile_min > p_max + TILE_PROGRESS_MARGIN_PX
    )


def map_point_in_prefix_digit(x, y):
    for x1, y1, x2, y2 in _PREFIX_DIGIT_REGIONS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def get_filtered_tile_features(tile, route_id, p_min, p_max, drop_prefix_digits=False):
    progress_key = f"route{route_id}_progress"
    distance_key = f"route{route_id}_distance"

    if progress_key not in tile:
        return None

    progress = tile[progress_key]
    distance = tile.get(distance_key)

    mask = (
        (progress >= p_min - FEATURE_PROGRESS_MARGIN_PX) &
        (progress <= p_max + FEATURE_PROGRESS_MARGIN_PX)
    )

    if distance is not None:
        mask &= distance <= ROUTE_FEATURE_MAX_DISTANCE_PX

    if drop_prefix_digits:
        cached = tile.get("prefix_digit_mask")
        if cached is not None and len(cached) == len(mask):
            mask &= ~cached.astype(bool)
        elif _PREFIX_DIGIT_REGIONS:
            kpts = tile["kpts_0deg"]
            prefix = np.zeros(len(kpts), dtype=bool)
            for x1, y1, x2, y2 in _PREFIX_DIGIT_REGIONS:
                prefix |= (
                    (kpts[:, 0] >= x1) & (kpts[:, 0] <= x2) &
                    (kpts[:, 1] >= y1) & (kpts[:, 1] <= y2)
                )
            mask &= ~prefix

    ids = np.flatnonzero(mask)
    min_feat = MIN_FILTERED_MAP_FEATURES_DIGIT if drop_prefix_digits else MIN_FILTERED_MAP_FEATURES

    if len(ids) < min_feat:
        return None

    if len(ids) > MAX_ONLINE_MAP_FEATURES:
        order = np.argsort(tile["scores"][ids])[::-1]
        ids = ids[order[:MAX_ONLINE_MAP_FEATURES]]

    return {
        "kpts_local": tile["kpts_local"][ids],
        "kpts_0deg": tile["kpts_0deg"][ids],
        "descs": tile["descs"][ids]
    }


# ============================================================
# 19. Homography工具
# ============================================================

def project_center(H, img_w, img_h):
    point = H @ np.array([img_w / 2.0, img_h / 2.0, 1.0], dtype=np.float64)

    if abs(point[2]) < 1e-8:
        return None

    return float(point[0] / point[2]), float(point[1] / point[2])


def compute_inlier_spread(kp_uav, inlier_mask, img_w, img_h):
    pts = kp_uav[inlier_mask]

    if len(pts) < 4:
        return 0.0, 0.0

    spread_x = np.ptp(pts[:, 0]) / max(float(img_w), 1.0)
    spread_y = np.ptp(pts[:, 1]) / max(float(img_h), 1.0)

    return float(spread_x), float(spread_y)


def compute_hull_ratio(kp_uav, inlier_mask, img_w, img_h):
    pts = kp_uav[inlier_mask].astype(np.float32)

    if len(pts) < 4:
        return 0.0

    hull = cv2.convexHull(pts)
    area = cv2.contourArea(hull)

    return float(area / max(float(img_w * img_h), 1.0))


def compute_reprojection_error(H, kp_uav, kp_map, inlier_mask):
    src = kp_uav[inlier_mask]
    dst = kp_map[inlier_mask]

    if len(src) == 0:
        return float("inf")

    homo = np.hstack([src, np.ones((len(src), 1), dtype=np.float32)])
    projected = (H @ homo.T).T

    valid = np.abs(projected[:, 2]) > 1e-8

    if not np.any(valid):
        return float("inf")

    projected = projected[valid, :2] / projected[valid, 2:3]
    errors = np.linalg.norm(projected - dst[valid], axis=1)

    return float(np.median(errors))


def compute_visual_score(inliers, ratio, median_error, spread_x, spread_y, hull_ratio, center_support):
    spread_area = max(spread_x * spread_y, 1e-4)
    hull_factor = min(hull_ratio / 0.10, 1.0)

    return float(
        inliers * ratio * spread_area *
        (0.5 + 0.5 * hull_factor) *
        (0.5 + 0.5 * center_support) /
        (1.0 + median_error)
    )


# ============================================================
# 20. Center Support
# ============================================================

def check_center_support(kp_uav, inlier_mask, img_w, img_h):
    pts = kp_uav[inlier_mask].astype(np.float32)

    if len(pts) < 4:
        return False, 0.0, "內點不足"

    cx, cy = img_w / 2.0, img_h / 2.0

    min_x, max_x = float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))
    min_y, max_y = float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))

    margin_x = img_w * CENTER_SUPPORT_MARGIN_X
    margin_y = img_h * CENTER_SUPPORT_MARGIN_Y

    center_in_bbox = (
        min_x - margin_x <= cx <= max_x + margin_x and
        min_y - margin_y <= cy <= max_y + margin_y
    )

    mean_x, mean_y = np.mean(pts, axis=0)

    normalized_dist = math.hypot(
        (float(mean_x) - cx) / max(float(img_w), 1.0),
        (float(mean_y) - cy) / max(float(img_h), 1.0)
    )

    center_score = max(0.0, 1.0 - normalized_dist / 0.707)

    center_in_hull = False

    if USE_CONVEX_HULL_CENTER_CHECK and len(pts) >= 4:
        hull = cv2.convexHull(pts)
        center_in_hull = cv2.pointPolygonTest(hull, (float(cx), float(cy)), False) >= 0

    if center_in_hull:
        return True, 1.0, "中心位於Inlier凸包內"

    if not center_in_bbox:
        return False, center_score, "中心超出Inlier支撐範圍"

    if center_score < MIN_CENTER_SUPPORT_SCORE:
        return False, center_score, "Inlier整體距離中心太遠"

    return True, center_score, "中心支撐通過"


# ============================================================
# 21. Homography四角檢查
# ============================================================

def validate_projected_quad(H, img_w, img_h):
    corners = np.float32([
        [0, 0],
        [img_w - 1, 0],
        [img_w - 1, img_h - 1],
        [0, img_h - 1]
    ]).reshape(-1, 1, 2)

    try:
        projected = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    except cv2.error:
        return False

    if not np.all(np.isfinite(projected)):
        return False

    contour = projected.astype(np.float32).reshape(-1, 1, 2)

    if not cv2.isContourConvex(contour):
        return False

    area = abs(cv2.contourArea(contour))

    if area < 100.0:
        return False

    return True


# ============================================================
# 21b. 數字主導幾何檢查
# ============================================================

def point_in_painted_number(x, y):
    for x1, y1, x2, y2 in PARKING_NUMBER_REGIONS.values():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    for x1, y1, x2, y2 in _PREFIX_DIGIT_REGIONS:
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    for x1, y1, x2, y2 in _LAST_DIGIT_REGIONS.values():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True
    return False


def non_digit_inlier_ratio(kp_map, inlier_mask):
    pts = kp_map[inlier_mask]
    if len(pts) == 0:
        return 0.0
    count = sum(1 for x, y in pts if not point_in_painted_number(x, y))
    return count / len(pts)


def last_digit_inlier_ratio(kp_map, inlier_mask, allowed):
    if not allowed:
        return 1.0

    pts = kp_map[inlier_mask]
    if len(pts) == 0:
        return 0.0

    count = 0
    for x, y in pts:
        for number in allowed:
            box = _LAST_DIGIT_REGIONS.get(number)
            if box is None:
                continue
            x1, y1, x2, y2 = box
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1
                break

    return count / len(pts)


# ============================================================
# 22. LightGlue單Tile (已優化：消除 torch.from_numpy 轉型)
# ============================================================

def estimate_pose(kp_uav, kp_map, digit_dominated):
    if digit_dominated:
        M, mask = cv2.estimateAffinePartial2D(
            kp_uav.astype(np.float32),
            kp_map.astype(np.float32),
            method=cv2.RANSAC,
            ransacReprojThreshold=RANSAC_REPROJ_THRESH,
            maxIters=5000,
            confidence=0.999
        )
        if M is None or mask is None:
            return None, None
        H = np.eye(3, dtype=np.float64)
        H[:2, :] = M
        return H, mask

    H, mask = cv2.findHomography(kp_uav, kp_map, cv2.USAC_MAGSAC, RANSAC_REPROJ_THRESH)
    return H, mask


def match_one_tile(gray_shape, tile, filtered, feats0, matcher, digit_dominated=False):
    min_map = MIN_FILTERED_MAP_FEATURES_DIGIT if digit_dominated else MIN_FILTERED_MAP_FEATURES

    if filtered is None or len(filtered["kpts_local"]) < min_map:
        return None

    num_kpts0 = feats0["keypoints"].shape[1]
    num_kpts1 = len(filtered["kpts_local"])
    
    if num_kpts0 < MIN_MATCHES or num_kpts1 < min_map:
        return None
    if min(num_kpts0, num_kpts1) / max(num_kpts0, num_kpts1) < 0.02:
        return None

    img_h, img_w = gray_shape[:2]

    map_local = filtered["kpts_local"]
    map_global = filtered["kpts_0deg"]
    descriptors = filtered["descs"]

    feats1 = {
        "keypoints": map_local[None],
        "descriptors": descriptors[None],
        "image_size": tile["size"]
    }

    with torch.inference_mode():
        output = matcher({"image0": feats0, "image1": feats1})

    matches = output["matches"][0]

    if len(matches) < MIN_MATCHES:
        return None

    matches_np = matches.cpu().numpy()

    kp_uav = feats0["keypoints"][0][matches[:, 0]].cpu().numpy()
    kp_map = map_global[matches_np[:, 1]]

    H, mask = estimate_pose(kp_uav, kp_map, digit_dominated)

    if H is None or mask is None:
        return None

    inlier_mask = mask.ravel().astype(bool)
    inliers = int(np.sum(inlier_mask))
    ratio = inliers / max(len(matches), 1)
    min_ratio = MIN_INLIER_RATIO_DIGIT_DOMINATED if digit_dominated else MIN_INLIER_RATIO

    if inliers < MIN_INLIERS or ratio < min_ratio:
        return None

    if not validate_projected_quad(H, img_w, img_h):
        return None

    spread_x, spread_y = compute_inlier_spread(kp_uav, inlier_mask, img_w, img_h)

    if spread_x < MIN_FINAL_SPREAD_X or spread_y < MIN_FINAL_SPREAD_Y:
        return None

    center_ok, center_support, center_reason = check_center_support(
        kp_uav, inlier_mask, img_w, img_h
    )

    if not center_ok:
        return None

    if digit_dominated:
        nd_ratio = non_digit_inlier_ratio(kp_map, inlier_mask)
        if nd_ratio < MIN_NON_DIGIT_INLIER_RATIO:
            return None
    else:
        nd_ratio = non_digit_inlier_ratio(kp_map, inlier_mask)

    position = project_center(H, img_w, img_h)

    if position is None:
        return None

    if not (0 <= position[0] <= MAP_WIDTH_PX and 0 <= position[1] <= MAP_HEIGHT_PX):
        return None

    hull_ratio = compute_hull_ratio(kp_uav, inlier_mask, img_w, img_h)
    median_error = compute_reprojection_error(H, kp_uav, kp_map, inlier_mask)

    score = compute_visual_score(
        inliers,
        ratio,
        median_error,
        spread_x,
        spread_y,
        hull_ratio,
        center_support
    )

    return {
        "position": position,
        "H": H,
        "pose_model": "affine" if digit_dominated else "homography",
        "matches": int(len(matches)),
        "inliers": inliers,
        "inlier_ratio": ratio,
        "non_digit_ratio": nd_ratio,
        "median_error": median_error,
        "spread_x": spread_x,
        "spread_y": spread_y,
        "hull_ratio": hull_ratio,
        "center_support": center_support,
        "center_reason": center_reason,
        "score": score,
        "kp_uav": kp_uav,
        "kp_map": kp_map,
        "inlier_mask": inlier_mask
    }


# ============================================================
# 23. Semantic一致性
# ============================================================

def point_in_semantic_region(point, allowed):
    if not allowed:
        return True

    x, y = point

    for number in allowed:
        x1, y1, x2, y2 = PARKING_MATCH_REGIONS[number]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return True

    return False


def semantic_inlier_ratio(result, allowed):
    if not allowed:
        return 1.0

    pts = result["kp_map"][result["inlier_mask"]]

    if len(pts) == 0:
        return 0.0

    count = 0

    for x, y in pts:
        for number in allowed:
            x1, y1, x2, y2 = PARKING_MATCH_REGIONS[number]
            if x1 <= x <= x2 and y1 <= y <= y2:
                count += 1
                break

    return count / len(pts)


def semantic_result_is_valid(result, allowed, semantic_conf):
    if not allowed:
        return True

    position_ok = point_in_semantic_region(result["position"], allowed)
    region_ratio = semantic_inlier_ratio(result, allowed)
    last_ratio = last_digit_inlier_ratio(result["kp_map"], result["inlier_mask"], allowed)

    if semantic_conf >= SEMANTIC_HIGH_CONF:
        inlier_ok = region_ratio >= MIN_SEMANTIC_INLIER_RATIO_HIGH or last_ratio > 0.0
        return position_ok or inlier_ok

    inlier_ok = region_ratio >= MIN_SEMANTIC_INLIER_RATIO
    return position_ok or inlier_ok


# ============================================================
# 24. L2 / L3
# ============================================================

def build_angle_levels(last_angle):
    if last_angle is None:
        return [("L3全角度", ANGLES.copy())]

    l2 = [
        (last_angle - 30) % 360,
        last_angle,
        (last_angle + 30) % 360
    ]

    used = set(l2)
    l3 = [angle for angle in ANGLES if angle not in used]

    return [
        ("L2鄰近角度", l2),
        ("L3全角度", l3)
    ]


# ============================================================
# 25. 單Level搜尋 (已優化：使用 PyTorch Tensor 矩陣點積)
# ============================================================

def search_level(
    level_angles,
    semantic,
    semantic_conf,
    route_id,
    p_min,
    p_max,
    feats0,
    gray_shape,
    matcher,
    digit_dominated=False
):
    allowed = get_allowed_semantics(semantic, semantic_conf)
    candidates = []

    stats = {
        "tiles_before_progress": 0,
        "tiles_after_progress": 0,
        "lightglue_calls": 0
    }

    query_descs = feats0["descriptors"][0]
    coarse_candidates = []

    for angle in level_angles:
        if allowed:
            tile_ids = set()
            for number in allowed:
                tile_ids.update(_SEMANTIC_INDEX.get(number, {}).get(angle, []))
            tile_ids = list(tile_ids)
        else:
            tile_ids = list(_TILE_CACHE.get(angle, {}).keys())

        stats["tiles_before_progress"] += len(tile_ids)

        for tile_id in tile_ids:
            tile = _TILE_CACHE.get(angle, {}).get(tile_id)

            if tile is None:
                continue

            if not tile_progress_overlaps(tile, route_id, p_min, p_max):
                continue

            filtered = get_filtered_tile_features(
                tile, route_id, p_min, p_max,
                drop_prefix_digits=digit_dominated
            )

            if filtered is None:
                continue

            stats["tiles_after_progress"] += 1

            tile_descs = filtered["descs"]
            sim_matrix = torch.mm(query_descs, tile_descs.T)

            max_sim_per_point = torch.max(sim_matrix, dim=1).values
            coarse_score = int((max_sim_per_point > COARSE_SIM_THRESH).sum().item())

            coarse_candidates.append({
                "coarse_score": coarse_score,
                "tile": tile,
                "filtered": filtered,
                "angle": angle,
                "tile_id": tile_id
            })

    if not coarse_candidates:
        return candidates, stats

    coarse_candidates.sort(key=lambda x: x["coarse_score"], reverse=True)
    top_k_candidates = coarse_candidates[:TOP_K_COARSE_TILES]

    for item in top_k_candidates:
        stats["lightglue_calls"] += 1

        result = match_one_tile(
            gray_shape, item["tile"], item["filtered"], feats0, matcher,
            digit_dominated=digit_dominated
        )

        if result is None:
            continue

        result["angle"] = item["angle"]
        result["tile_id"] = item["tile_id"]
        result["semantic_ratio"] = semantic_inlier_ratio(result, allowed)
        result["last_digit_ratio"] = last_digit_inlier_ratio(
            result["kp_map"], result["inlier_mask"], allowed
        )

        if not semantic_result_is_valid(result, allowed, semantic_conf):
            continue

        candidates.append(result)

        if result["inliers"] >= EARLY_STOP_INLIERS and result["score"] >= EARLY_STOP_SCORE:
            return candidates, stats

    return candidates, stats


# ============================================================
# 26. 最終候選
# ============================================================

def find_second_distinct_candidate(candidates, best):
    for candidate in candidates[1:]:
        distance = math.hypot(
            candidate["position"][0] - best["position"][0],
            candidate["position"][1] - best["position"][1]
        )
        if distance > SAME_LOCATION_RADIUS_PX:
            return candidate
    return None


def choose_valid_candidate(candidates, waypoints, p_min, p_max):
    valid = []

    for result in candidates:
        route = project_point_to_route(result["position"], waypoints)

        if route["distance"] > MAX_ROUTE_DISTANCE_PX:
            continue

        if not (
            p_min <=
            route["progress"] <=
            p_max
        ):
            continue

        result["route_distance"] = route["distance"]
        result["route_progress"] = route["progress"]
        result["route_projection"] = route["projection"]
        valid.append(result)

    if not valid:
        return None, "無Route/Progress可靠候選"

    valid.sort(key=lambda r: r["score"], reverse=True)

    best = valid[0]
    second = find_second_distinct_candidate(valid, best)

    if second is not None:
        score_ratio = best["score"] / max(second["score"], 1e-8)
        if score_ratio < MIN_DISTINCT_SCORE_RATIO:
            return None, f"候選歧義 Ratio={score_ratio:.2f}"

    return best, "可靠"


# ============================================================
# 27. 分層搜尋
# ============================================================

def hierarchical_search(
    last_angle,
    semantic,
    semantic_conf,
    route_id,
    p_min,
    p_max,
    feats0,
    gray_shape,
    matcher,
    waypoints,
    digit_dominated=False
):
    levels = build_angle_levels(last_angle)

    total_stats = {
        "tiles_before_progress": 0,
        "tiles_after_progress": 0,
        "lightglue_calls": 0
    }

    for level_name, level_angles in levels:
        candidates, stats = search_level(
            level_angles,
            semantic,
            semantic_conf,
            route_id,
            p_min,
            p_max,
            feats0,
            gray_shape,
            matcher,
            digit_dominated=digit_dominated
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        best, reason = choose_valid_candidate(candidates, waypoints, p_min, p_max)

        if best is not None:
            return best, level_name, "語意搜尋", total_stats

        print(f"   ↳ {level_name}失敗｜{reason}")

    allowed = get_allowed_semantics(semantic, semantic_conf)

    allow_fallback = (
        ENABLE_SEMANTIC_FALLBACK
        and allowed
        and semantic_conf < SEMANTIC_HIGH_CONF
    )

    if allow_fallback:
        print("⚠️ 中信心語意搜尋失敗，解除TinyCNN限制重新搜尋")

        for level_name, level_angles in levels:
            candidates, stats = search_level(
                level_angles,
                "NONE",
                0.0,
                route_id,
                p_min,
                p_max,
                feats0,
                gray_shape,
                matcher,
                digit_dominated=digit_dominated
            )

            for key in total_stats:
                total_stats[key] += stats[key]

            best, reason = choose_valid_candidate(candidates, waypoints, p_min, p_max)

            if best is not None:
                return best, level_name, "解除語意Fallback", total_stats

            print(f"   ↳ Fallback {level_name}失敗｜{reason}")
    elif allowed and semantic_conf >= SEMANTIC_HIGH_CONF:
        print("🔒 高信心最後一碼，不解除語意限制")

    return None, None, None, total_stats


# ============================================================
# 28. Progress確認
# ============================================================

def check_progress_confirmation(
    new_progress,
    last_progress,
    pending_progress,
    pending_count,
    semantic_conf
):
    if last_progress is None:
        return True, None, 0, "第一次定位"

    jump = new_progress - last_progress

    max_forward = (
        MAX_SEMANTIC_PROGRESS_JUMP_PX
        if semantic_conf >= SEMANTIC_HIGH_CONF
        else MAX_NORMAL_PROGRESS_JUMP_PX
    )

    if -MAX_BACKWARD_PROGRESS_PX <= jump <= max_forward:
        return True, None, 0, "正常Progress"

    if pending_progress is None:
        return False, new_progress, 1, "Progress跳躍，等待確認"

    if abs(new_progress - pending_progress) <= PENDING_PROGRESS_TOLERANCE_PX:
        pending_count += 1

        if pending_count >= PENDING_CONFIRM_COUNT:
            return True, None, 0, "跨幀確認成功"

        return False, new_progress, pending_count, "持續確認"

    return False, new_progress, 1, "暫定位置改變"


# ============================================================
# 29. 0.5m邊界
# ============================================================

def apply_error_boundary(position, waypoints):
    route = project_point_to_route(position, waypoints)

    x, y = position
    qx, qy = route["projection"]

    dx_m = (x - qx) * X_GSD_M_PER_PX
    dy_m = (y - qy) * Y_GSD_M_PER_PX

    error_m = math.hypot(dx_m, dy_m)

    if not ENABLE_ERROR_BOUNDARY or error_m <= POSITION_ERROR_BOUNDARY_M:
        return (float(x), float(y)), error_m, False

    scale = POSITION_ERROR_BOUNDARY_M / error_m

    corrected = (
        float(qx + (x - qx) * scale),
        float(qy + (y - qy) * scale)
    )

    return corrected, error_m, True


# ============================================================
# 30. CTE
# ============================================================

def compute_cte_m(point, waypoints):
    px, py = point
    px_m = px * X_GSD_M_PER_PX
    py_m = py * Y_GSD_M_PER_PX

    best = float("inf")

    for i in range(len(waypoints) - 1):
        ax, ay = waypoints[i]
        bx, by = waypoints[i + 1]

        ax_m = ax * X_GSD_M_PER_PX
        ay_m = ay * Y_GSD_M_PER_PX
        bx_m = bx * X_GSD_M_PER_PX
        by_m = by * Y_GSD_M_PER_PX

        dx, dy = bx_m - ax_m, by_m - ay_m
        length_sq = dx * dx + dy * dy

        if length_sq < 1e-12:
            distance = math.hypot(px_m - ax_m, py_m - ay_m)
        else:
            t = ((px_m - ax_m) * dx + (py_m - ay_m) * dy) / length_sq
            t = float(np.clip(t, 0.0, 1.0))
            qx = ax_m + t * dx
            qy = ay_m + t * dy
            distance = math.hypot(px_m - qx, py_m - qy)

        best = min(best, distance)

    return float(best)


# ============================================================
# 31. final_image (直接使用記憶體中的地圖)
# ============================================================

def save_final_position_image(result, waypoints, map_base):
    SUCCESS_POSITIONS.append(result["position"])

    # 直接 copy 記憶體中的地圖，不用再讀檔
    img = map_base.copy()

    route = np.array(waypoints, dtype=np.int32).reshape(-1, 1, 2)

    cv2.polylines(img, [route], False, (0, 165, 255), 5, cv2.LINE_AA)

    if len(SUCCESS_POSITIONS) >= 2:
        trajectory = np.array(
            [(int(round(x)), int(round(y))) for x, y in SUCCESS_POSITIONS],
            dtype=np.int32
        ).reshape(-1, 1, 2)
        cv2.polylines(img, [trajectory], False, (0, 255, 0), 4, cv2.LINE_AA)

    for index, (x, y) in enumerate(SUCCESS_POSITIONS, 1):
        p = int(round(x)), int(round(y))
        cv2.circle(img, p, 6, (0, 255, 0), -1)
        cv2.putText(img, str(index), (p[0] + 7, p[1] - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 2)

    latest = SUCCESS_POSITIONS[-1]
    cv2.circle(img, (int(round(latest[0])), int(round(latest[1]))), 11, (0, 0, 255), -1)
    cv2.imwrite(str(FINAL_IMAGE_PATH), img)


# ============================================================
# 32. match_data (直接使用記憶體中的地圖)
# ============================================================

def save_match_diagnostic(
    frame_path,
    result,
    waypoints,
    frame_index,
    semantic,
    semantic_conf,
    semantic_second,
    semantic_second_conf,
    level_name,
    search_mode,
    map_base,
    digit_info=None
):
    uav = cv2.imread(str(frame_path))
    # 使用傳入的記憶體地圖 copy，省去 2985x1730 圖像解碼時間
    map_img = map_base.copy()

    if uav is None:
        return

    inlier_mask = result["inlier_mask"]
    uav_pts = result["kp_uav"][inlier_mask]
    map_pts = result["kp_map"][inlier_mask]

    if len(map_pts) == 0:
        return

    raw_x, raw_y = result["raw_position"]
    final_x, final_y = result["position"]

    xs = list(map_pts[:, 0]) + [raw_x, final_x]
    ys = list(map_pts[:, 1]) + [raw_y, final_y]

    x1 = int(min(xs)) - MATCH_CROP_MARGIN_PX
    x2 = int(max(xs)) + MATCH_CROP_MARGIN_PX
    y1 = int(min(ys)) - MATCH_CROP_MARGIN_PX
    y2 = int(max(ys)) + MATCH_CROP_MARGIN_PX

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    if x2 - x1 < MATCH_CROP_MIN_SIZE_PX:
        half = MATCH_CROP_MIN_SIZE_PX // 2
        x1, x2 = cx - half, cx + half

    if y2 - y1 < MATCH_CROP_MIN_SIZE_PX:
        half = MATCH_CROP_MIN_SIZE_PX // 2
        y1, y2 = cy - half, cy + half

    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(map_img.shape[1], x2)
    y2 = min(map_img.shape[0], y2)

    crop = map_img[y1:y2, x1:x2].copy()

    route_local = np.array(
        [(int(round(x - x1)), int(round(y - y1))) for x, y in waypoints],
        dtype=np.int32
    ).reshape(-1, 1, 2)

    cv2.polylines(crop, [route_local], False, (0, 165, 255), 4, cv2.LINE_AA)

    raw_local = int(round(raw_x - x1)), int(round(raw_y - y1))
    final_local = int(round(final_x - x1)), int(round(final_y - y1))

    cv2.circle(crop, raw_local, 9, (0, 0, 255), -1)
    cv2.circle(crop, final_local, 9, (0, 255, 0), -1)

    target_h = max(uav.shape[0], crop.shape[0])

    def resize_to_height(img, target):
        scale = target / img.shape[0]
        width = max(1, int(round(img.shape[1] * scale)))
        return cv2.resize(img, (width, target)), scale

    left, left_scale = resize_to_height(uav, target_h)
    right, right_scale = resize_to_height(crop, target_h)

    canvas = np.hstack([left, right])
    right_offset = left.shape[1]

    for index, (p_uav, p_map) in enumerate(zip(uav_pts, map_pts), 1):
        a = (
            int(round(p_uav[0] * left_scale)),
            int(round(p_uav[1] * left_scale))
        )
        b = (
            right_offset + int(round((p_map[0] - x1) * right_scale)),
            int(round((p_map[1] - y1) * right_scale))
        )
        cv2.line(canvas, a, b, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.circle(canvas, a, 3, (0, 255, 0), -1)
        cv2.circle(canvas, b, 3, (0, 255, 0), -1)

        if len(uav_pts) <= 20:
            cv2.putText(canvas, str(index), (a[0] + 3, a[1] - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    header = np.zeros((160, canvas.shape[1], 3), dtype=np.uint8)

    digit_note = "無三字框"
    if digit_info is not None:
        digit_note = (
            f"DigitArea={digit_info.get('digit_area_ratio', 0):.2f} "
            f"{'DOMINATED' if digit_info.get('digit_dominated') else 'ok'} "
            f"det={digit_info.get('angle')}°/{digit_info.get('score', 0):.2f} "
            f"pose={result.get('pose_model', '-')}"
        )

    line1 = (
        f"Frame={frame_index:03d} | "
        f"Semantic={semantic}({semantic_conf:.3f}) | "
        f"Second={semantic_second}({semantic_second_conf:.3f})"
    )
    line2 = (
        f"Level={level_name} | Mode={search_mode} | "
        f"Angle={result['angle']} | Tile={result['tile_id']} | "
        f"M={result['matches']} | I={result['inliers']} | "
        f"R={result['inlier_ratio']:.2f}"
    )
    line3 = (
        f"Progress={result['route_progress']:.1f}px | "
        f"SemanticRatio={result.get('semantic_ratio', 1.0):.2f} | "
        f"NonDigit={result.get('non_digit_ratio', 0):.2f} | "
        f"CenterSupport={result['center_support']:.2f}"
    )
    line4 = (
        f"Raw=({raw_x:.1f},{raw_y:.1f}) | "
        f"Final=({final_x:.1f},{final_y:.1f}) | {digit_note}"
    )

    cv2.putText(header, line1, (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(header, line2, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(header, line3, (10, 92), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1)
    cv2.putText(header, line4, (10, 122), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1)

    output = np.vstack([header, canvas])
    cv2.imwrite(str(MATCH_DATA_DIR / f"frame_{frame_index:03d}_match.jpg"), output)


# ============================================================
# C. 更新異步任務呼叫與 main()
# ============================================================

def async_save_tasks(
    best_copy,
    waypoints,
    frame_path,
    frame_index,
    semantic,
    semantic_conf,
    semantic_second,
    semantic_second_conf,
    used_level,
    search_mode,
    digit_info,
    map_base  # 新增地圖參數
):
    if SAVE_FINAL_IMAGE:
        save_final_position_image(best_copy, waypoints, map_base)

    if SAVE_MATCH_DATA:
        save_match_diagnostic(
            frame_path,
            best_copy,
            waypoints,
            frame_index,
            semantic,
            semantic_conf,
            semantic_second,
            semantic_second_conf,
            used_level,
            search_mode,
            map_base,
            digit_info=digit_info
        )


# ============================================================
# 33. Metrics
# ============================================================

def compute_metrics(errors):
    if not errors:
        return None, None

    arr = np.asarray(errors, dtype=np.float64)
    return float(np.mean(arr)), float(np.sqrt(np.mean(arr ** 2)))


# ============================================================
# 34. 主程式
# ============================================================

def main():
    global GLOBAL_MAP_IMG

    print("\n" + "=" * 78)
    print("🚀 CPU TinyCNN + SuperPoint + LightGlue VPS v2.1 (雙軸自適應與多角度檢測模組載入完成)")
    print("=" * 78)

    udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    print(f"📡 UDP 發射端已建立目標: {UDP_IP}:{UDP_PORT}")

    # A. 載入全域常駐地圖
    GLOBAL_MAP_IMG = cv2.imread(str(MAP_IMAGE_PATH))

    if GLOBAL_MAP_IMG is None:
        raise FileNotFoundError(f"找不到母圖：{MAP_IMAGE_PATH}")

    map_h, map_w = GLOBAL_MAP_IMG.shape[:2]

    if (map_w, map_h) != (MAP_WIDTH_PX, MAP_HEIGHT_PX):
        raise ValueError(f"母圖尺寸錯誤：{map_w}×{map_h}")

    semantic_model, semantic_classes, semantic_image_size = load_semantic_classifier()

    print("🧩 載入SuperPoint / LightGlue...")

    extractor = SuperPoint(
        max_num_keypoints=ONLINE_MAX_KEYPOINTS,
        detection_threshold=0.005,
        nms_radius=3
    ).eval().to(DEVICE)

    matcher = LightGlue(features="superpoint").eval().to(DEVICE)

    print("✅ SuperPoint / LightGlue完成")

    load_tile_cache()

    route_id, waypoints, route_name = select_user_route()
    total_route_length = get_route_total_length(waypoints)

    print(f"📍 航線：{route_name}")
    print(f"📏 航線長度：{total_route_length:.1f}px")

    if SAVE_MATCH_DATA:
        for old in MATCH_DATA_DIR.glob("frame_*_match.jpg"):
            try:
                old.unlink()
            except OSError:
                pass

    processed = set()

    total_frames = 0
    success_frames = 0
    fail_frames = 0

    last_angle = None
    last_progress = None

    pending_progress = None
    pending_count = 0

    SUCCESS_POSITIONS.clear()
    CORRECTED_FRAME_ERRORS.clear()

    io_executor = ThreadPoolExecutor(max_workers=1)

    try:
        while True:
            files = sorted(
                glob.glob(str(DATA_DIR / "*.jpg")) +
                glob.glob(str(DATA_DIR / "*.jpeg")) +
                glob.glob(str(DATA_DIR / "*.png"))
            )

            new_files = [f for f in files if f not in processed]

            if not new_files:
                time.sleep(0.1)
                continue

            for frame_path in new_files:
                processed.add(frame_path)
                total_frames += 1

                frame_t0 = time.perf_counter()

                gray = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)

                if gray is None:
                    print(f"❌ 無法讀取：{frame_path}")
                    continue

                print("\n" + "=" * 88)
                print(f"🖼 Frame {total_frames:03d}｜{Path(frame_path).name}")

                # ----------------------------------------------------
                # A. 三字偵測 + 最後一碼 TinyCNN
                # ----------------------------------------------------
                t0 = time.perf_counter()

                digit_info = detect_uav_digits(gray)
                semantic, semantic_conf, semantic_second, semantic_second_conf, digit_info = classify_uav_last_digit(
                    gray,
                    digit_info,
                    semantic_model,
                    semantic_classes,
                    semantic_image_size
                )

                digit_dominated = bool(digit_info and digit_info.get("digit_dominated"))

                if digit_info is None:
                    print("🧠 TinyCNN：未偵測到三字框 → 語意=NONE（不鎖車格）")
                else:
                    print(
                        f"🧠 TinyCNN：{semantic} ({semantic_conf:.3f})｜"
                        f"第二名={semantic_second} ({semantic_second_conf:.3f})｜"
                        f"DigitArea={digit_info['digit_area_ratio']:.2f}"
                        f"{' 數字主導' if digit_dominated else ''}｜"
                        f"detAngle={digit_info['angle']}° score={digit_info['score']:.2f}"
                    )

                semantic_time = time.perf_counter() - t0

                # ----------------------------------------------------
                # B. 蓋掉前導數字再跑 SuperPoint
                # ----------------------------------------------------
                t0 = time.perf_counter()

                sp_gray = mask_non_discriminative_digits(gray, digit_info)
                tensor = torch.from_numpy(sp_gray).float()[None, None].to(DEVICE) / 255.0

                with torch.inference_mode():
                    feats0 = extractor({"image": tensor})

                img_h, img_w = gray.shape[:2]

                feats0["image_size"] = torch.tensor(
                    [[float(img_w), float(img_h)]],
                    dtype=torch.float32,
                    device=DEVICE
                )

                superpoint_time = time.perf_counter() - t0

                if fail_frames >= ANGLE_RESET_AFTER_FAIL_FRAMES and last_angle is not None:
                    print(f"🔄 已失敗{fail_frames}張，舊Angle {last_angle}°失效")
                    last_angle = None

                p_min, p_max, progress_state = get_progress_window(
                    last_progress,
                    fail_frames,
                    total_route_length
                )

                print(
                    f"🧭 {progress_state}｜Progress={p_min:.1f}~{p_max:.1f}px｜"
                    f"LastAngle={'未知' if last_angle is None else str(last_angle)+'°'}｜"
                    f"Pose={'Affine' if digit_dominated else 'Homography'}"
                )

                t0 = time.perf_counter()

                best, used_level, search_mode, match_stats = hierarchical_search(
                    last_angle,
                    semantic,
                    semantic_conf,
                    route_id,
                    p_min,
                    p_max,
                    feats0,
                    gray.shape,
                    matcher,
                    waypoints,
                    digit_dominated=digit_dominated
                )

                match_time = time.perf_counter() - t0

                if best is None:
                    fail_frames += 1
                    total_time = time.perf_counter() - frame_t0

                    print("❌ 本張沒有可靠定位")
                    print(
                        f"⏱ TinyCNN={semantic_time:.3f}s｜"
                        f"SP={superpoint_time:.3f}s｜"
                        f"LG={match_time:.3f}s｜"
                        f"LG次數={match_stats['lightglue_calls']}｜"
                        f"總計={total_time:.3f}s"
                    )

                    if ENABLE_UDP_SEND:
                        fail_payload = {
                            "timestamp": time.time(),
                            "frame_id": total_frames,
                            "status": "FAIL"
                        }
                        send_udp_telemetry(udp_socket, UDP_IP, UDP_PORT, fail_payload)

                    continue

                confirmed, pending_progress, pending_count, confirm_reason = check_progress_confirmation(
                    best["route_progress"],
                    last_progress,
                    pending_progress,
                    pending_count,
                    semantic_conf
                )

                if not confirmed:
                    fail_frames += 1

                    print(
                        f"⚠️ 找到候選但暫不接受｜{confirm_reason}｜"
                        f"New={best['route_progress']:.1f}px｜Last={last_progress}"
                    )

                    if ENABLE_UDP_SEND:
                        fail_payload = {
                            "timestamp": time.time(),
                            "frame_id": total_frames,
                            "status": "PENDING"
                        }
                        send_udp_telemetry(udp_socket, UDP_IP, UDP_PORT, fail_payload)

                    continue

                raw_position = best["position"]

                corrected_position, boundary_error, boundary_corrected = apply_error_boundary(
                    raw_position,
                    waypoints
                )

                corrected_cte = compute_cte_m(corrected_position, waypoints)

                best["raw_position"] = raw_position
                best["position"] = corrected_position

                CORRECTED_FRAME_ERRORS.append(corrected_cte)
                curr_mcte, curr_rmse = compute_metrics(CORRECTED_FRAME_ERRORS)

                last_angle = best["angle"]
                last_progress = best["route_progress"]

                pending_progress = None
                pending_count = 0

                fail_frames = 0
                success_frames += 1

                if ENABLE_UDP_SEND:
                    telemetry_payload = {
                        "timestamp": time.time(),
                        "frame_id": total_frames,
                        "status": "SUCCESS",
                        "x_px": round(float(corrected_position[0]), 2),
                        "y_px": round(float(corrected_position[1]), 2),
                        "x_m": round(float(corrected_position[0] * X_GSD_M_PER_PX), 4),
                        "y_m": round(float(corrected_position[1] * Y_GSD_M_PER_PX), 4),
                        "angle_deg": int(best["angle"]),
                        "progress_px": round(float(best["route_progress"]), 2),
                        "cte_m": round(float(corrected_cte), 4),
                        "inliers": int(best["inliers"]),
                        "semantic": semantic,
                        "digit_dominated": digit_dominated
                    }
                    send_udp_telemetry(udp_socket, UDP_IP, UDP_PORT, telemetry_payload)
                    print(
                        f" 📡 UDP發射 -> {UDP_IP}:{UDP_PORT} "
                        f"[X={telemetry_payload['x_m']}m, Y={telemetry_payload['y_m']}m, Angle={best['angle']}°]"
                    )

                save_t0 = time.perf_counter()

                # C. 異步任務呼叫傳入常駐記憶體地圖
                io_executor.submit(
                    async_save_tasks,
                    best.copy(),
                    waypoints,
                    frame_path,
                    total_frames,
                    semantic,
                    semantic_conf,
                    semantic_second,
                    semantic_second_conf,
                    used_level,
                    search_mode,
                    digit_info,
                    GLOBAL_MAP_IMG  # 傳入記憶體地圖
                )

                save_time = time.perf_counter() - save_t0
                total_time = time.perf_counter() - frame_t0

                print("\n✅ 定位成功")
                print(f" ├─ Semantic：{semantic} ({semantic_conf:.3f})")
                print(f" ├─ Mode：{search_mode}")
                print(f" ├─ Level：{used_level}")
                print(f" ├─ Angle：{best['angle']}°")
                print(f" ├─ Tile：{best['tile_id']}")
                print(f" ├─ Pose：{best.get('pose_model', 'homography')}")
                print(
                    f" ├─ Match/Inlier：{best['matches']}/{best['inliers']} "
                    f"(R={best['inlier_ratio']:.2f})"
                )
                print(
                    f" ├─ NonDigit：{best.get('non_digit_ratio', 0):.2f}｜"
                    f"Spread：({best['spread_x']:.2f},{best['spread_y']:.2f})｜"
                    f"CenterSupport={best['center_support']:.2f}"
                )
                print(f" ├─ Center：{best['center_reason']}")
                print(f" ├─ Progress：{best['route_progress']:.1f}px")
                print(f" ├─ Raw VPS：({raw_position[0]:.1f}, {raw_position[1]:.1f})")
                print(f" ├─ Final VPS：({corrected_position[0]:.1f}, {corrected_position[1]:.1f})")
                print(
                    f" ├─ 單幀 CTE：{corrected_cte:.4f}m"
                    + ("｜已套0.5m限制" if boundary_corrected else "")
                )
                print(f" ├─ 累計 MCTE：{curr_mcte:.4f}m")
                print(f" └─ 累計 RMSE：{curr_rmse:.4f}m")

                print("\n⏱ 即時耗時")
                print(f" ├─ TinyCNN：{semantic_time:.3f}s")
                print(f" ├─ SuperPoint：{superpoint_time:.3f}s")
                print(f" ├─ Tile+LightGlue：{match_time:.3f}s")
                print(
                    f" ├─ Tile：{match_stats['tiles_before_progress']} → "
                    f"{match_stats['tiles_after_progress']}"
                )
                print(f" ├─ LightGlue次數：{match_stats['lightglue_calls']}")
                print(f" ├─ 圖片輸出 (提交異步)：{save_time:.3f}s")
                print(f" └─ 總計：{total_time:.3f}s")

    except KeyboardInterrupt:
        print("\n" + "=" * 70)
        print("🛑 系統停止")
        print(f"總影像：{total_frames}")
        print(f"成功定位：{success_frames}")

        if total_frames:
            print(f"定位成功率：{success_frames / total_frames * 100:.2f}%")

        print("=" * 70)
    finally:
        io_executor.shutdown(wait=False)
        udp_socket.close()


if __name__ == "__main__":
    main()