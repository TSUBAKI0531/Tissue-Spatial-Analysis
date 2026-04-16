"""
generate_test_data.py - テスト用合成データ生成スクリプト
======================================================

生成物:
    1. test_tissue_image.tif — 4チャンネル合成組織画像
    2. test_model.joblib     — 学習済みRandomForest分類モデル

データ設計:
    - 正常細胞: 画像全体に分散配置
    - がん細胞: 2種の配置
        (a) 中央クラスター (Distal群相当): 低バイオマーカー発現
        (b) 境界部スポット (Proximal群相当): 高バイオマーカー発現
    → Mann-Whitney U検定で p < 0.05 が期待できる

使い方:
    python generate_test_data.py
    → data/ ディレクトリに2ファイルが出力されます
"""

import numpy as np
import pandas as pd
import tifffile
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from skimage.draw import disk

OUTPUT_DIR = Path("data")
OUTPUT_DIR.mkdir(exist_ok=True)

rng = np.random.default_rng(42)

# ──────────────────────────────────────────────
# パラメータ
# ──────────────────────────────────────────────
HEIGHT, WIDTH = 512, 512
N_CHANNELS = 4
N_NORMAL = 80
N_CANCER_CENTRAL = 12       # 中央クラスター = Distal群 (低バイオマーカー)
N_CANCER_PERIPHERAL = 18    # 境界部 = Proximal群 (高バイオマーカー)
CELL_RADIUS_RANGE = (7, 11)
BIOMARKER_NORMAL = 40
BIOMARKER_DISTAL = 80
BIOMARKER_PROXIMAL = 190


# ──────────────────────────────────────────────
# 1. 合成組織画像の生成
# ──────────────────────────────────────────────
def generate_tissue_image():
    """4チャンネル合成組織画像を生成する

    チャンネル構成:
        ch0: DAPI (核シグナル)
        ch1: 膜マーカー
        ch2: 構造マーカー
        ch3: バイオマーカー (Proximal cancer > Distal cancer > Normal)
    """
    image = rng.poisson(5, (N_CHANNELS, HEIGHT, WIDTH)).astype(np.float32)
    cells = []

    def draw_cell(y, x, r, label, biomarker_mean):
        """細胞を画像に描画し、メタデータを記録"""
        rr, cc = disk((y, x), r, shape=(HEIGHT, WIDTH))
        image[0, rr, cc] += rng.normal(130, 15)
        image[1, rr, cc] += rng.normal(
            90 if label == 1 else 60, 12
        )
        image[2, rr, cc] += rng.normal(
            80 if label == 1 else 50, 10
        )
        image[3, rr, cc] += rng.normal(biomarker_mean, 15)
        cells.append({
            "y": y, "x": x, "r": r, "label": label,
            "biomarker_mean": biomarker_mean,
        })

    # ── 正常細胞 (中央クラスター領域を避けて配置) ──
    # 中央 ±CENTRAL_EXCLUSION_RADIUS px を避けることで、
    # 中央のがん細胞は正常細胞から遠い=Distal群になる
    CENTRAL_EXCLUSION_RADIUS = 90
    center_y, center_x = HEIGHT / 2, WIDTH / 2

    grid_rows = int(np.ceil(np.sqrt(N_NORMAL * 1.5)))  # 除外分を考慮して多めに
    grid_spacing = min(HEIGHT, WIDTH) // (grid_rows + 1)
    placed = 0
    for gy in range(grid_rows):
        for gx in range(grid_rows):
            if placed >= N_NORMAL:
                break
            y = grid_spacing * (gy + 1) + rng.integers(-12, 12)
            x = grid_spacing * (gx + 1) + rng.integers(-12, 12)
            y = np.clip(y, 20, HEIGHT - 20)
            x = np.clip(x, 20, WIDTH - 20)

            # 中央エリアを除外
            if (y - center_y) ** 2 + (x - center_x) ** 2 < CENTRAL_EXCLUSION_RADIUS ** 2:
                continue

            r = rng.integers(*CELL_RADIUS_RANGE)
            draw_cell(y, x, r, label=0, biomarker_mean=BIOMARKER_NORMAL)
            placed += 1

    # ── がん細胞 (a): 中央クラスター (Distal想定・低バイオマーカー) ──
    # CENTRAL_EXCLUSION_RADIUS = 90 の内側に配置することで正常細胞から遠い
    # 重なりを減らすため、各細胞が前配置と min_gap 以上離れるようリトライ
    central_positions = []
    for _ in range(N_CANCER_CENTRAL):
        for _retry in range(50):
            y = int(rng.normal(center_y, 22))
            x = int(rng.normal(center_x, 22))
            if (y - center_y) ** 2 + (x - center_x) ** 2 >= 75 ** 2:
                continue
            # 既存中央cancer細胞との最小距離を確保
            too_close = any(
                (y - py) ** 2 + (x - px) ** 2 < 22 ** 2
                for py, px in central_positions
            )
            if not too_close:
                central_positions.append((y, x))
                break
        else:
            continue
        y = int(np.clip(y, 20, HEIGHT - 20))
        x = int(np.clip(x, 20, WIDTH - 20))
        r = rng.integers(CELL_RADIUS_RANGE[0] + 1, CELL_RADIUS_RANGE[1] + 2)
        draw_cell(y, x, r, label=1, biomarker_mean=BIOMARKER_DISTAL)

    # ── がん細胞 (b): 境界部スポット (Proximal想定・高バイオマーカー) ──
    # 正常細胞群に混ざって配置され、近接距離が短くなる
    spot_centers = [
        (HEIGHT * 0.18, WIDTH * 0.82),
        (HEIGHT * 0.82, WIDTH * 0.18),
        (HEIGHT * 0.20, WIDTH * 0.25),
        (HEIGHT * 0.80, WIDTH * 0.78),
    ]
    for i in range(N_CANCER_PERIPHERAL):
        cy, cx = spot_centers[i % len(spot_centers)]
        y = int(rng.normal(cy, 25))
        x = int(rng.normal(cx, 25))
        y = int(np.clip(y, 20, HEIGHT - 20))
        x = int(np.clip(x, 20, WIDTH - 20))
        r = rng.integers(CELL_RADIUS_RANGE[0] + 1, CELL_RADIUS_RANGE[1] + 2)
        draw_cell(y, x, r, label=1, biomarker_mean=BIOMARKER_PROXIMAL)

    image = np.clip(image, 0, 255).astype(np.uint16)
    return image, cells


# ──────────────────────────────────────────────
# 2. 学習済みモデルの生成
# ──────────────────────────────────────────────
def train_test_model(cells: list) -> RandomForestClassifier:
    """合成特徴量から分類モデルを学習する

    spatial_analysis_tool.extract_features() が出力する
    特徴量カラムに合わせて学習データを構築。
    """
    records = []
    for c in cells:
        is_cancer = c["label"] == 1
        records.append({
            "centroid-0": c["y"] + rng.normal(0, 1),
            "centroid-1": c["x"] + rng.normal(0, 1),
            "area": np.pi * c["r"] ** 2 + rng.normal(0, 5),
            "mean_intensity-0": rng.normal(130, 15),
            "mean_intensity-1": rng.normal(90 if is_cancer else 60, 10),
            "mean_intensity-2": rng.normal(80 if is_cancer else 50, 10),
            "mean_intensity-3": rng.normal(c["biomarker_mean"], 15),
            "solidity": rng.uniform(0.85, 0.98),
            "eccentricity": rng.uniform(0.1, 0.5),
            "perimeter": 2 * np.pi * c["r"] + rng.normal(0, 2),
        })

    df = pd.DataFrame(records)
    labels = [c["label"] for c in cells]

    clf = RandomForestClassifier(
        n_estimators=50,
        max_depth=5,
        random_state=42,
        class_weight="balanced",
    )
    clf.fit(df, labels)
    print(f"  Training accuracy: {clf.score(df, labels):.3f}")
    return clf


# ──────────────────────────────────────────────
# メイン
# ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 50)
    print("テストデータ生成スクリプト")
    print("=" * 50)

    print("\n[1/2] 合成組織画像を生成中...")
    image, cells = generate_tissue_image()
    img_path = OUTPUT_DIR / "test_tissue_image.tif"
    # axes='CYX' を明示してチャンネル軸を保持
    tifffile.imwrite(str(img_path), image, metadata={"axes": "CYX"})
    print(f"  保存: {img_path}")
    print(f"  Shape: {image.shape} (CYX)")
    print(f"  細胞数: Normal={N_NORMAL}, "
          f"Cancer(central)={N_CANCER_CENTRAL}, "
          f"Cancer(peripheral)={N_CANCER_PERIPHERAL}")

    print("\n[2/2] 分類モデルを学習中...")
    model = train_test_model(cells)
    model_path = OUTPUT_DIR / "test_model.joblib"
    joblib.dump(model, str(model_path))
    print(f"  保存: {model_path}")

    print(f"\n✅ 完了!")
    print(f"  📷 {img_path}  ({img_path.stat().st_size / 1024:.0f} KB)")
    print(f"  🤖 {model_path}  ({model_path.stat().st_size / 1024:.0f} KB)")
    print(f"\n使い方:")
    print(f"  1. streamlit run app.py")
    print(f"  2. サイドバーから test_model.joblib をアップロード")
    print(f"  3. test_tissue_image.tif をアップロード")
    print(f"  4. Run Analysis をクリック")
    print(f"  → Proximal群のバイオマーカー > Distal群 で p < 0.05 を期待")
