"""
config.py - Tissue Spatial Analysis 設定モジュール
全定数・デフォルト値を集約し、保守性を確保する。
"""

from dataclasses import dataclass, field
from typing import List, Optional


# ──────────────────────────────────────────────
# アプリケーション情報
# ──────────────────────────────────────────────
APP_TITLE = "🔬 Tissue Spatial Analysis Application"
APP_SUBTITLE = "組織切片の空間的細胞分布と統計解析を行うStreamlitアプリケーション"
APP_VERSION = "2.0.0"


# ──────────────────────────────────────────────
# Cellpose 設定
# ──────────────────────────────────────────────
@dataclass
class CellposeConfig:
    """Cellposeモデルの設定"""
    model_type: str = "nuclei"
    use_gpu: bool = False          # デフォルトはCPU（安全側）
    diameter: Optional[float] = None  # Noneで自動推定
    channels: List[int] = field(default_factory=lambda: [0, 0])
    flow_threshold: float = 0.4
    cellprob_threshold: float = 0.0


# ──────────────────────────────────────────────
# 解析パラメータ
# ──────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    """空間解析のパラメータ"""
    proximity_threshold_px: int = 50
    proximity_threshold_min: int = 10
    proximity_threshold_max: int = 200
    biomarker_channel: str = "mean_intensity-3"
    kdtree_k_neighbors: int = 1
    stat_test_alternative: str = "two-sided"


# ──────────────────────────────────────────────
# 可視化設定
# ──────────────────────────────────────────────
@dataclass
class VisualizationConfig:
    """プロット表示の設定"""
    scatter_size: int = 3
    scatter_cmap: str = "coolwarm"
    figure_dpi: int = 100
    color_normal: str = "#3498db"
    color_cancer: str = "#e74c3c"
    color_proximal: str = "#e67e22"
    color_distal: str = "#2ecc71"


# ──────────────────────────────────────────────
# 特徴量カラム定義
# ──────────────────────────────────────────────
REGIONPROPS_PROPERTIES = [
    "label", "centroid", "area", "mean_intensity",
    "solidity", "eccentricity", "perimeter",
]

COLUMNS_TO_DROP_FOR_PREDICTION = ["label", "predicted_class", "dist_to_normal"]


# ──────────────────────────────────────────────
# デモデータ設定
# ──────────────────────────────────────────────
DEMO_IMAGE_SHAPE = (4, 256, 256)   # (C, H, W)
DEMO_N_CELLS = 120
DEMO_CANCER_RATIO = 0.3
