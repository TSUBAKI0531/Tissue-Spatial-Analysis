"""
spatial_analysis_tool.py - 組織切片の空間統計解析エンジン
=====================================================

概要:
    蛍光組織切片画像に対し、以下のパイプラインを実行する。
    1. Cellposeによる核セグメンテーション
    2. regionpropsによる細胞特徴量抽出
    3. 学習済みモデルによる細胞分類 (Normal / Cancer)
    4. KDTreeによる空間近接度計算
    5. Mann-Whitney U検定による群間比較

Graceful Degradation:
    - Cellpose未インストール時 → セグメンテーションをスキップし警告
    - GPU不使用環境 → CPU自動フォールバック
    - 分類モデル未ロード → ランダム分類によるデモモード
"""

import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import KDTree
from skimage import measure

from config import (
    COLUMNS_TO_DROP_FOR_PREDICTION,
    REGIONPROPS_PROPERTIES,
    AnalysisConfig,
    CellposeConfig,
)

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Cellpose Graceful Import
# ──────────────────────────────────────────────
try:
    from cellpose import models as cp_models

    CELLPOSE_AVAILABLE = True
except ImportError:
    CELLPOSE_AVAILABLE = False
    logger.warning(
        "cellpose が見つかりません。セグメンテーション機能は利用できません。"
    )

# ──────────────────────────────────────────────
# joblib Graceful Import
# ──────────────────────────────────────────────
try:
    import joblib

    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False
    logger.warning("joblib が見つかりません。モデル読み込み機能は利用できません。")


# ──────────────────────────────────────────────
# データクラス: 解析結果
# ──────────────────────────────────────────────
@dataclass
class AnalysisResult:
    """パイプライン全体の解析結果を保持するデータクラス"""

    cell_df: pd.DataFrame
    masks: Optional[np.ndarray]
    p_value: Optional[float]
    cancer_df: Optional[pd.DataFrame]
    summary: Dict[str, object]
    warnings: List[str]


class SpatialAnalyzer:
    """組織切片の空間統計解析を行うクラス

    Attributes:
        rf_model: 学習済み分類モデル (scikit-learn互換)
        cp_model: Cellposeセグメンテーションモデル
        cp_config: Cellpose設定
        analysis_config: 解析パラメータ
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        cp_config: Optional[CellposeConfig] = None,
        analysis_config: Optional[AnalysisConfig] = None,
    ):
        self.cp_config = cp_config or CellposeConfig()
        self.analysis_config = analysis_config or AnalysisConfig()
        self._warnings: List[str] = []

        # ── 分類モデル読み込み ──
        self.rf_model = self._load_model(model_path)

        # ── Cellpose初期化 ──
        self.cp_model = self._init_cellpose()

    # ──────────────────────────────────────────
    # 初期化ヘルパー
    # ──────────────────────────────────────────
    def _load_model(self, model_path: Optional[str]) -> object:
        """分類モデルを安全にロードする"""
        if model_path is None:
            logger.info("分類モデル未指定: デモモードで動作します。")
            return None

        if not JOBLIB_AVAILABLE:
            self._warnings.append("joblib未インストールのためモデルを読み込めません。")
            return None

        path = Path(model_path)
        if not path.exists():
            self._warnings.append(f"モデルファイルが見つかりません: {path}")
            return None

        try:
            model = joblib.load(path)
            logger.info(f"分類モデルを読み込みました: {path.name}")
            return model
        except Exception as e:
            self._warnings.append(f"モデル読み込みエラー: {e}")
            return None

    def _init_cellpose(self) -> object:
        """CellposeモデルをGPUフォールバック付きで初期化"""
        if not CELLPOSE_AVAILABLE:
            self._warnings.append(
                "cellpose未インストール: セグメンテーションは外部マスクが必要です。"
            )
            return None

        try:
            model = cp_models.Cellpose(
                gpu=self.cp_config.use_gpu,
                model_type=self.cp_config.model_type,
            )
            logger.info(
                f"Cellpose初期化完了 (GPU={'ON' if self.cp_config.use_gpu else 'OFF'})"
            )
            return model
        except Exception:
            # GPUフォールバック
            if self.cp_config.use_gpu:
                logger.warning("GPU初期化失敗 → CPUにフォールバック")
                self._warnings.append("GPU利用不可のためCPUモードで動作します。")
                try:
                    model = cp_models.Cellpose(
                        gpu=False,
                        model_type=self.cp_config.model_type,
                    )
                    return model
                except Exception as e:
                    self._warnings.append(f"Cellpose初期化エラー: {e}")
                    return None
            return None

    # ──────────────────────────────────────────
    # パイプラインメソッド
    # ──────────────────────────────────────────
    def segment(self, image: np.ndarray) -> np.ndarray:
        """Cellposeによる核セグメンテーション

        Args:
            image: 入力画像。shape = (C, H, W) を想定。

        Returns:
            masks: ラベルマスク (H, W)

        Raises:
            RuntimeError: Cellposeが利用不可の場合
        """
        if self.cp_model is None:
            raise RuntimeError(
                "Cellposeが利用できません。外部マスクを使用してください。"
            )

        # チャンネルファースト → 最初のチャンネルを使用
        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            input_img = image[0]
        else:
            input_img = image

        masks, flows, styles, diams = self.cp_model.eval(
            input_img,
            diameter=self.cp_config.diameter,
            channels=self.cp_config.channels,
            flow_threshold=self.cp_config.flow_threshold,
            cellprob_threshold=self.cp_config.cellprob_threshold,
        )
        logger.info(f"セグメンテーション完了: {masks.max()} 個の細胞を検出")
        return masks

    def extract_features(
        self, image: np.ndarray, masks: np.ndarray
    ) -> pd.DataFrame:
        """細胞ごとに輝度・面積・形状の特徴量を抽出

        Args:
            image: 多チャンネル画像 (C, H, W)
            masks: ラベルマスク (H, W)

        Returns:
            DataFrame: 各細胞の特徴量テーブル
        """
        # (C, H, W) → (H, W, C) に変換
        if image.ndim == 3 and image.shape[0] < image.shape[-1]:
            intensity_img = np.moveaxis(image, 0, -1)
        elif image.ndim == 2:
            intensity_img = image[:, :, np.newaxis]
        else:
            intensity_img = image

        props = measure.regionprops_table(
            masks,
            intensity_image=intensity_img,
            properties=REGIONPROPS_PROPERTIES,
        )
        df = pd.DataFrame(props)
        logger.info(f"特徴量抽出完了: {len(df)} 細胞, {len(df.columns)} 特徴量")
        return df

    def classify_cells(self, df: pd.DataFrame) -> pd.DataFrame:
        """学習済みモデルで細胞を分類 (0: Normal, 1: Cancer)

        モデル未ロード時はランダム分類を行いデモモードとして動作。

        Args:
            df: 特徴量DataFrame

        Returns:
            DataFrame: predicted_class列を追加
        """
        result_df = df.copy()

        if self.rf_model is None:
            logger.warning("分類モデル未ロード: ランダム分類でデモ動作します。")
            self._warnings.append("デモモード: ランダム分類を使用しています。")
            rng = np.random.default_rng(42)
            result_df["predicted_class"] = rng.choice(
                [0, 1], size=len(result_df), p=[0.7, 0.3]
            )
            return result_df

        # 予測に不要なカラムを除外
        feature_cols = [
            c for c in result_df.columns if c not in COLUMNS_TO_DROP_FOR_PREDICTION
        ]
        X = result_df[feature_cols]

        try:
            result_df["predicted_class"] = self.rf_model.predict(X)
        except Exception as e:
            self._warnings.append(f"分類エラー ({e}): ランダム分類にフォールバック")
            rng = np.random.default_rng(42)
            result_df["predicted_class"] = rng.choice(
                [0, 1], size=len(result_df), p=[0.7, 0.3]
            )

        n_cancer = (result_df["predicted_class"] == 1).sum()
        n_normal = (result_df["predicted_class"] == 0).sum()
        logger.info(f"分類完了: Normal={n_normal}, Cancer={n_cancer}")
        return result_df

    def compute_proximity(self, df: pd.DataFrame) -> pd.DataFrame:
        """がん細胞から最近傍の正常細胞までのユークリッド距離を計算

        数式: d = √((x₂ − x₁)² + (y₂ − y₁)²)
        KDTreeにより O(n log n) で高速に計算。

        Args:
            df: predicted_class列を含むDataFrame

        Returns:
            DataFrame: dist_to_normal列を追加
        """
        result_df = df.copy()
        result_df["dist_to_normal"] = np.nan

        cancer_mask = result_df["predicted_class"] == 1
        normal_mask = result_df["predicted_class"] == 0

        cancer_coords = result_df.loc[
            cancer_mask, ["centroid-0", "centroid-1"]
        ].values
        normal_coords = result_df.loc[
            normal_mask, ["centroid-0", "centroid-1"]
        ].values

        if len(normal_coords) == 0:
            self._warnings.append("正常細胞が検出されず、距離計算をスキップしました。")
            return result_df

        if len(cancer_coords) == 0:
            self._warnings.append("がん細胞が検出されず、距離計算をスキップしました。")
            return result_df

        tree = KDTree(normal_coords)
        distances, _ = tree.query(cancer_coords, k=self.analysis_config.kdtree_k_neighbors)
        result_df.loc[cancer_mask, "dist_to_normal"] = distances

        logger.info(
            f"距離計算完了: 中央値={np.nanmedian(distances):.1f}px, "
            f"平均={np.nanmean(distances):.1f}px"
        )
        return result_df

    def run_stat_test(
        self, df: pd.DataFrame, threshold: Optional[int] = None
    ) -> Tuple[Optional[float], Optional[pd.DataFrame]]:
        """距離閾値による2群間比較 (Mann-Whitney U検定)

        Proximal群 (距離 ≤ threshold) と Distal群の
        バイオマーカー強度を比較する。

        Args:
            df: dist_to_normal列を含むDataFrame
            threshold: 距離閾値 (px)。Noneの場合はconfig値を使用。

        Returns:
            (p_value, cancer_df): 検定のp値と群分けされたDataFrame
        """
        if threshold is None:
            threshold = self.analysis_config.proximity_threshold_px

        biomarker_col = self.analysis_config.biomarker_channel

        cancer_df = df[df["predicted_class"] == 1].copy()

        if cancer_df.empty or cancer_df["dist_to_normal"].isna().all():
            self._warnings.append("がん細胞の距離データがなく、検定をスキップしました。")
            return None, cancer_df

        cancer_df["Group"] = cancer_df["dist_to_normal"].apply(
            lambda x: "Proximal" if x <= threshold else "Distal"
        )

        # バイオマーカー列の存在確認
        if biomarker_col not in cancer_df.columns:
            available = [c for c in cancer_df.columns if "mean_intensity" in c]
            if available:
                biomarker_col = available[-1]  # 最後のチャンネルを使用
                self._warnings.append(
                    f"指定バイオマーカー列が見つからず {biomarker_col} を使用します。"
                )
            else:
                self._warnings.append("バイオマーカー列が見つかりません。")
                return None, cancer_df

        prox = cancer_df.loc[
            cancer_df["Group"] == "Proximal", biomarker_col
        ].dropna()
        dist = cancer_df.loc[
            cancer_df["Group"] == "Distal", biomarker_col
        ].dropna()

        if len(prox) < 2 or len(dist) < 2:
            self._warnings.append(
                f"サンプル数不足 (Proximal={len(prox)}, Distal={len(dist)}): "
                "検定には各群2サンプル以上が必要です。"
            )
            return None, cancer_df

        try:
            _, p_val = stats.mannwhitneyu(
                prox, dist, alternative=self.analysis_config.stat_test_alternative
            )
            logger.info(f"Mann-Whitney U検定: p={p_val:.4e}")
            return p_val, cancer_df
        except Exception as e:
            self._warnings.append(f"統計検定エラー: {e}")
            return None, cancer_df

    # ──────────────────────────────────────────
    # 統合パイプライン
    # ──────────────────────────────────────────
    def run_pipeline(
        self,
        image: np.ndarray,
        masks: Optional[np.ndarray] = None,
        threshold: Optional[int] = None,
    ) -> AnalysisResult:
        """全解析ステップを統合実行

        Args:
            image: 多チャンネル組織画像 (C, H, W)
            masks: 外部マスク。Noneの場合はCellposeで生成。
            threshold: 近接度閾値 (px)

        Returns:
            AnalysisResult: 解析結果を格納したデータクラス
        """
        self._warnings = []

        # Step 1: セグメンテーション
        if masks is None:
            masks = self.segment(image)

        # Step 2: 特徴量抽出
        df = self.extract_features(image, masks)

        if df.empty:
            return AnalysisResult(
                cell_df=df,
                masks=masks,
                p_value=None,
                cancer_df=None,
                summary={"total_cells": 0},
                warnings=self._warnings + ["細胞が検出されませんでした。"],
            )

        # Step 3: 細胞分類
        df = self.classify_cells(df)

        # Step 4: 距離計算
        df = self.compute_proximity(df)

        # Step 5: 統計検定
        p_val, cancer_df = self.run_stat_test(df, threshold=threshold)

        # サマリー生成
        n_cancer = (df["predicted_class"] == 1).sum()
        n_normal = (df["predicted_class"] == 0).sum()
        summary = {
            "total_cells": len(df),
            "normal_cells": int(n_normal),
            "cancer_cells": int(n_cancer),
            "cancer_ratio": f"{n_cancer / len(df) * 100:.1f}%",
            "p_value": p_val,
            "threshold_px": threshold or self.analysis_config.proximity_threshold_px,
        }

        if cancer_df is not None and "Group" in cancer_df.columns:
            summary["proximal_count"] = int(
                (cancer_df["Group"] == "Proximal").sum()
            )
            summary["distal_count"] = int(
                (cancer_df["Group"] == "Distal").sum()
            )

        return AnalysisResult(
            cell_df=df,
            masks=masks,
            p_value=p_val,
            cancer_df=cancer_df,
            summary=summary,
            warnings=self._warnings,
        )
