"""
app.py - Tissue Spatial Analysis Streamlit アプリケーション
=========================================================

機能:
    - 学習済みモデル (.joblib) と組織画像 (.tif) のアップロード
    - Cellposeセグメンテーション → 細胞分類 → 空間解析の統合パイプライン
    - デモモード: モデル/画像なしでも合成データで動作確認可能
    - 4パネル可視化: 空間分布 / 統計検定 / 距離ヒストグラム / サマリー
"""

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from skimage import io

from config import (
    APP_SUBTITLE,
    APP_TITLE,
    APP_VERSION,
    DEMO_CANCER_RATIO,
    DEMO_IMAGE_SHAPE,
    DEMO_N_CELLS,
    AnalysisConfig,
    CellposeConfig,
    VisualizationConfig,
)
from spatial_analysis_tool import SpatialAnalyzer, AnalysisResult

# ──────────────────────────────────────────────
# ページ設定
# ──────────────────────────────────────────────
st.set_page_config(page_title="Tissue Spatial Analysis", layout="wide")
st.title(APP_TITLE)
st.caption(f"{APP_SUBTITLE}　|　v{APP_VERSION}")

viz_config = VisualizationConfig()


# ──────────────────────────────────────────────
# セッション状態の初期化
# ──────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = False


# ──────────────────────────────────────────────
# デモデータ生成
# ──────────────────────────────────────────────
def generate_demo_data() -> AnalysisResult:
    """合成データでパイプラインをデモ実行する"""
    rng = np.random.default_rng(42)

    # 合成細胞データ
    n = DEMO_N_CELLS
    n_cancer = int(n * DEMO_CANCER_RATIO)
    n_normal = n - n_cancer

    h, w = DEMO_IMAGE_SHAPE[1], DEMO_IMAGE_SHAPE[2]

    # 正常細胞: 全体に分布
    normal_y = rng.uniform(20, h - 20, n_normal)
    normal_x = rng.uniform(20, w - 20, n_normal)

    # がん細胞: 中央付近にクラスター
    cancer_y = rng.normal(h / 2, h / 6, n_cancer)
    cancer_x = rng.normal(w / 2, w / 6, n_cancer)

    df = pd.DataFrame(
        {
            "label": range(1, n + 1),
            "centroid-0": np.concatenate([normal_y, cancer_y]),
            "centroid-1": np.concatenate([normal_x, cancer_x]),
            "area": rng.normal(150, 30, n).clip(50),
            "mean_intensity-0": rng.normal(100, 20, n),
            "mean_intensity-1": rng.normal(80, 15, n),
            "mean_intensity-2": rng.normal(90, 18, n),
            "mean_intensity-3": np.concatenate(
                [
                    rng.normal(60, 15, n_normal),   # Normal: 低バイオマーカー
                    rng.normal(120, 25, n_cancer),   # Cancer: 高バイオマーカー
                ]
            ),
            "solidity": rng.uniform(0.8, 1.0, n),
            "eccentricity": rng.uniform(0.1, 0.7, n),
            "perimeter": rng.normal(45, 8, n).clip(10),
            "predicted_class": np.concatenate(
                [np.zeros(n_normal), np.ones(n_cancer)]
            ).astype(int),
        }
    )

    # 距離計算を実際に実行
    analyzer = SpatialAnalyzer(model_path=None)
    df = analyzer.compute_proximity(df)
    p_val, cancer_df = analyzer.run_stat_test(df, threshold=50)

    n_cancer_actual = int((df["predicted_class"] == 1).sum())
    n_normal_actual = int((df["predicted_class"] == 0).sum())
    summary = {
        "total_cells": len(df),
        "normal_cells": n_normal_actual,
        "cancer_cells": n_cancer_actual,
        "cancer_ratio": f"{n_cancer_actual / len(df) * 100:.1f}%",
        "p_value": p_val,
        "threshold_px": 50,
    }

    if cancer_df is not None and "Group" in cancer_df.columns:
        summary["proximal_count"] = int((cancer_df["Group"] == "Proximal").sum())
        summary["distal_count"] = int((cancer_df["Group"] == "Distal").sum())

    return AnalysisResult(
        cell_df=df,
        masks=None,
        p_value=p_val,
        cancer_df=cancer_df,
        summary=summary,
        warnings=["デモモード: 合成データを使用しています。"],
    )


# ──────────────────────────────────────────────
# 可視化関数
# ──────────────────────────────────────────────
def plot_spatial_distribution(df: pd.DataFrame) -> plt.Figure:
    """細胞の空間分布プロット"""
    fig, ax = plt.subplots(figsize=(6, 6))

    normal = df[df["predicted_class"] == 0]
    cancer = df[df["predicted_class"] == 1]

    ax.scatter(
        normal["centroid-1"],
        normal["centroid-0"],
        c=viz_config.color_normal,
        s=viz_config.scatter_size,
        label=f"Normal (n={len(normal)})",
        alpha=0.6,
    )
    ax.scatter(
        cancer["centroid-1"],
        cancer["centroid-0"],
        c=viz_config.color_cancer,
        s=viz_config.scatter_size + 2,
        label=f"Cancer (n={len(cancer)})",
        alpha=0.7,
    )
    ax.invert_yaxis()
    ax.set_xlabel("X (px)")
    ax.set_ylabel("Y (px)")
    ax.set_title("Cell Spatial Distribution")
    ax.legend(loc="upper right", fontsize=8, markerscale=4)
    ax.set_aspect("equal")
    fig.tight_layout()
    return fig


def plot_boxplot(
    cancer_df: pd.DataFrame, biomarker_col: str
) -> plt.Figure:
    """Proximal / Distal 群のバイオマーカー比較ボックスプロット"""
    fig, ax = plt.subplots(figsize=(5, 5))
    palette = {
        "Proximal": viz_config.color_proximal,
        "Distal": viz_config.color_distal,
    }
    sns.boxplot(
        data=cancer_df,
        x="Group",
        y=biomarker_col,
        palette=palette,
        ax=ax,
        width=0.5,
    )
    sns.stripplot(
        data=cancer_df,
        x="Group",
        y=biomarker_col,
        color="black",
        alpha=0.3,
        size=3,
        ax=ax,
    )
    ax.set_title("Biomarker Intensity by Proximity Group")
    ax.set_ylabel("Mean Intensity (Biomarker)")
    fig.tight_layout()
    return fig


def plot_distance_histogram(cancer_df: pd.DataFrame, threshold: int) -> plt.Figure:
    """がん細胞の距離分布ヒストグラム"""
    fig, ax = plt.subplots(figsize=(5, 4))
    distances = cancer_df["dist_to_normal"].dropna()

    ax.hist(
        distances,
        bins=30,
        color=viz_config.color_cancer,
        alpha=0.7,
        edgecolor="white",
    )
    ax.axvline(
        threshold,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"Threshold = {threshold} px",
    )
    ax.set_xlabel("Distance to Nearest Normal Cell (px)")
    ax.set_ylabel("Count")
    ax.set_title("Distance Distribution")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# ──────────────────────────────────────────────
# サイドバー
# ──────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    st.subheader("Demo Mode")
    demo_mode = st.toggle("合成データでデモ実行", value=False)

    st.divider()

    st.subheader("Model & Image")
    model_file = st.file_uploader(
        "Upload Classification Model (.joblib)",
        type=["joblib"],
        disabled=demo_mode,
    )
    image_file = st.file_uploader(
        "Upload Tissue Image (.tif)",
        type=["tif", "tiff"],
        disabled=demo_mode,
    )

    st.divider()

    st.subheader("Analysis Parameters")
    analysis_cfg = AnalysisConfig()
    dist_thresh = st.slider(
        "Proximity Threshold (px)",
        min_value=analysis_cfg.proximity_threshold_min,
        max_value=analysis_cfg.proximity_threshold_max,
        value=analysis_cfg.proximity_threshold_px,
        step=5,
        help="がん細胞をProximal / Distalに分ける距離閾値",
    )

    use_gpu = st.checkbox("Use GPU (Cellpose)", value=False)

    st.divider()

    st.subheader("Visualization")
    scatter_size = st.slider("Scatter Point Size", 1, 10, viz_config.scatter_size)
    viz_config.scatter_size = scatter_size

    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)


# ──────────────────────────────────────────────
# メイン処理
# ──────────────────────────────────────────────
if run_button:
    if demo_mode:
        with st.spinner("デモデータを生成中..."):
            st.session_state.result = generate_demo_data()
            st.session_state.demo_mode = True

    elif model_file is None:
        st.error("サイドバーから分類モデル (.joblib) をアップロードしてください。")

    elif image_file is None:
        st.error("組織画像 (.tif) をアップロードしてください。")

    else:
        with st.spinner("解析を実行中... (セグメンテーション → 分類 → 空間解析)"):
            try:
                # 一時ファイルに保存
                with tempfile.NamedTemporaryFile(
                    suffix=".joblib", delete=False
                ) as tmp_model:
                    tmp_model.write(model_file.getbuffer())
                    tmp_model_path = tmp_model.name

                with tempfile.NamedTemporaryFile(
                    suffix=".tif", delete=False
                ) as tmp_img:
                    tmp_img.write(image_file.getbuffer())
                    tmp_img_path = tmp_img.name

                # 画像読み込み
                img = io.imread(tmp_img_path)

                # Analyzer初期化 & パイプライン実行
                cp_cfg = CellposeConfig(use_gpu=use_gpu)
                analyzer = SpatialAnalyzer(
                    model_path=tmp_model_path,
                    cp_config=cp_cfg,
                )
                result = analyzer.run_pipeline(
                    image=img, threshold=dist_thresh
                )
                st.session_state.result = result
                st.session_state.demo_mode = False

                # 一時ファイル削除
                Path(tmp_model_path).unlink(missing_ok=True)
                Path(tmp_img_path).unlink(missing_ok=True)

            except Exception as e:
                st.error(f"解析中にエラーが発生しました: {e}")
                st.session_state.result = None


# ──────────────────────────────────────────────
# 結果表示
# ──────────────────────────────────────────────
result: AnalysisResult = st.session_state.result

if result is None:
    st.info(
        "サイドバーからモデルと画像をアップロードして **Run Analysis** を押してください。\n\n"
        "または **Demo Mode** をONにして合成データで動作を確認できます。"
    )
else:
    # 警告表示
    for w in result.warnings:
        st.warning(w)

    # ── サマリーメトリクス ──
    st.subheader("📊 Analysis Summary")
    m_cols = st.columns(5)

    m_cols[0].metric("Total Cells", result.summary.get("total_cells", "—"))
    m_cols[1].metric("Normal Cells", result.summary.get("normal_cells", "—"))
    m_cols[2].metric("Cancer Cells", result.summary.get("cancer_cells", "—"))
    m_cols[3].metric("Cancer Ratio", result.summary.get("cancer_ratio", "—"))

    p_val = result.summary.get("p_value")
    if p_val is not None:
        p_display = f"{p_val:.4e}"
        significance = " ✱" if p_val < 0.05 else ""
        m_cols[4].metric("P-Value (MWU)", p_display + significance)
    else:
        m_cols[4].metric("P-Value (MWU)", "N/A")

    st.divider()

    # ── 4パネル可視化 ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🗺️ Spatial Distribution")
        fig1 = plot_spatial_distribution(result.cell_df)
        st.pyplot(fig1)
        plt.close(fig1)

    with col2:
        if result.cancer_df is not None and "Group" in result.cancer_df.columns:
            # バイオマーカー列を自動検出
            biomarker_col = "mean_intensity-3"
            if biomarker_col not in result.cancer_df.columns:
                intensity_cols = [
                    c for c in result.cancer_df.columns if "mean_intensity" in c
                ]
                biomarker_col = intensity_cols[-1] if intensity_cols else None

            if biomarker_col:
                st.subheader("📈 Biomarker Comparison")
                fig2 = plot_boxplot(result.cancer_df, biomarker_col)
                st.pyplot(fig2)
                plt.close(fig2)
            else:
                st.info("バイオマーカー列が見つかりません。")
        else:
            st.info("統計検定を実行できませんでした。")

    col3, col4 = st.columns(2)

    with col3:
        if result.cancer_df is not None and "dist_to_normal" in result.cancer_df.columns:
            st.subheader("📏 Distance Distribution")
            threshold = result.summary.get("threshold_px", 50)
            fig3 = plot_distance_histogram(result.cancer_df, threshold)
            st.pyplot(fig3)
            plt.close(fig3)

    with col4:
        st.subheader("📋 Cell Data Preview")
        display_cols = [
            c
            for c in [
                "label",
                "predicted_class",
                "area",
                "dist_to_normal",
                "solidity",
            ]
            if c in result.cell_df.columns
        ]
        st.dataframe(
            result.cell_df[display_cols].head(20),
            use_container_width=True,
            hide_index=True,
        )

    # ── ダウンロード ──
    st.divider()
    csv_data = result.cell_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Download Cell Data (CSV)",
        data=csv_data,
        file_name="spatial_analysis_results.csv",
        mime="text/csv",
    )
