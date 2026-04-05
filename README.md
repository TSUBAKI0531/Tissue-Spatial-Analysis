# 🔬 Tissue Spatial Analysis Application

蛍光組織切片画像を対象とした、細胞の空間分布解析と統計検定を行うStreamlitアプリケーション。

## 概要

本ツールは、蛍光多重染色された組織切片画像に対して以下のパイプラインを実行します。

1. **核セグメンテーション** — Cellposeによる細胞核の自動検出
2. **特徴量抽出** — scikit-image regionpropsによる形態・輝度特徴量の定量化
3. **細胞分類** — 学習済みランダムフォレストモデルによるNormal/Cancerの二値分類
4. **空間近接度計算** — KDTreeを用いたがん細胞-正常細胞間のユークリッド距離算出
5. **統計検定** — Proximal/Distal群間のバイオマーカー強度比較（Mann-Whitney U検定）

## 主な特徴

| 機能 | 詳細 |
|------|------|
| **Graceful Degradation** | Cellpose未インストール時やGPU不使用時に自動フォールバック |
| **デモモード** | 合成データによるパイプライン動作確認（モデル・画像不要） |
| **4パネル可視化** | 空間分布 / ボックスプロット / 距離ヒストグラム / データプレビュー |
| **CSV出力** | 解析結果のダウンロード機能 |
| **Config集約** | 全パラメータをdataclassで一元管理 |

## プロジェクト構成

```
Tissue-Spatial-Analysis/
├── app.py                    # Streamlit UI & 可視化
├── spatial_analysis_tool.py  # 解析エンジン（SpatialAnalyzer クラス）
├── config.py                 # 設定・定数の集約
├── requirements.txt          # 依存パッケージ
├── .gitignore
└── README.md
```

## セットアップ

```bash
# リポジトリをクローン
git clone https://github.com/<your-username>/Tissue-Spatial-Analysis.git
cd Tissue-Spatial-Analysis

# 仮想環境の作成・有効化
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 依存パッケージのインストール
pip install -r requirements.txt

# アプリケーション起動
streamlit run app.py
```

## 使用方法

### デモモード（推奨：初回動作確認）

1. サイドバーの **「合成データでデモ実行」** をON
2. **Run Analysis** をクリック
3. 合成された120細胞のデータで全パイプラインが動作確認できます

### 実データ解析

1. サイドバーから学習済みモデル（`.joblib`）をアップロード
2. 組織画像（`.tif`、多チャンネル）をアップロード
3. Proximity Threshold 等のパラメータを調整
4. **Run Analysis** をクリック

## 入力データ仕様

| 項目 | 仕様 |
|------|------|
| **画像形式** | マルチチャンネルTIFF（shape: `C × H × W`） |
| **想定チャンネル** | ch0-2: 核・膜マーカー、ch3: バイオマーカー |
| **分類モデル** | scikit-learn互換の `.joblib` ファイル |

## 解析手法の詳細

### 空間近接度の計算

がん細胞から最近傍の正常細胞までのユークリッド距離を、SciPy KDTreeにより O(n log n) で計算します。

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

### 統計検定

距離閾値を基準にがん細胞を **Proximal群** と **Distal群** に分割し、バイオマーカー強度について **Mann-Whitney U検定**（両側検定）を実施します。

## 技術スタック

- **Python 3.10+**
- **Streamlit** — Web UI
- **Cellpose** — ディープラーニング核セグメンテーション
- **scikit-image** — 画像処理・特徴量抽出
- **scikit-learn** — 細胞分類モデル
- **SciPy** — KDTree空間検索・統計検定
- **Matplotlib / Seaborn** — 可視化

## Graceful Degradation 設計

本ツールでは、依存ライブラリが欠けていても可能な範囲で動作を継続する設計を採用しています。

- **cellpose未インストール** → 外部マスクの入力で代替可能
- **GPU不使用** → 自動的にCPUモードへフォールバック
- **分類モデル未読込** → ランダム分類によるデモ動作
- **バイオマーカー列の不一致** → 利用可能な最後のintensityチャンネルを自動選択
- **サンプル数不足** → 検定をスキップし警告を表示

## ライセンス

MIT License
