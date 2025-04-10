# EPOCH: 3Dカメラと人体ポーズの共同推定

これは[EPOCH: Jointly Estimating the 3D Pose of Cameras and Humans](https://arxiv.org/abs/2406.19726)論文の実装です。

## 概要

EPOCHフレームワークは、単一の2D画像から3Dカメラパラメータと人体の3Dポーズを同時に推定する手法です。本フレームワークは以下の2つの主要コンポーネントから構成されています：

1. **ポーズリフターネットワーク (LiftNet)** - フルパースペクティブカメラモデルを活用し、2Dポーズを正確に3Dポーズに変換します。
2. **ポーズレグレッサーネットワーク (RegNet)** - 単一画像から2Dポーズとカメラパラメータを推定します。

## 特徴

- フルパースペクティブカメラモデルによる精度の高い2D-3D変換
- 教師なし学習による3Dポーズ推定
- 単純な1x1畳み込みに基づくNormalizing Flows
- 解剖学的制約による自然なポーズ推定
- Human3.6MとMPI-INF-3DHPデータセットでの最先端の結果

## インストール

```bash
# 環境を作成
conda env create -f conda_env.yml
# 環境を有効化
conda activate epoch
```

## 使用方法

### 訓練

```bash
# LiftNetの訓練
python scripts/train_liftnet.py

# RegNetの訓練
python scripts/train_regnet.py

# EPOCHフレームワーク全体の訓練
python scripts/train_epoch.py
```

### 評価

```bash
python scripts/evaluate.py --model_path data/outputs/models/epoch_model.pth
```

### 推論

```bash
python scripts/inference.py --image_path data/examples/example.jpg --output_path data/outputs/visualizations/
```

### デモ

```bash
python scripts/demo.py
```

## プロジェクト構造

```
EPOCH/
├── requirements.txt           # 必要なPythonパッケージリスト
├── README.md                  # プロジェクト説明書
├── conda_env.yml              # Anaconda環境定義ファイル
├── conda-lock.yml             # 固定バージョンを含む環境定義
├── .gitignore                 # Gitで無視するファイル設定
│
├── src/                       # ソースコードディレクトリ
│   ├── __init__.py
│   ├── config.py              # 設定ファイル
│   ...
│
├── scripts/                   # 実行スクリプト
│   ├── preprocess.py          # データセット前処理用
│   ...
│
└── data/                      # データ保存ディレクトリ
    ├── Human3.6M/             # Human3.6Mデータセット
    ├── MPI-INF-3DHP/          # MPI-INF-3DHPデータセット
    ...
```

## 引用

```
@article{garau2024epoch,
  title={EPOCH: Jointly Estimating the 3D Pose of Cameras and Humans},
  author={Garau, Nicola and Martinelli, Giulia and Bisagno, Niccolò and Tomè, Denis and Stoll, Carsten},
  journal={arXiv preprint arXiv:2406.19726},
  year={2024}
}
```