"""
EPOCHフレームワークの設定ファイル
"""
import os
from pathlib import Path

# プロジェクトのルートディレクトリ
ROOT_DIR = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# データディレクトリ
DATA_DIR = ROOT_DIR / 'data'
HUMAN36M_DIR = DATA_DIR / 'Human3.6M'
MPI_INF_3DHP_DIR = DATA_DIR / 'MPI-INF-3DHP'
EXAMPLES_DIR = DATA_DIR / 'examples'

# 出力ディレクトリ
OUTPUT_DIR = ROOT_DIR / 'outputs'
MODELS_DIR = OUTPUT_DIR / 'models'
LOGS_DIR = OUTPUT_DIR / 'logs'
VISUALIZATIONS_DIR = OUTPUT_DIR / 'visualizations'

# 各ディレクトリが存在することを確認
for dir_path in [DATA_DIR, HUMAN36M_DIR, MPI_INF_3DHP_DIR, EXAMPLES_DIR, 
                OUTPUT_DIR, MODELS_DIR, LOGS_DIR, VISUALIZATIONS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# モデル設定
MODEL_CONFIG = {
    # LiftNet設定
    'lift_net': {
        'dim_l': 1024,         # 埋め込みベクトルサイズ
        'num_joints': 17,      # 関節数
        'residual_blocks': 3,  # 残差ブロック数
    },
    
    # RegNet設定
    'reg_net': {
        'encoder': 'resnet50',  # エンコーダアーキテクチャ
        'dim': 2048 + 6,        # 特徴量ベクトルサイズ（ResNet50出力 + 内部パラメータ数）
        'num_joints': 17,       # 関節数
    },
    
    # Normalizing Flow設定
    'normalizing_flow': {
        'num_flow_blocks': 6,   # フローブロック数
        'hidden_channels': 128, # 隠れチャネル数
    }
}

# 訓練設定
TRAIN_CONFIG = {
    # LiftNet訓練設定
    'lift_net': {
        'batch_size': 256,
        'lr': 2e-4,
        'weight_decay': 1e-5,
        'epochs': 100,
        'optimizer': 'adamw',
        'rotation_range': [10, 350],  # 回転範囲（度）
    },
    
    # RegNet訓練設定
    'reg_net': {
        'batch_size': 256,
        'lr': 1e-3,
        'weight_decay': 1e-4,
        'epochs': 45,
        'optimizer': 'adamw',
    },
    
    # EPOCH統合モデル訓練設定
    'epoch': {
        'batch_size': 128,
        'lr': 5e-4,
        'weight_decay': 1e-5,
        'epochs': 2,
        'optimizer': 'adamw',
    },
    
    # 損失関数の重み
    'loss_weights': {
        'lift_net': {
            'l2d': 10.0,        # 2Dサイクル一貫性損失
            'l3d': 1.0,         # 3Dサイクル一貫性損失
            'nf': 1.0,          # Normalizing Flow損失
            'bone': 10.0,       # 骨長比率損失
            'limbs': 0.1,       # 関節曲げ制約損失
            'def': 1.0,         # 変形損失
        },
        'reg_net': {
            'rle': 10.0,        # 残差対数尤度推定損失
            'nf': 1.0,          # Normalizing Flow損失
            'bone': 1.0,        # 骨長比率損失
            'limbs': 0.1,       # 関節曲げ制約損失
        },
        'epoch': {
            'rle': 10.0,        # 残差対数尤度推定損失（RegNetから）
            'bone': 5.0,        # 骨長比率損失（両方のモデルで使用）
            'limbs': 0.1,       # 関節曲げ制約損失（両方のモデルで使用）
            'nf': 1.0,          # Normalizing Flow損失（両方のモデルで使用）
            'l2d': 10.0,        # 2Dサイクル一貫性損失（LiftNetから）
            'l3d': 1.0,         # 3Dサイクル一貫性損失（LiftNetから）
            'def': 1.0,         # 変形損失（LiftNetから）
        }
    }
}

# データセット設定
DATASET_CONFIG = {
    'human36m': {
        'train_subjects': ['S1', 'S5', 'S6', 'S7', 'S8'],
        'test_subjects': ['S9', 'S11'],
        'actions': ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 
                    'Posing', 'Purchases', 'Sitting', 'SittingDown', 'Smoking', 
                    'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether'],
        'image_size': (224, 224),
    },
    'mpi_inf_3dhp': {
        'train_sequences': [1, 2, 3, 4, 5, 6, 7],
        'test_sequences': [8],
        'image_size': (224, 224),
    }
}