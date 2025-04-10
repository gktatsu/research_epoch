"""
RegNetモデルの実装

単一の画像から2Dポーズとカメラパラメータを同時に推定するネットワーク
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np

from ..config import MODEL_CONFIG
from .camera import CameraModel, estimate_camera_from_bbox
from .layers.capsule import CapsuleNetwork


class ContrastiveEncoder(nn.Module):
    """
    コントラスト学習で事前訓練されたエンコーダ
    
    画像から特徴ベクトルを抽出
    """
    def __init__(self, encoder_name='resnet50', pretrained=True):
        """
        Args:
            encoder_name: エンコーダのアーキテクチャ名
            pretrained: 事前訓練済みの重みを使用するかどうか
        """
        super(ContrastiveEncoder, self).__init__()
        
        # ResNet50をベースにしたエンコーダを構築
        if encoder_name == 'resnet50':
            base_model = models.resnet50(pretrained=pretrained)
            self.encoder = nn.Sequential(*list(base_model.children())[:-1])  # 最後の全結合層を除外
            self.out_features = 2048
        else:
            raise ValueError(f"未知のエンコーダアーキテクチャ: {encoder_name}")
    
    def forward(self, x):
        """
        画像から特徴ベクトルを抽出
        
        Args:
            x: 入力画像 [バッチサイズ, チャネル数, 高さ, 幅]
            
        Returns:
            features: 特徴ベクトル [バッチサイズ, out_features]
        """
        # 特徴抽出
        features = self.encoder(x)
        # 空間的次元を除去
        features = features.reshape(features.size(0), -1)
        
        return features


class CameraParamsEstimator(nn.Module):
    """
    画像のバウンディングボックスからカメラパラメータを推定
    """
    def __init__(self, image_size=(224, 224)):
        """
        Args:
            image_size: 入力画像のサイズ (幅, 高さ)
        """
        super(CameraParamsEstimator, self).__init__()
        
        self.image_size = image_size
        
        # 内部パラメータの推定モジュール（シンプルなMLPで実装）
        self.fc1 = nn.Linear(4, 64)  # バウンディングボックス [左, 上, 幅, 高さ]
        self.fc2 = nn.Linear(64, 64)
        self.fc_out = nn.Linear(64, 6)  # fx, fy, cx, cy, sx, sy
    
    def forward(self, bbox):
        """
        バウンディングボックスからカメラの内部パラメータを推定
        
        Args:
            bbox: バウンディングボックス [バッチサイズ, 4] (左, 上, 幅, 高さ)
            
        Returns:
            focal_length: 焦点距離 [バッチサイズ, 2] (fx, fy)
            principal_point: 主点 [バッチサイズ, 2] (cx, cy)
            scaling: スケーリング係数 [バッチサイズ, 2] (sx, sy)
        """
        # FCレイヤーを通して特徴量を抽出
        x = F.relu(self.fc1(bbox))
        x = F.relu(self.fc2(x))
        params = self.fc_out(x)
        
        # パラメータを分解
        focal_length = params[:, :2]  # fx, fy
        principal_point = params[:, 2:4]  # cx, cy
        scaling = params[:, 4:]  # sx, sy
        
        # スケーリング係数は正の値に制限
        scaling = F.softplus(scaling)
        
        return focal_length, principal_point, scaling


class RegNet(nn.Module):
    """
    画像から2Dポーズとカメラパラメータを推定するネットワーク
    
    コントラスト学習で事前訓練されたエンコーダとカプセルネットワークを使用
    """
    def __init__(self, num_joints=17, encoder_name='resnet50', pretrained=True, image_size=(224, 224)):
        """
        Args:
            num_joints: 関節の数
            encoder_name: エンコーダのアーキテクチャ名
            pretrained: 事前訓練済みの重みを使用するかどうか
            image_size: 入力画像のサイズ (幅, 高さ)
        """
        super(RegNet, self).__init__()
        
        self.num_joints = num_joints
        self.image_size = image_size
        
        # エンコーダ
        self.encoder = ContrastiveEncoder(encoder_name, pretrained)
        encoder_out_dim = self.encoder.out_features
        
        # カメラパラメータ推定器
        self.camera_estimator = CameraParamsEstimator(image_size)
        
        # 特徴量とカメラパラメータを結合した次元
        combined_dim = encoder_out_dim + 6  # 6 = fx, fy, cx, cy, sx, sy
        
        # カプセルネットワーク
        self.capsule_net = CapsuleNetwork(
            in_features=combined_dim,
            hidden_features=1024,
            num_pose_capsules=1,
            num_camera_capsules=1,
            num_presence_capsules=1,
            pose_dim=num_joints * 3,         # 3Dポーズ (x, y, z) x 関節数
            camera_dim=3,                    # カメラの外部パラメータ（回転角 θx, θy, 骨盤からカメラまでの距離 wp）
            presence_dim=num_joints          # 各関節の存在確率
        )
        
        # カメラモデル
        self.camera = CameraModel(batch_size=1)
    
    def update_camera_batch_size(self, batch_size, device):
        """
        バッチサイズに合わせてカメラモデルを更新
        
        Args:
            batch_size: バッチサイズ
            device: デバイス
        """
        self.camera = CameraModel(batch_size=batch_size, device=device)
    
    def forward(self, image, bbox=None):
        """
        画像から2Dポーズとカメラパラメータを推定
        
        Args:
            image: 入力画像 [バッチサイズ, チャネル数, 高さ, 幅]
            bbox: バウンディングボックス [バッチサイズ, 4] (左, 上, 幅, 高さ)
            
        Returns:
            dict: 推定された2Dポーズ、3Dポーズ、カメラパラメータ、関節存在確率
        """
        batch_size = image.shape[0]
        device = image.device
        
        # カメラモデルを更新
        self.update_camera_batch_size(batch_size, device)
        
        # バウンディングボックスが提供されていない場合は画像全体を使用
        if bbox is None:
            h, w = self.image_size
            bbox = torch.tensor([[0, 0, w, h]], device=device).repeat(batch_size, 1)
        
        # エンコーダで特徴抽出
        features = self.encoder(image)
        
        # カメラパラメータを推定
        focal_length, principal_point, scaling = self.camera_estimator(bbox)
        
        # カメラの内部パラメータを構築
        self.camera.set_intrinsics(
            focal_length=focal_length,
            principal_point=principal_point,
            scaling=scaling
        )
        
        # 特徴量とカメラパラメータを結合
        camera_params = torch.cat([focal_length, principal_point, scaling], dim=1)  # [B, 6]
        combined_features = torch.cat([features, camera_params], dim=1)  # [B, encoder_dim + 6]
        
        # カプセルネットワークで3Dポーズ、カメラの外部パラメータ、関節存在確率を推定
        pose_caps, camera_caps, presence_caps = self.capsule_net(combined_features)
        
        # カプセルの出力を整形
        pose_3d = pose_caps.reshape(batch_size, -1, 3)  # [B, num_joints, 3]
        camera_params = camera_caps.reshape(batch_size, -1)  # [B, 3]
        joint_presence = presence_caps.reshape(batch_size, -1)  # [B, num_joints]
        
        # カメラパラメータを分解
        theta_x = camera_params[:, 0]  # X軸周りの回転角
        theta_y = camera_params[:, 1]  # Y軸周りの回転角
        wp = camera_params[:, 2]       # 骨盤からカメラまでの距離（Z方向）
        
        # 回転行列を構築
        R = self._construct_rotation_matrix(theta_x, theta_y)
        
        # 平行移動ベクトルを構築（カメラ座標の原点は骨盤）
        t = self._construct_translation_vector(wp, focal_length, principal_point)
        
        # カメラの外部パラメータを設定
        self.camera.set_extrinsics(R=R, t=t)
        
        # 3Dポーズを2Dポーズに投影
        pose_2d = self.camera.project(pose_3d)
        
        return {
            'pose_2d': pose_2d,                # 推定された2Dポーズ [B, num_joints, 2]
            'pose_3d': pose_3d,                # 推定された3Dポーズ [B, num_joints, 3]
            'joint_presence': joint_presence,  # 関節存在確率 [B, num_joints]
            'cam_K': self.camera.K,            # カメラ内部パラメータ [B, 3, 3]
            'cam_R': R,                        # カメラ回転行列 [B, 3, 3]
            'cam_t': t,                        # カメラ平行移動ベクトル [B, 3]
            'focal_length': focal_length,      # 焦点距離 [B, 2]
            'principal_point': principal_point,# 主点 [B, 2]
            'scaling': scaling                 # スケーリング係数 [B, 2]
        }
    
    def _construct_rotation_matrix(self, theta_x, theta_y):
        """
        X軸とY軸周りの回転角からカメラの回転行列を構築
        
        Args:
            theta_x: X軸周りの回転角 [バッチサイズ]
            theta_y: Y軸周りの回転角 [バッチサイズ]
            
        Returns:
            R: 回転行列 [バッチサイズ, 3, 3]
        """
        batch_size = theta_x.shape[0]
        device = theta_x.device
        
        # 回転行列を初期化
        R = torch.zeros(batch_size, 3, 3, device=device)
        
        # 各バッチについて回転行列を計算
        for i in range(batch_size):
            # X軸周りの回転行列
            cos_x = torch.cos(theta_x[i])
            sin_x = torch.sin(theta_x[i])
            R_x = torch.tensor([
                [1, 0, 0],
                [0, cos_x, -sin_x],
                [0, sin_x, cos_x]
            ], device=device)
            
            # Y軸周りの回転行列
            cos_y = torch.cos(theta_y[i])
            sin_y = torch.sin(theta_y[i])
            R_y = torch.tensor([
                [cos_y, 0, sin_y],
                [0, 1, 0],
                [-sin_y, 0, cos_y]
            ], device=device)
            
            # 合成回転行列（Y軸回転の後にX軸回転）
            R[i] = torch.matmul(R_x, R_y)
        
        return R
    
    def _construct_translation_vector(self, wp, focal_length, principal_point):
        """
        カメラの平行移動ベクトルを構築
        
        Args:
            wp: 骨盤からカメラまでの距離（Z方向）[バッチサイズ]
            focal_length: 焦点距離 [バッチサイズ, 2] (fx, fy)
            principal_point: 主点 [バッチサイズ, 2] (cx, cy)
            
        Returns:
            t: 平行移動ベクトル [バッチサイズ, 3]
        """
        batch_size = wp.shape[0]
        device = wp.device
        
        # 平行移動ベクトルを計算
        t_x = -principal_point[:, 0] * wp / focal_length[:, 0]
        t_y = -principal_point[:, 1] * wp / focal_length[:, 1]
        t_z = wp
        
        # ベクトルに整形
        t = torch.stack([t_x, t_y, t_z], dim=1)  # [B, 3]
        
        return t
    
    def estimate_pose(self, image, bbox=None):
        """
        画像から2Dポーズとカメラパラメータを推定（推論時用の簡易バージョン）
        
        Args:
            image: 入力画像 [バッチサイズ, チャネル数, 高さ, 幅]
            bbox: バウンディングボックス [バッチサイズ, 4] (左, 上, 幅, 高さ)
            
        Returns:
            pose_2d: 推定された2Dポーズ [バッチサイズ, 関節数, 2]
            pose_3d: 推定された3Dポーズ [バッチサイズ, 関節数, 3]
            joint_presence: 関節存在確率 [バッチサイズ, 関節数]
            cam_K: カメラ内部パラメータ [バッチサイズ, 3, 3]
            cam_R: カメラ回転行列 [バッチサイズ, 3, 3]
            cam_t: カメラ平行移動ベクトル [バッチサイズ, 3]
        """
        # モデルの出力を取得
        outputs = self.forward(image, bbox)
        
        return (
            outputs['pose_2d'],
            outputs['pose_3d'],
            outputs['joint_presence'],
            outputs['cam_K'],
            outputs['cam_R'],
            outputs['cam_t']
        )