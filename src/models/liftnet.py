"""
LiftNetモデルの実装

2Dポーズからカメラパラメータを利用して3Dポーズを推定するネットワーク
サイクル一貫性に基づく教師なし学習アプローチを採用
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from ..config import MODEL_CONFIG
from .camera import CameraModel


class ResidualBlock(nn.Module):
    """
    残差ブロック
    """
    def __init__(self, in_features, hidden_features):
        """
        Args:
            in_features: 入力特徴量の次元数
            hidden_features: 隠れ層の特徴量の次元数
        """
        super(ResidualBlock, self).__init__()
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        """
        Args:
            x: 入力特徴量 [バッチサイズ, in_features]
            
        Returns:
            out: 出力特徴量 [バッチサイズ, in_features]
        """
        identity = x
        
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        
        out += identity
        out = self.relu(out)
        
        return out


class DepthEstimator(nn.Module):
    """
    2Dポーズの各関節の深度を推定するネットワーク
    """
    def __init__(self, num_joints, feat_dim=1024, num_residual_blocks=3):
        """
        Args:
            num_joints: 関節の数
            feat_dim: 特徴量の次元数
            num_residual_blocks: 残差ブロックの数
        """
        super(DepthEstimator, self).__init__()
        
        # 2Dポーズとカメラパラメータの次元数
        pose_dim = num_joints * 2  # 各関節(x, y)
        camera_dim = 18  # 回転行列(9) + 並進ベクトル(3) + 内部パラメータ(6)
        input_dim = pose_dim + camera_dim
        
        # 特徴抽出ネットワーク
        self.fc1 = nn.Linear(input_dim, feat_dim)
        
        # 残差ブロック
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(feat_dim, feat_dim) for _ in range(num_residual_blocks)
        ])
        
        # 深度推定
        self.fc_out = nn.Linear(feat_dim, num_joints)
    
    def forward(self, pose_2d, cam_params):
        """
        Args:
            pose_2d: 2Dポーズ [バッチサイズ, 関節数, 2]
            cam_params: カメラパラメータ [バッチサイズ, 18]
            
        Returns:
            depth: 推定された各関節の深度 [バッチサイズ, 関節数]
        """
        batch_size = pose_2d.shape[0]
        
        # 入力を平坦化
        pose_2d_flat = pose_2d.reshape(batch_size, -1)  # [B, J*2]
        
        # 入力を結合
        x = torch.cat([pose_2d_flat, cam_params], dim=1)  # [B, J*2 + 18]
        
        # 特徴抽出
        x = F.relu(self.fc1(x))
        
        # 残差ブロックを適用
        for block in self.residual_blocks:
            x = block(x)
        
        # 深度推定
        depth = self.fc_out(x)  # [B, J]
        
        # ソフトプラス関数で深度を正の値に制限
        depth = F.softplus(depth)
        
        return depth


class LiftNet(nn.Module):
    """
    2Dポーズから3Dポーズを推定するネットワーク
    
    カメラパラメータを利用し、サイクル一貫性の制約下で学習
    """
    def __init__(self, num_joints=17, feat_dim=1024, num_residual_blocks=3):
        """
        Args:
            num_joints: 関節の数
            feat_dim: 特徴量の次元数
            num_residual_blocks: 残差ブロックの数
        """
        super(LiftNet, self).__init__()
        
        # 設定
        self.num_joints = num_joints
        self.feat_dim = feat_dim
        
        # 深度推定ネットワーク
        self.depth_estimator = DepthEstimator(
            num_joints=num_joints,
            feat_dim=feat_dim,
            num_residual_blocks=num_residual_blocks
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
    
    def forward(self, pose_2d, cam_K, cam_R, cam_t, rotation_range=None):
        """
        サイクル一貫性を利用した順伝播
        
        Args:
            pose_2d: 2Dポーズ [バッチサイズ, 関節数, 2]
            cam_K: カメラ内部パラメータ [バッチサイズ, 3, 3]
            cam_R: カメラ回転行列 [バッチサイズ, 3, 3]
            cam_t: カメラ平行移動ベクトル [バッチサイズ, 3]
            rotation_range: ランダム回転の範囲 [最小角度, 最大角度]（度数法）
            
        Returns:
            dict: 順伝播と逆伝播のポーズとカメラパラメータを含む辞書
        """
        batch_size = pose_2d.shape[0]
        device = pose_2d.device
        
        # カメラモデルを更新
        self.update_camera_batch_size(batch_size, device)
        
        # カメラパラメータを設定
        self.camera.set_intrinsics(
            focal_length=torch.stack([cam_K[:, 0, 0], cam_K[:, 1, 1]], dim=1),
            principal_point=torch.stack([cam_K[:, 0, 2], cam_K[:, 1, 2]], dim=1),
            scaling=torch.ones(batch_size, 2, device=device)
        )
        self.camera.set_extrinsics(R=cam_R, t=cam_t)
        
        # カメラパラメータを平坦化（深度推定器への入力用）
        cam_params = torch.cat([
            cam_R.reshape(batch_size, 9),     # 回転行列を平坦化
            cam_t.reshape(batch_size, 3),     # 平行移動ベクトル
            cam_K[:, 0, 0].unsqueeze(1),      # fx
            cam_K[:, 1, 1].unsqueeze(1),      # fy
            cam_K[:, 0, 2].unsqueeze(1),      # cx
            cam_K[:, 1, 2].unsqueeze(1),      # cy
            torch.ones(batch_size, 2, device=device)  # スケーリング係数
        ], dim=1)
        
        # 深度を推定
        depth = self.depth_estimator(pose_2d, cam_params)  # [B, J]
        depth = depth.unsqueeze(2)  # [B, J, 1]
        
        # 3Dポーズに変換（逆投影）
        pose_3d = self.camera.back_project(pose_2d, depth)  # [B, J, 3]
        
        # サイクルマネージャで一貫性を確保
        # ランダムな回転を適用
        if rotation_range is None:
            rotation_range = [10, 350]  # デフォルト値
        
        # ランダムな回転角度（Y軸周り）
        angles = torch.FloatTensor(batch_size).uniform_(rotation_range[0], rotation_range[1]).to(device)
        
        # 回転行列を生成
        rotations = []
        for angle in angles:
            angle_rad = angle * np.pi / 180.0
            cos_val = torch.cos(angle_rad)
            sin_val = torch.sin(angle_rad)
            
            # Y軸周りの回転行列
            rot_mat = torch.tensor([
                [cos_val, 0, sin_val],
                [0, 1, 0],
                [-sin_val, 0, cos_val]
            ], device=device)
            
            rotations.append(rot_mat)
        
        rot_matrices = torch.stack(rotations)  # [B, 3, 3]
        
        # ポーズを回転
        pose_3d_rotated = torch.bmm(pose_3d, rot_matrices.transpose(1, 2))  # [B, J, 3]
        
        # 回転されたポーズを2Dに投影
        # カメラパラメータを更新
        self.camera.set_extrinsics(R=cam_R, t=cam_t)
        
        # 投影
        pose_2d_rotated = self.camera.project(pose_3d_rotated)  # [B, J, 2]
        
        # サイクル一貫性: 回転された2Dポーズから3Dポーズを再推定
        # 回転された2Dポーズから深度を推定
        depth_rotated = self.depth_estimator(pose_2d_rotated, cam_params)  # [B, J]
        depth_rotated = depth_rotated.unsqueeze(2)  # [B, J, 1]
        
        # 3Dポーズに変換（逆投影）
        pose_3d_rec_rotated = self.camera.back_project(pose_2d_rotated, depth_rotated)  # [B, J, 3]
        
        # 逆回転を適用して元の向きに戻す
        pose_3d_rec = torch.bmm(pose_3d_rec_rotated, rot_matrices)  # [B, J, 3]
        
        # 再構築された3Dポーズから2Dポーズに投影
        pose_2d_rec = self.camera.project(pose_3d_rec)  # [B, J, 2]
        
        return {
            'pose_2d': pose_2d,                    # 入力2Dポーズ
            'pose_3d': pose_3d,                    # 推定された3Dポーズ
            'pose_3d_rotated': pose_3d_rotated,    # 回転された3Dポーズ
            'pose_2d_rotated': pose_2d_rotated,    # 投影された回転2Dポーズ
            'pose_3d_rec_rotated': pose_3d_rec_rotated,  # 回転された再構築3Dポーズ
            'pose_3d_rec': pose_3d_rec,            # 再構築された3Dポーズ
            'pose_2d_rec': pose_2d_rec,            # 再投影された2Dポーズ
            'depth': depth,                        # 推定された深度
            'depth_rotated': depth_rotated,        # 回転後の推定深度
            'rot_matrices': rot_matrices          # 適用された回転行列
        }
        
    def estimate_3d_pose(self, pose_2d, cam_K, cam_R, cam_t):
        """
        2Dポーズから3Dポーズを推定（推論時用の簡易バージョン）
        
        Args:
            pose_2d: 2Dポーズ [バッチサイズ, 関節数, 2]
            cam_K: カメラ内部パラメータ [バッチサイズ, 3, 3]
            cam_R: カメラ回転行列 [バッチサイズ, 3, 3]
            cam_t: カメラ平行移動ベクトル [バッチサイズ, 3]
            
        Returns:
            pose_3d: 推定された3Dポーズ [バッチサイズ, 関節数, 3]
        """
        batch_size = pose_2d.shape[0]
        device = pose_2d.device
        
        # カメラモデルを更新
        self.update_camera_batch_size(batch_size, device)
        
        # カメラパラメータを設定
        self.camera.set_intrinsics(
            focal_length=torch.stack([cam_K[:, 0, 0], cam_K[:, 1, 1]], dim=1),
            principal_point=torch.stack([cam_K[:, 0, 2], cam_K[:, 1, 2]], dim=1),
            scaling=torch.ones(batch_size, 2, device=device)
        )
        self.camera.set_extrinsics(R=cam_R, t=cam_t)
        
        # カメラパラメータを平坦化（深度推定器への入力用）
        cam_params = torch.cat([
            cam_R.reshape(batch_size, 9),     # 回転行列を平坦化
            cam_t.reshape(batch_size, 3),     # 平行移動ベクトル
            cam_K[:, 0, 0].unsqueeze(1),      # fx
            cam_K[:, 1, 1].unsqueeze(1),      # fy
            cam_K[:, 0, 2].unsqueeze(1),      # cx
            cam_K[:, 1, 2].unsqueeze(1),      # cy
            torch.ones(batch_size, 2, device=device)  # スケーリング係数
        ], dim=1)
        
        # 深度を推定
        depth = self.depth_estimator(pose_2d, cam_params)  # [B, J]
        depth = depth.unsqueeze(2)  # [B, J, 1]
        
        # 3Dポーズに変換（逆投影）
        pose_3d = self.camera.back_project(pose_2d, depth)  # [B, J, 3]
        
        return pose_3d