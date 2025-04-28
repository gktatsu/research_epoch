"""
カメラモデルの実装
透視投影カメラモデルと逆投影のためのユーティリティを提供
"""
import torch
import torch.nn as nn
import numpy as np


class CameraModel(nn.Module):
    """
    フルパースペクティブカメラモデル
    
    カメラの内部パラメータと外部パラメータを管理し、
    3Dポイントから2Dポイントへの投影と、2Dポイントから3Dポイントへの逆投影を行う
    """
    def __init__(self, batch_size=1, device='cuda'):
        """
        Args:
            batch_size: バッチサイズ
            device: デバイス ('cuda' または 'cpu')
        """
        super(CameraModel, self).__init__()
        self.batch_size = batch_size
        self.device = device
        
        # デフォルトのカメラパラメータを初期化
        # 内部パラメータ
        self.K = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
        
        # 外部パラメータ
        self.R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, 3]
        self.t = torch.zeros(batch_size, 3, 1, device=device)  # [B, 3, 1]
    
    def set_intrinsics(self, focal_length=None, principal_point=None, scaling=None):
        """
        カメラの内部パラメータを設定
        
        Args:
            focal_length: 焦点距離 [バッチサイズ, 2] (fx, fy)
            principal_point: 主点 [バッチサイズ, 2] (cx, cy)
            scaling: スケーリング係数 [バッチサイズ, 2] (sx, sy)
        """
        batch_size = self.batch_size
        device = self.device
        
        # デフォルト値を設定
        if focal_length is None:
            focal_length = torch.tensor([[1000.0, 1000.0]], device=device).repeat(batch_size, 1)
        if principal_point is None:
            principal_point = torch.tensor([[0.0, 0.0]], device=device).repeat(batch_size, 1)
        if scaling is None:
            scaling = torch.tensor([[1.0, 1.0]], device=device).repeat(batch_size, 1)
        
        # カメラ行列を構築
        K = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        
        # 各バッチのカメラ行列を更新
        K[:, 0, 0] = focal_length[:, 0] * scaling[:, 0]  # fx * sx
        K[:, 1, 1] = focal_length[:, 1] * scaling[:, 1]  # fy * sy
        K[:, 0, 2] = principal_point[:, 0]  # cx
        K[:, 1, 2] = principal_point[:, 1]  # cy
        
        self.K = K
    
    def set_extrinsics(self, R=None, t=None):
        """
        カメラの外部パラメータを設定
        
        Args:
            R: 回転行列 [バッチサイズ, 3, 3]
            t: 平行移動ベクトル [バッチサイズ, 3] または [バッチサイズ, 3, 1]
        """
        batch_size = self.batch_size
        device = self.device
        
        # デフォルト値を設定
        if R is None:
            R = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        if t is None:
            t = torch.zeros(batch_size, 3, 1, device=device)
        
        # 平行移動ベクトルの形状を調整
        if t.dim() == 2:
            t = t.unsqueeze(-1)  # [B, 3] -> [B, 3, 1]
        
        self.R = R
        self.t = t
    
    def project(self, points_3d):
        """
        3Dポイントを2Dポイントに投影
        
        Args:
            points_3d: 3Dポイント [バッチサイズ, ポイント数, 3]
            
        Returns:
            points_2d: 投影された2Dポイント [バッチサイズ, ポイント数, 2]
        """
        batch_size, num_points = points_3d.shape[0], points_3d.shape[1]
        device = points_3d.device
        
        # 同次座標に変換
        homogeneous = torch.ones(batch_size, num_points, 4, device=device)
        homogeneous[:, :, :3] = points_3d
        
        # 各バッチのポイントに変換行列を適用
        points_2d_homogeneous = []
        for i in range(batch_size):
            # 外部パラメータを適用: [R|t] * X
            RT = torch.cat([self.R[i], self.t[i]], dim=1)  # [3, 4]
            points_camera = torch.matmul(RT, homogeneous[i].transpose(0, 1))  # [3, 4] @ [4, P] -> [3, P]
            
            # 内部パラメータを適用: K * [R|t] * X
            points_image = torch.matmul(self.K[i], points_camera)  # [3, 3] @ [3, P] -> [3, P]
            
            # 同次座標から2D座標へ
            points_image = points_image.transpose(0, 1)  # [P, 3]
            
            # エラー回避のために、Z値に小さな値を加えてゼロ除算を防止
            z_values = points_image[:, 2:3].clone()
            z_values = torch.where(torch.abs(z_values) < 1e-6, torch.ones_like(z_values) * 1e-6, z_values)
            points_2d_i = points_image[:, :2] / z_values  # [P, 2]
            
            points_2d_homogeneous.append(points_2d_i)
        
        points_2d = torch.stack(points_2d_homogeneous)
        
        # 無限大やNaNをフィルタリング
        points_2d = torch.nan_to_num(points_2d, nan=0.0, posinf=1e4, neginf=-1e4)
        
        return points_2d
    
    def back_project(self, points_2d, depth):
        """
        2Dポイントと深度から3Dポイントを復元
        
        Args:
            points_2d: 2Dポイント [バッチサイズ, ポイント数, 2]
            depth: 各ポイントの深度 [バッチサイズ, ポイント数, 1]
            
        Returns:
            points_3d: 復元された3Dポイント [バッチサイズ, ポイント数, 3]
        """
        batch_size, num_points = points_2d.shape[0], points_2d.shape[1]
        device = points_2d.device
        
        # 深度方向も含めた同次座標に変換
        points_image = torch.ones(batch_size, num_points, 3, device=device)
        points_image[:, :, :2] = points_2d
        points_image = points_image * depth
        
        # 逆変換して3D座標を復元
        points_3d = []
        for i in range(batch_size):
            # 内部パラメータの逆変換を適用
            K_inv = torch.inverse(self.K[i])
            points_camera = torch.matmul(K_inv, points_image[i].transpose(0, 1))  # [3, 3] @ [3, P] -> [3, P]
            
            # 外部パラメータの逆変換を適用
            RT = torch.cat([self.R[i], self.t[i]], dim=1)  # [3, 4]
            RT_inv = torch.zeros(4, 3, device=device)
            R_inv = torch.inverse(self.R[i])
            RT_inv[:3, :] = R_inv
            RT_inv[3, :] = -torch.matmul(R_inv.transpose(0, 1), self.t[i]).squeeze()
            
            points_world = torch.matmul(RT_inv, points_camera)  # [4, 3] @ [3, P] -> [4, P]
            points_3d.append(points_world.transpose(0, 1)[:, :3])  # [P, 3]
        
        points_3d = torch.stack(points_3d)
        
        return points_3d


def estimate_camera_from_bbox(image_size, bbox, device='cuda'):
    """
    画像のバウンディングボックスからカメラの内部パラメータを推定
    
    Args:
        image_size: 元画像のサイズ (幅, 高さ)
        bbox: バウンディングボックス [左, 上, 幅, 高さ]
        device: デバイス ('cuda' または 'cpu')
        
    Returns:
        K: 推定されたカメラ内部パラメータ行列 [3, 3]
        scaling: スケーリング係数 [2]
    """
    W_full, H_full = image_size
    left, top, W_bb, H_bb = bbox
    
    # 画像の中心
    C_w = W_full / 2
    C_h = H_full / 2
    
    # カメラ行列の焦点距離を推定
    f = np.sqrt(W_full**2 + H_full**2)
    
    # 平均的な頭から骨盤までの距離（単位は固定）
    mean_height = 0.7  # 仮定値（実際には事前統計から計算）
    
    # スケーリング係数の計算
    s_w = W_full / W_bb * mean_height
    s_h = H_full / H_bb * mean_height
    
    # 主点の計算 (画像座標系)
    c_w = (C_w - left - W_bb/2) * s_w
    c_h = (C_h - top - H_bb/2) * s_h
    
    # カメラ行列の構築
    K = torch.eye(3, device=device)
    K[0, 0] = f * s_w
    K[1, 1] = f * s_h
    K[0, 2] = c_w
    K[1, 2] = c_h
    
    scaling = torch.tensor([s_w, s_h], device=device)
    
    return K, scaling