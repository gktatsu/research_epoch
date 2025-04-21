"""
カプセルネットワークの実装

RegNetモデルで使用されるカプセルネットワークレイヤー
特徴ベクトルから複数のカプセルを作成し、エンティティの存在と属性を表現
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class RoutingLayer(nn.Module):
    """
    ソフトアテンションに基づくルーティングレイヤー
    
    入力特徴を異なるカプセルに割り当てるためのルーティング機構
    """
    def __init__(self, num_capsules, in_features):
        """
        Args:
            num_capsules: カプセルの数
            in_features: 入力特徴量の次元数
        """
        super(RoutingLayer, self).__init__()
        
        # 各カプセルへのルーティングロジット
        self.route_weights = nn.Parameter(torch.randn(num_capsules, in_features))
    
    def forward(self, x):
        """
        ソフトアテンションに基づくルーティング
        
        Args:
            x: 入力特徴 [バッチサイズ, in_features]
            
        Returns:
            attention: 各カプセルへのアテンション重み [バッチサイズ, num_capsules]
        """
        # 各カプセルのルーティングロジットを計算
        # [B, in] @ [num_caps, in].T -> [B, num_caps]
        logits = torch.matmul(x, self.route_weights.transpose(0, 1))
        
        # ソフトマックスでアテンション重みに変換
        attention = F.softmax(logits, dim=1)
        
        return attention


class CapsuleLayer(nn.Module):
    """
    カプセルレイヤー
    
    複数のカプセルを生成し、それぞれが特定のエンティティの存在確率と属性を表現
    """
    def __init__(self, num_capsules, in_features, out_features):
        """
        Args:
            num_capsules: カプセルの数
            in_features: 入力特徴量の次元数
            out_features: 出力カプセルベクトルの次元数
        """
        super(CapsuleLayer, self).__init__()
        
        self.num_capsules = num_capsules
        self.in_features = in_features
        self.out_features = out_features
        
        # カプセルの変換行列
        self.W = nn.Parameter(torch.randn(num_capsules, in_features, out_features))
        
        # ルーティングレイヤー
        self.routing = RoutingLayer(num_capsules, in_features)
    
    def forward(self, x):
        """
        カプセル変換
        
        Args:
            x: 入力特徴 [バッチサイズ, in_features]
            
        Returns:
            capsules: カプセルベクトル [バッチサイズ, num_capsules, out_features]
        """
        batch_size = x.shape[0]
        
        # ルーティングアテンションを計算
        attention = self.routing(x)  # [B, num_caps]
        
        # 入力を拡張して各カプセルに適用
        x_expand = x.unsqueeze(1).expand(batch_size, self.num_capsules, self.in_features)  # [B, num_caps, in]
        
        # 変換行列をバッチサイズに拡張
        W_expanded = self.W.unsqueeze(0).expand(batch_size, self.num_capsules, self.in_features, self.out_features)
        
        # バッチマトリックス乗算で変換
        # [B, num_caps, in, 1] @ [B, num_caps, in, out] -> [B, num_caps, 1, out]
        transformed = torch.matmul(x_expand.unsqueeze(-1).transpose(-1, -2), W_expanded)
        transformed = transformed.squeeze(-2)  # [B, num_caps, out]
        
        # アテンション重みを適用
        # [B, num_caps, 1] * [B, num_caps, out] -> [B, num_caps, out]
        capsules = attention.unsqueeze(2) * transformed
        
        return capsules


class CapsuleNetwork(nn.Module):
    """
    カプセルネットワーク
    
    入力特徴から複数のエンティティ表現（カプセル）を生成
    RegNetモデルの一部として使用
    """
    def __init__(self, in_features, hidden_features, num_pose_capsules, num_camera_capsules, num_presence_capsules, 
                 pose_dim, camera_dim, presence_dim):
        """
        Args:
            in_features: 入力特徴量の次元数
            hidden_features: 隠れ層の特徴量の次元数
            num_pose_capsules: ポーズカプセルの数
            num_camera_capsules: カメラパラメータカプセルの数
            num_presence_capsules: 存在確率カプセルの数
            pose_dim: ポーズベクトルの次元数
            camera_dim: カメラパラメータベクトルの次元数
            presence_dim: 存在確率ベクトルの次元数
        """
        super(CapsuleNetwork, self).__init__()
        
        # 入力特徴量を隠れ層に変換
        self.fc = nn.Linear(in_features, hidden_features)
        
        # カプセルレイヤー
        self.pose_capsule = CapsuleLayer(num_pose_capsules, hidden_features, pose_dim)
        self.camera_capsule = CapsuleLayer(num_camera_capsules, hidden_features, camera_dim)
        self.presence_capsule = CapsuleLayer(num_presence_capsules, hidden_features, presence_dim)
    
    def forward(self, x):
        """
        カプセルネットワークの順伝播
        
        Args:
            x: 入力特徴 [バッチサイズ, in_features]
            
        Returns:
            pose_caps: ポーズカプセル [バッチサイズ, num_pose_capsules, pose_dim]
            camera_caps: カメラパラメータカプセル [バッチサイズ, num_camera_capsules, camera_dim]
            presence_caps: 存在確率カプセル [バッチサイズ, num_presence_capsules, presence_dim]
        """
        # 入力特徴量を変換
        h = F.relu(self.fc(x))
        
        # 各カプセルレイヤーで変換
        pose_caps = self.pose_capsule(h)
        camera_caps = self.camera_capsule(h)
        presence_caps = self.presence_capsule(h)
        
        return pose_caps, camera_caps, presence_caps