"""
Normalizing Flowモデルの実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ActNorm(nn.Module):
    """
    アクティベーション正規化層
    
    入力をバッチ正規化するためのスケールとバイアスパラメータを学習
    """
    def __init__(self, num_features):
        """
        Args:
            num_features: 特徴量の次元数
        """
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.register_parameter("bias", nn.Parameter(torch.zeros(1, num_features)))
        self.register_parameter("logs", nn.Parameter(torch.zeros(1, num_features)))
        self.initialized = False
    
    def initialize(self, x):
        """
        データバッチを使用してパラメータを初期化
        
        Args:
            x: 入力データ [B, D]
        """
        if self.initialized:
            return
        
        # バッチ統計からパラメータを計算
        mean = x.mean(dim=0, keepdim=True)
        std = x.std(dim=0, keepdim=True)
        
        # バイアスを設定
        self.bias.data.copy_(-mean)
        
        # スケールを設定
        self.logs.data.copy_(-torch.log(std + 1e-6))
        
        self.initialized = True
    
    def forward(self, x, reverse=False):
        """
        順方向および逆方向変換
        
        Args:
            x: 入力データ [B, D]
            reverse: 逆方向変換を行うかどうか
            
        Returns:
            y: 変換された出力
            ldj: log det jacobian
        """
        # データの形状を取得
        batch_size = x.shape[0]
        
        # パラメータが初期化されていなければ初期化
        if not self.initialized:
            self.initialize(x)
        
        # スケールを計算
        scale = torch.exp(self.logs)
        
        # 順方向変換
        if not reverse:
            # 正規化: y = (x + bias) * scale
            y = (x + self.bias) * scale
            # log det jacobianの計算
            ldj = torch.sum(self.logs) * batch_size
        # 逆方向変換
        else:
            # 逆正規化: x = y / scale - bias
            y = x / scale - self.bias
            # log det jacobianの計算
            ldj = -torch.sum(self.logs) * batch_size
        
        return y, ldj


class InvertibleConv1x1(nn.Module):
    """
    1x1可逆畳み込み層
    
    特徴量間の相関関係を学習するための可逆線形変換
    """
    def __init__(self, num_features):
        """
        Args:
            num_features: 特徴量の次元数
        """
        super(InvertibleConv1x1, self).__init__()
        
        # ランダムな直交行列で初期化
        W = torch.qr(torch.randn(num_features, num_features))[0]
        
        # 行列をパラメータとして登録
        self.register_parameter("weight", nn.Parameter(W))
    
    def forward(self, x, reverse=False):
        """
        順方向および逆方向変換
        
        Args:
            x: 入力データ [B, D]
            reverse: 逆方向変換を行うかどうか
            
        Returns:
            y: 変換された出力
            ldj: log det jacobian
        """
        # データの形状を取得
        batch_size = x.shape[0]
        
        # 重み行列
        weight = self.weight
        
        # 行列式の対数を計算
        ldj = torch.slogdet(weight)[1] * batch_size
        
        # 順方向変換
        if not reverse:
            # 線形変換: y = x * W
            y = F.linear(x, weight)
        # 逆方向変換
        else:
            # 逆変換: x = y * W^(-1)
            y = F.linear(x, weight.inverse())
            ldj = -ldj
        
        return y, ldj


class AffineCoupling(nn.Module):
    """
    アフィン結合層
    
    入力の一部を使用して残りの部分に非線形変換を適用
    """
    def __init__(self, num_features, hidden_features=128):
        """
        Args:
            num_features: 特徴量の次元数
            hidden_features: 隠れ層の次元数
        """
        super(AffineCoupling, self).__init__()
        
        # 特徴量を分割
        self.num_features = num_features
        self.hidden_features = hidden_features
        self.split_size = num_features // 2
        
        # 変換ネットワーク
        self.net = nn.Sequential(
            nn.Linear(self.split_size, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, (num_features - self.split_size) * 2)
        )
        
        # スケールパラメータの初期化
        self.net[-1].weight.data.zero_()
        self.net[-1].bias.data.zero_()
    
    def forward(self, x, reverse=False):
        """
        順方向および逆方向変換
        
        Args:
            x: 入力データ [B, D]
            reverse: 逆方向変換を行うかどうか
            
        Returns:
            y: 変換された出力
            ldj: log det jacobian
        """
        # 入力を分割
        x1, x2 = torch.split(x, [self.split_size, self.num_features - self.split_size], dim=1)
        
        # x1に基づいて変換パラメータを計算
        h = self.net(x1)
        scale, shift = torch.chunk(h, 2, dim=1)
        
        # 数値安定性のためスケールにsigmoidを適用
        scale = torch.sigmoid(scale + 2) + 1e-6
        
        # 順方向変換
        if not reverse:
            # アフィン変換: y2 = x2 * scale + shift
            y2 = x2 * scale + shift
            y = torch.cat([x1, y2], dim=1)
            # log det jacobianの計算
            ldj = torch.sum(torch.log(scale), dim=1)
        # 逆方向変換
        else:
            # 逆アフィン変換: x2 = (y2 - shift) / scale
            y2 = (x2 - shift) / scale
            y = torch.cat([x1, y2], dim=1)
            # log det jacobianの計算
            ldj = -torch.sum(torch.log(scale), dim=1)
        
        return y, torch.sum(ldj)


class FlowBlock(nn.Module):
    """
    Normalizing Flowの基本ブロック
    
    ActNorm, 1x1畳み込み、アフィン結合を順に適用
    """
    def __init__(self, num_features, hidden_features=128):
        """
        Args:
            num_features: 特徴量の次元数
            hidden_features: 隠れ層の次元数
        """
        super(FlowBlock, self).__init__()
        
        self.actnorm = ActNorm(num_features)
        self.conv = InvertibleConv1x1(num_features)
        self.coupling = AffineCoupling(num_features, hidden_features)
    
    def forward(self, x, reverse=False):
        """
        順方向および逆方向変換
        
        Args:
            x: 入力データ [B, D]
            reverse: 逆方向変換を行うかどうか
            
        Returns:
            y: 変換された出力
            ldj: log det jacobian
        """
        total_ldj = 0.0
        
        # 順方向
        if not reverse:
            # ActNorm
            x, ldj1 = self.actnorm(x, reverse=False)
            # 1x1畳み込み
            x, ldj2 = self.conv(x, reverse=False)
            # アフィン結合
            x, ldj3 = self.coupling(x, reverse=False)
        # 逆方向
        else:
            # 逆順に適用
            # アフィン結合
            x, ldj3 = self.coupling(x, reverse=True)
            # 1x1畳み込み
            x, ldj2 = self.conv(x, reverse=True)
            # ActNorm
            x, ldj1 = self.actnorm(x, reverse=True)
        
        # 全体のlog det jacobianを計算
        total_ldj = ldj1 + ldj2 + ldj3
        
        return x, total_ldj


class NormalizingFlow(nn.Module):
    """
    複数のFlowBlockを積み重ねたNormalizing Flowモデル
    """
    def __init__(self, num_features, hidden_features=128, num_blocks=6):
        """
        Args:
            num_features: 特徴量の次元数（入力の次元数）
            hidden_features: 隠れ層の次元数
            num_blocks: フローブロックの数
        """
        super(NormalizingFlow, self).__init__()
        
        self.num_features = num_features
        self.hidden_features = hidden_features
        self.num_blocks = num_blocks
        
        # フローブロックのリスト
        self.blocks = nn.ModuleList([
            FlowBlock(num_features, hidden_features) for _ in range(num_blocks)
        ])
        
        # 事前分布
        self.register_buffer("prior_mean", torch.zeros(1, num_features))
        self.register_buffer("prior_std", torch.ones(1, num_features))
    
    def forward(self, x, reverse=False):
        """
        順方向および逆方向変換
        
        Args:
            x: 入力データ [B, D]（逆方向の場合は潜在変数）
            reverse: 逆方向変換を行うかどうか
            
        Returns:
            y: 変換された出力
            ldj: log det jacobian
        """
        total_ldj = 0.0
        
        # 順方向
        if not reverse:
            # 各ブロックを順に適用
            for block in self.blocks:
                x, ldj = block(x, reverse=False)
                total_ldj += ldj
        # 逆方向
        else:
            # 各ブロックを逆順に適用
            for block in reversed(self.blocks):
                x, ldj = block(x, reverse=True)
                total_ldj += ldj
        
        return x, total_ldj
    
    def log_prob(self, x):
        """
        データの対数確率を計算
        
        Args:
            x: 入力データ [B, D]
            
        Returns:
            log_prob: 対数確率 [B]
        """
        # データを潜在空間に変換
        z, ldj = self.forward(x, reverse=False)
        
        # 正規分布の対数確率を計算
        log_prob_z = -0.5 * torch.sum(((z - self.prior_mean) / self.prior_std) ** 2, dim=1)
        log_prob_z = log_prob_z - 0.5 * self.num_features * np.log(2 * np.pi)
        log_prob_z = log_prob_z - torch.sum(torch.log(self.prior_std), dim=1)
        
        # 変換のヤコビアンの対数行列式を加える
        log_prob = log_prob_z + ldj
        
        return log_prob
    
    def sample(self, num_samples=1, device=None):
        """
        モデルからサンプルを生成
        
        Args:
            num_samples: 生成するサンプル数
            device: デバイス
            
        Returns:
            samples: 生成されたサンプル [num_samples, D]
        """
        # 事前分布からサンプリング
        z = self.prior_std * torch.randn(num_samples, self.num_features, device=device) + self.prior_mean
        
        # 逆変換を適用
        samples, _ = self.forward(z, reverse=True)
        
        return samples