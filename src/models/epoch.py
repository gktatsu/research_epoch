"""
EPOCHフレームワークの実装

RegNetとLiftNetを組み合わせたエンドツーエンドのフレームワーク
単一の画像から直接3Dポーズとカメラパラメータを推定
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..config import MODEL_CONFIG
from .regnet import RegNet
from .liftnet import LiftNet


class EPOCH(nn.Module):
    """
    EPOCHフレームワーク
    
    RegNetとLiftNetを組み合わせて、単一の画像から3Dポーズとカメラパラメータを推定
    """
    def __init__(self, num_joints=17, encoder_name='resnet50', pretrained=True, image_size=(224, 224)):
        """
        Args:
            num_joints: 関節の数
            encoder_name: エンコーダのアーキテクチャ名
            pretrained: 事前訓練済みの重みを使用するかどうか
            image_size: 入力画像のサイズ (幅, 高さ)
        """
        super(EPOCH, self).__init__()
        
        self.num_joints = num_joints
        self.image_size = image_size
        
        # RegNet: 画像から2Dポーズとカメラパラメータを推定
        self.regnet = RegNet(
            num_joints=num_joints,
            encoder_name=encoder_name,
            pretrained=pretrained,
            image_size=image_size
        )
        
        # LiftNet: 2Dポーズとカメラパラメータからより正確な3Dポーズを推定
        self.liftnet = LiftNet(
            num_joints=num_joints,
            feat_dim=MODEL_CONFIG['lift_net']['dim_l'],
            num_residual_blocks=MODEL_CONFIG['lift_net']['residual_blocks']
        )
    
    def forward(self, image, bbox=None):
        """
        画像から3Dポーズとカメラパラメータを推定
        
        Args:
            image: 入力画像 [バッチサイズ, チャネル数, 高さ, 幅]
            bbox: バウンディングボックス [バッチサイズ, 4] (左, 上, 幅, 高さ)
            
        Returns:
            dict: 推定された2Dポーズ、3Dポーズ、カメラパラメータ、中間結果
        """
        # RegNetで2Dポーズとカメラパラメータを推定
        regnet_outputs = self.regnet(image, bbox)
        
        pose_2d = regnet_outputs['pose_2d']           # [B, num_joints, 2]
        pose_3d_reg = regnet_outputs['pose_3d']       # [B, num_joints, 3]
        joint_presence = regnet_outputs['joint_presence']  # [B, num_joints]
        cam_K = regnet_outputs['cam_K']               # [B, 3, 3]
        cam_R = regnet_outputs['cam_R']               # [B, 3, 3]
        cam_t = regnet_outputs['cam_t']               # [B, 3]
        
        # LiftNetでより正確な3Dポーズを推定
        liftnet_outputs = self.liftnet(pose_2d, cam_K, cam_R, cam_t)
        
        pose_3d_lift = liftnet_outputs['pose_3d']     # [B, num_joints, 3]
        
        return {
            'pose_2d': pose_2d,                       # RegNetで推定された2Dポーズ
            'pose_3d_reg': pose_3d_reg,               # RegNetで推定された3Dポーズ
            'pose_3d_lift': pose_3d_lift,             # LiftNetで推定された3Dポーズ
            'joint_presence': joint_presence,         # 関節存在確率
            'cam_K': cam_K,                           # カメラ内部パラメータ
            'cam_R': cam_R,                           # カメラ回転行列
            'cam_t': cam_t,                           # カメラ平行移動ベクトル
            'regnet_outputs': regnet_outputs,         # RegNetの全出力
            'liftnet_outputs': liftnet_outputs        # LiftNetの全出力
        }
    
    def estimate_pose(self, image, bbox=None):
        """
        画像から3Dポーズとカメラパラメータを推定（推論時用の簡易バージョン）
        
        Args:
            image: 入力画像 [バッチサイズ, チャネル数, 高さ, 幅]
            bbox: バウンディングボックス [バッチサイズ, 4] (左, 上, 幅, 高さ)
            
        Returns:
            pose_2d: 推定された2Dポーズ [バッチサイズ, 関節数, 2]
            pose_3d: 推定された3Dポーズ [バッチサイズ, 関節数, 3]
            cam_K: カメラ内部パラメータ [バッチサイズ, 3, 3]
            cam_R: カメラ回転行列 [バッチサイズ, 3, 3]
            cam_t: カメラ平行移動ベクトル [バッチサイズ, 3]
        """
        # モデルの出力を取得
        outputs = self.forward(image, bbox)
        
        return (
            outputs['pose_2d'],
            outputs['pose_3d_lift'],  # LiftNetの出力を最終結果として使用
            outputs['cam_K'],
            outputs['cam_R'],
            outputs['cam_t']
        )
    
    def load_regnet_weights(self, weights_path):
        """
        RegNetの事前学習済み重みを読み込む
        
        Args:
            weights_path: 重みファイルのパス
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        self.regnet.load_state_dict(checkpoint['model_state_dict'])
        print(f"RegNetの重みを読み込みました: {weights_path}")
    
    def load_liftnet_weights(self, weights_path):
        """
        LiftNetの事前学習済み重みを読み込む
        
        Args:
            weights_path: 重みファイルのパス
        """
        checkpoint = torch.load(weights_path, map_location='cpu')
        self.liftnet.load_state_dict(checkpoint['model_state_dict'])
        print(f"LiftNetの重みを読み込みました: {weights_path}")