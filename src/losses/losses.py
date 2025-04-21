"""
EPOCHフレームワークに必要な損失関数の実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..utils.pose_utils import JOINT_CONNECTIONS, BONE_PAIRS, CONSTRAINED_LIMBS


class L2DLoss(nn.Module):
    """
    2Dポーズの一貫性損失
    
    入力された2Dポーズと再投影された2Dポーズ間のL1距離を計算
    """
    def __init__(self):
        super(L2DLoss, self).__init__()
    
    def forward(self, pred_2d, target_2d):
        """
        Args:
            pred_2d: 予測された2Dポーズ [バッチサイズ, 関節数, 2]
            target_2d: 目標の2Dポーズ [バッチサイズ, 関節数, 2]
            
        Returns:
            loss: L1損失
        """
        return torch.mean(torch.abs(pred_2d - target_2d))


class L3DLoss(nn.Module):
    """
    3Dポーズの一貫性損失
    
    入力された3Dポーズと再構築された3Dポーズ間のL2距離を計算
    """
    def __init__(self):
        super(L3DLoss, self).__init__()
    
    def forward(self, pred_3d, target_3d):
        """
        Args:
            pred_3d: 予測された3Dポーズ [バッチサイズ, 関節数, 3]
            target_3d: 目標の3Dポーズ [バッチサイズ, 関節数, 3]
            
        Returns:
            loss: L2損失
        """
        return torch.mean(torch.sqrt(torch.sum((pred_3d - target_3d) ** 2, dim=-1)))


class BoneLoss(nn.Module):
    """
    骨の長さの比率に関する損失
    
    人体の骨の長さの比率が一定であるという解剖学的制約を適用
    """
    def __init__(self):
        super(BoneLoss, self).__init__()
    
    def forward(self, pose_3d):
        """
        Args:
            pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3]
            
        Returns:
            loss: 骨の長さの比率のL1損失
        """
        batch_size = pose_3d.shape[0]
        device = pose_3d.device
        
        # データの異常値をチェック
        if torch.isnan(pose_3d).any() or torch.isinf(pose_3d).any():
            return torch.tensor(0.1, device=device)  # エラーがある場合は小さな損失を返す
        
        try:
            loss = 0.0
            
            # 骨のペアごとに比率を計算
            for (joint1a, joint1b), (joint2a, joint2b) in BONE_PAIRS:
                # 骨ベクトルの計算
                bone1 = pose_3d[:, joint1b] - pose_3d[:, joint1a]  # [B, 3]
                bone2 = pose_3d[:, joint2b] - pose_3d[:, joint2a]  # [B, 3]
                
                # 骨の長さの計算
                length1 = torch.norm(bone1, dim=1)  # [B]
                length2 = torch.norm(bone2, dim=1)  # [B]
                
                # ゼロ長さのボーンがないか確認（異常値）
                valid_lengths = (length1 > 1e-6) & (length2 > 1e-6)
                if not valid_lengths.all():
                    # 一部の骨の長さが無効なサンプルは除外
                    valid_indices = torch.where(valid_lengths)[0]
                    if len(valid_indices) == 0:
                        continue
                    length1 = length1[valid_indices]
                    length2 = length2[valid_indices]
                
                # 左右の比率を1に近づける（左右対称性）
                ratio = length1 / (length2 + 1e-8)
                target_ratio = torch.ones_like(ratio)
                
                # 異常な比率を排除
                valid_ratios = (ratio < 10.0) & (ratio > 0.1)
                if not valid_ratios.all():
                    valid_indices = torch.where(valid_ratios)[0]
                    if len(valid_indices) == 0:
                        continue
                    ratio = ratio[valid_indices]
                    target_ratio = target_ratio[valid_indices]
                
                # 損失を追加
                loss += torch.mean(torch.abs(ratio - target_ratio))
            
            return loss / len(BONE_PAIRS)
        except Exception as e:
            print(f"BoneLoss計算中にエラーが発生しました: {e}")
            return torch.tensor(0.1, device=device)  # エラー時は小さな損失を返す


class LimbsLoss(nn.Module):
    """
    肢の自然な曲げを保証する損失
    
    膝と肘が不自然な方向に曲がらないように制約を適用
    """
    def __init__(self):
        super(LimbsLoss, self).__init__()
    
    def forward(self, pose_3d):
        """
        Args:
            pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3]
            
        Returns:
            loss: 不自然な曲げに対するペナルティ
        """
        batch_size = pose_3d.shape[0]
        device = pose_3d.device
        loss = torch.tensor(0.0, device=device)
        
        # データの異常値をチェック
        if torch.isnan(pose_3d).any() or torch.isinf(pose_3d).any():
            return loss  # エラーがある場合は損失を0として返す
        
        try:
            # 各ポーズについて法線ベクトルを計算
            # 骨盤と両側の股関節の位置から平面の法線を計算
            hip_left = pose_3d[:, 4] - pose_3d[:, 0]   # 左股関節 - 骨盤
            hip_right = pose_3d[:, 7] - pose_3d[:, 0]  # 右股関節 - 骨盤
            
            # 外積で法線ベクトルを計算
            normal = torch.cross(hip_left, hip_right, dim=1)  # [B, 3]
            normal_norm = torch.norm(normal, dim=1, keepdim=True)
            
            # ゼロベクトルを避ける
            valid_normals = normal_norm > 1e-6
            if not valid_normals.all():
                # 一部の法線ベクトルが無効な場合、代替の法線ベクトルを使用
                normal[~valid_normals.squeeze()] = torch.tensor([0.0, 1.0, 0.0], device=device)
                normal_norm[~valid_normals] = 1.0
            
            normal = normal / (normal_norm + 1e-8)  # 正規化
            
            # 各制約付き肢について計算
            for proximal, distal in CONSTRAINED_LIMBS:
                # 近位部と遠位部のベクトル
                proximal_vec = pose_3d[:, proximal] - pose_3d[:, 0]  # 近位関節 - 骨盤
                distal_vec = pose_3d[:, distal] - pose_3d[:, proximal]  # 遠位関節 - 近位関節
                
                # 法線との内積
                proximal_proj = torch.sum(normal * proximal_vec, dim=1)  # [B]
                distal_proj = torch.sum(normal * distal_vec, dim=1)  # [B]
                
                # 不自然な曲げに対するペナルティ（ReLUで正の差のみを取得）
                diff = proximal_proj - distal_proj
                penalty = torch.maximum(torch.zeros_like(diff), diff)
                
                # 損失を追加
                loss += torch.mean(penalty)
        except Exception as e:
            print(f"LimbsLoss計算中にエラーが発生しました: {e}")
            return torch.tensor(0.1, device=device)  # エラー時は小さな損失を返す
        
        return loss / len(CONSTRAINED_LIMBS)


class DeformationLoss(nn.Module):
    """
    姿勢の変形の一貫性を保証する損失
    
    同じバッチ内の2つのポーズ間の距離が、投影と再構築の後も保存されるように制約
    """
    def __init__(self):
        super(DeformationLoss, self).__init__()
    
    def forward(self, pred_3d, reconstructed_3d):
        """
        Args:
            pred_3d: 予測された3Dポーズ [バッチサイズ, 関節数, 3]
            reconstructed_3d: 再構築された3Dポーズ [バッチサイズ, 関節数, 3]
            
        Returns:
            loss: バッチ内のペア間の変形の差のL2損失
        """
        batch_size = pred_3d.shape[0]
        if batch_size <= 1:
            return torch.tensor(0.0, device=pred_3d.device)
        
        loss = 0.0
        
        # バッチ内の全てのペアについて
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                # 元のポーズの差
                orig_diff = pred_3d[i] - pred_3d[j]  # [J, 3]
                
                # 再構築されたポーズの差
                recon_diff = reconstructed_3d[i] - reconstructed_3d[j]  # [J, 3]
                
                # 変形の差のL2ノルム
                deform_diff = torch.norm(orig_diff - recon_diff, dim=-1)  # [J]
                
                # 損失を追加
                loss += torch.mean(deform_diff)
        
        # ペアの数で正規化
        num_pairs = (batch_size * (batch_size - 1)) // 2
        return loss / num_pairs


class NFLoss(nn.Module):
    """
    Normalizing Flowを用いた2Dポーズの尤度損失
    
    生成された2Dポーズが実際の2Dポーズの分布に従うように制約
    """
    def __init__(self, normalizing_flow):
        """
        Args:
            normalizing_flow: 学習済みのNormalizing Flowモデル
        """
        super(NFLoss, self).__init__()
        self.normalizing_flow = normalizing_flow
    
    def forward(self, pose_2d):
        """
        Args:
            pose_2d: 2Dポーズ [バッチサイズ, 関節数, 2]
            
        Returns:
            loss: 負の対数尤度
        """
        # 入力を平坦化（NFモデルによって必要な形状が異なる場合がある）
        batch_size = pose_2d.shape[0]
        flattened_pose = pose_2d.reshape(batch_size, -1)  # [B, J*2]
        
        # 負の対数尤度を計算
        neg_log_likelihood = -self.normalizing_flow.log_prob(flattened_pose)
        
        return torch.mean(neg_log_likelihood)


class ResidualLogLikelihoodLoss(nn.Module):
    """
    残差対数尤度推定（RLE）損失
    
    予測の不確かさを考慮した2Dポーズ推定の損失
    """
    def __init__(self):
        super(ResidualLogLikelihoodLoss, self).__init__()
    
    def forward(self, pred_2d, target_2d, sigma):
        """
        Args:
            pred_2d: 予測された2Dポーズ [バッチサイズ, 関節数, 2]
            target_2d: 目標の2Dポーズ [バッチサイズ, 関節数, 2]
            sigma: 予測の標準偏差（不確かさ） [バッチサイズ, 関節数]
            
        Returns:
            loss: 負の対数尤度と正規化項の和
        """
        # 残差の計算
        residual = torch.norm(pred_2d - target_2d, dim=-1)  # [B, J]
        
        # 負の対数尤度
        neg_log_likelihood = torch.log(sigma) + residual / sigma
        
        return torch.mean(neg_log_likelihood)