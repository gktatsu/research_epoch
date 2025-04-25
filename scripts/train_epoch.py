"""
EPOCHフレームワーク統合訓練スクリプト

RegNetとLiftNetを組み合わせたエンドツーエンドのモデルを訓練
"""

"""実行例：
ssh aryabhata
conda activate epoch
cd scripts
python train_epoch.py --dataset mpiinf3dhp --data_path /home2/t-hori/2025/develop/epoch/data/MPI-INF-3DHP/mpi_inf_3dhp_training_set_h5
"""
import os
import sys
import argparse
import time
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import MODELS_DIR, DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG
from src.models import EPOCH, NormalizingFlow
from src.data import get_dataloader
from src.losses import L2DLoss, L3DLoss, BoneLoss, LimbsLoss, DeformationLoss, NFLoss, ResidualLogLikelihoodLoss
from src.utils.metrics import calculate_mpjpe, calculate_pa_mpjpe, calculate_n_mpjpe


def train_epoch(model, normalizing_flow, train_loader, optimizer, criterion, device, loss_weights):
    """
    1エポックの訓練を実行
    
    Args:
        model: 訓練するモデル
        normalizing_flow: 学習済みのNormalizing Flow
        train_loader: 訓練データローダー
        optimizer: オプティマイザ
        criterion: 損失関数の辞書
        device: デバイス
        loss_weights: 損失関数の重み
        
    Returns:
        losses: 各損失の平均値
        metrics: 評価指標の平均値
    """
    model.train()
    
    # 損失の合計
    total_losses = {
        'total': 0,
        'rle': 0,
        'bone': 0,
        'limbs': 0,
        'nf': 0,
        'l2d': 0,
        'l3d': 0,
        'def': 0
    }
    
    # 評価指標の合計
    total_metrics = {
        'mpjpe_2d': 0,
        'mpjpe_3d': 0,
        'pa_mpjpe_3d': 0,
        'n_mpjpe_3d': 0
    }
    
    # バッチ数
    num_batches = len(train_loader)
    
    # 進捗バー
    with tqdm(train_loader, desc='Train', leave=False) as t:
        for batch_idx, batch in enumerate(t):
            # データをデバイスに転送
            image = batch['image'].to(device)
            pose_2d_gt = batch['pose_2d'].to(device)
            pose_3d_gt = batch['pose_3d'].to(device)
            
            # 入力画像のバウンディングボックスは、常に画像全体を使用
            batch_size = image.shape[0]
            h, w = image.shape[2], image.shape[3]
            bbox = torch.tensor([[0, 0, w, h]], device=device).repeat(batch_size, 1).float()
            
            # 勾配をゼロに初期化
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(image, bbox)
            
            # RegNetの出力
            regnet_outputs = outputs['regnet_outputs']
            pose_2d_pred = regnet_outputs['pose_2d']
            pose_3d_reg = regnet_outputs['pose_3d']
            joint_presence = regnet_outputs['joint_presence']
            
            # LiftNetの出力
            liftnet_outputs = outputs['liftnet_outputs']
            pose_3d_lift = liftnet_outputs['pose_3d']
            pose_2d_rec = liftnet_outputs['pose_2d_rec']
            pose_3d_rec = liftnet_outputs['pose_3d_rec']
            
            # 回転されたポーズと2D投影
            if 'pose_2d_rotated' in regnet_outputs:
                rotated_pose_2d_reg = regnet_outputs['pose_2d_rotated']
            else:
                rotated_pose_2d_reg = None
                
            rotated_pose_2d_lift = liftnet_outputs['pose_2d_rotated']
            
            # 関節存在確率をシグモイド関数で0~1に変換
            joint_presence = torch.sigmoid(joint_presence)
            
            # RegNetの損失をinit
            loss_reg_rle = torch.tensor(0.0, device=device)
            loss_reg_bone = torch.tensor(0.0, device=device)
            loss_reg_limbs = torch.tensor(0.0, device=device)
            
            # RegNet損失を計算
            loss_reg_rle = criterion['rle'](pose_2d_pred, pose_2d_gt, joint_presence)
            loss_reg_bone = criterion['bone'](pose_3d_reg)
            loss_reg_limbs = criterion['limbs'](pose_3d_reg)
            
            # RegNet Normalizing Flow損失（回転されたポーズがある場合）
            loss_reg_nf = torch.tensor(0.0, device=device)
            if rotated_pose_2d_reg is not None and criterion['nf'] is not None:
                loss_reg_nf = criterion['nf'](rotated_pose_2d_reg)
            
            # LiftNetの損失をinit
            loss_lift_l2d = torch.tensor(0.0, device=device)
            loss_lift_l3d = torch.tensor(0.0, device=device)
            loss_lift_bone = torch.tensor(0.0, device=device)
            loss_lift_limbs = torch.tensor(0.0, device=device)
            loss_lift_def = torch.tensor(0.0, device=device)

            # LiftNet損失を計算
            loss_lift_l2d = criterion['l2d'](pose_2d_rec, pose_2d_gt)
            loss_lift_l3d = criterion['l3d'](pose_3d_rec, pose_3d_lift)
            loss_lift_bone = criterion['bone'](pose_3d_lift)
            loss_lift_limbs = criterion['limbs'](pose_3d_lift)
            loss_lift_def = criterion['def'](pose_3d_lift, pose_3d_rec)
            
            # LiftNet Normalizing Flow損失
            loss_lift_nf = torch.tensor(0.0, device=device)
            if criterion['nf'] is not None:
                loss_lift_nf = criterion['nf'](rotated_pose_2d_lift)
            
            # 大きな損失値に対するクリッピング (テンソルのまま保持)
            max_loss_value = torch.tensor(100.0, device=device)
            # loss_reg_rle = torch.minimum(loss_reg_rle, max_loss_value)
            # loss_reg_bone = torch.minimum(loss_reg_bone, max_loss_value)
            # loss_reg_limbs = torch.minimum(loss_reg_limbs, max_loss_value)
            # loss_lift_l2d = torch.minimum(loss_lift_l2d, max_loss_value)
            # loss_lift_l3d = torch.minimum(loss_lift_l3d, max_loss_value)
            # loss_lift_bone = torch.minimum(loss_lift_bone, max_loss_value)
            # loss_lift_limbs = torch.minimum(loss_lift_limbs, max_loss_value)
            # loss_lift_def = torch.minimum(loss_lift_def, max_loss_value)

            # 各損失が確実にテンソルであることを確認
            loss_reg_rle = torch.minimum(loss_reg_rle if isinstance(loss_reg_rle, torch.Tensor) 
                                        else torch.tensor(loss_reg_rle, device=device), max_loss_value)
            loss_reg_bone = torch.minimum(loss_reg_bone if isinstance(loss_reg_bone, torch.Tensor) 
                                        else torch.tensor(loss_reg_bone, device=device), max_loss_value)
            loss_reg_limbs = torch.minimum(loss_reg_limbs if isinstance(loss_reg_limbs, torch.Tensor) 
                                        else torch.tensor(loss_reg_limbs, device=device), max_loss_value)
            loss_lift_l2d = torch.minimum(loss_lift_l2d if isinstance(loss_lift_l2d, torch.Tensor) 
                                        else torch.tensor(loss_lift_l2d, device=device), max_loss_value)
            loss_lift_l3d = torch.minimum(loss_lift_l3d if isinstance(loss_lift_l3d, torch.Tensor) 
                                        else torch.tensor(loss_lift_l3d, device=device), max_loss_value)
            loss_lift_bone = torch.minimum(loss_lift_bone if isinstance(loss_lift_bone, torch.Tensor) 
                                        else torch.tensor(loss_lift_bone, device=device), max_loss_value)
            loss_lift_limbs = torch.minimum(loss_lift_limbs if isinstance(loss_lift_limbs, torch.Tensor) 
                                        else torch.tensor(loss_lift_limbs, device=device), max_loss_value)
            loss_lift_def = torch.minimum(loss_lift_def if isinstance(loss_lift_def, torch.Tensor) 
                                        else torch.tensor(loss_lift_def, device=device), max_loss_value)
            
            # 重み付き損失の合計
            loss = (
                loss_weights['rle'] * loss_reg_rle +
                loss_weights['bone'] * (loss_lift_bone + loss_reg_bone) +
                loss_weights['limbs'] * (loss_lift_limbs + loss_reg_limbs) +
                loss_weights['l2d'] * loss_lift_l2d +
                loss_weights['l3d'] * loss_lift_l3d +
                loss_weights['def'] * loss_lift_def
            )

            
            # Normalizing Flow損失（利用可能な場合）
            if criterion['nf'] is not None:
                loss += loss_weights['nf'] * (loss_lift_nf + loss_reg_nf)
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()
            
            # 損失を累積
            total_losses['total'] += loss.item()
            total_losses['rle'] += loss_reg_rle.item()
            total_losses['bone'] += (loss_lift_bone.item() + loss_reg_bone.item()) / 2
            total_losses['limbs'] += (loss_lift_limbs.item() + loss_reg_limbs.item()) / 2
            total_losses['l2d'] += loss_lift_l2d.item()
            total_losses['l3d'] += loss_lift_l3d.item()
            total_losses['def'] += loss_lift_def.item()
            
            if criterion['nf'] is not None:
                total_losses['nf'] += (loss_lift_nf.item() + loss_reg_nf.item()) / 2
            
            # 評価指標を計算
            # 2Dポーズの平均関節位置誤差（ピクセル単位）
            mpjpe_2d = torch.mean(torch.sqrt(torch.sum((pose_2d_pred - pose_2d_gt) ** 2, dim=-1)))
            
            # 3Dポーズの平均関節位置誤差（mm単位）
            # CPUに転送して計算（NumPy関数を使用するため）
            pose_3d_pred_np = pose_3d_lift.detach().cpu().numpy()
            pose_3d_gt_np = pose_3d_gt.detach().cpu().numpy()
            
            mpjpe_3d = calculate_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            pa_mpjpe_3d = calculate_pa_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            n_mpjpe_3d = calculate_n_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            
            # 評価指標を累積
            total_metrics['mpjpe_2d'] += mpjpe_2d.item()
            total_metrics['mpjpe_3d'] += mpjpe_3d
            total_metrics['pa_mpjpe_3d'] += pa_mpjpe_3d
            total_metrics['n_mpjpe_3d'] += n_mpjpe_3d
            
            # 進捗バーに現在の損失を表示
            t.set_postfix(loss=loss.item())
    
    # 各損失と評価指標の平均を計算
    for key in total_losses:
        total_losses[key] /= num_batches
    
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_losses, total_metrics


def validate(model, normalizing_flow, val_loader, criterion, device, loss_weights):
    """
    検証を実行
    
    Args:
        model: 評価するモデル（EPOCHフレームワーク）
        normalizing_flow: 学習済みのNormalizing Flow
        val_loader: 検証データローダー
        criterion: 損失関数の辞書
        device: デバイス
        loss_weights: 損失関数の重み
        
    Returns:
        losses: 各損失の平均値
        metrics: 評価指標の平均値
    """
    model.eval()
    
    # 損失の合計
    total_losses = {
        'total': 0,
        'rle': 0,
        'bone': 0,
        'limbs': 0,
        'nf': 0,
        'l2d': 0,
        'l3d': 0,
        'def': 0
    }
    
    # 評価指標の合計
    total_metrics = {
        'mpjpe_2d': 0,
        'mpjpe_3d': 0,
        'pa_mpjpe_3d': 0,
        'n_mpjpe_3d': 0
    }
    
    # バッチ数
    num_batches = len(val_loader)
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 進捗バー
        with tqdm(val_loader, desc='Validate', leave=False) as t:
            for batch_idx, batch in enumerate(t):
                # データをデバイスに転送
                image = batch['image'].to(device)
                pose_2d_gt = batch['pose_2d'].to(device)
                pose_3d_gt = batch['pose_3d'].to(device)
                
                # 入力画像のバウンディングボックスは、常に画像全体を使用
                batch_size = image.shape[0]
                h, w = image.shape[2], image.shape[3]
                bbox = torch.tensor([[0, 0, w, h]], device=device).repeat(batch_size, 1).float()
                
                # 順伝播
                outputs = model(image, bbox)
                
                # RegNetの出力
                regnet_outputs = outputs['regnet_outputs']
                pose_2d_reg = regnet_outputs['pose_2d']         # RegNetで推定された2Dポーズ
                pose_3d_reg = regnet_outputs['pose_3d']         # RegNetで推定された3Dポーズ
                joint_presence = regnet_outputs['joint_presence'] # 関節存在確率
                rotated_pose_2d_reg = regnet_outputs.get('pose_2d_rotated', None)  # 回転された2Dポーズ（RegNet）
                
                # LiftNetの出力
                liftnet_outputs = outputs['liftnet_outputs']
                pose_3d_lift = outputs['pose_3d_lift']           # LiftNetで推定された3Dポーズ
                pose_2d_rec = liftnet_outputs.get('pose_2d_rec', None)  # 再投影された2Dポーズ
                pose_3d_rec = liftnet_outputs.get('pose_3d_rec', None)  # 再構築された3Dポーズ
                rotated_pose_2d_lift = liftnet_outputs.get('pose_2d_rotated', None)  # 回転された2Dポーズ（LiftNet）
                
                # 関節存在確率をシグモイド関数で0~1に変換
                joint_presence = torch.sigmoid(joint_presence)
                
                # RegNetの損失を計算
                loss_reg_rle = criterion['rle'](pose_2d_reg, pose_2d_gt, joint_presence)
                loss_reg_bone = criterion['bone'](pose_3d_reg)
                loss_reg_limbs = criterion['limbs'](pose_3d_reg)
                
                # Normalizing Flow損失（RegNet、回転されたポーズがある場合）
                loss_nf_reg = torch.tensor(0.0, device=device)
                if rotated_pose_2d_reg is not None and criterion['nf'] is not None:
                    loss_nf_reg = criterion['nf'](rotated_pose_2d_reg)
                
                # LiftNetの損失を計算
                loss_lift_l2d = torch.tensor(0.0, device=device)
                loss_lift_l3d = torch.tensor(0.0, device=device)
                loss_lift_def = torch.tensor(0.0, device=device)
                
                # サイクル一貫性が有効な場合（pose_2d_recとpose_3d_recが存在する場合）
                if pose_2d_rec is not None and pose_3d_rec is not None:
                    loss_lift_l2d = criterion['l2d'](pose_2d_rec, pose_2d_gt)
                    loss_lift_l3d = criterion['l3d'](pose_3d_rec, pose_3d_lift)
                    loss_lift_def = criterion['def'](pose_3d_lift, pose_3d_rec)
                
                loss_lift_bone = criterion['bone'](pose_3d_lift)
                loss_lift_limbs = criterion['limbs'](pose_3d_lift)
                
                # Normalizing Flow損失（LiftNet、回転されたポーズがある場合）
                loss_nf_lift = torch.tensor(0.0, device=device)
                if rotated_pose_2d_lift is not None and criterion['nf'] is not None:
                    loss_nf_lift = criterion['nf'](rotated_pose_2d_lift)
                    
                # 大きな損失値に対するクリッピング (テンソルのまま保持)
                max_loss_value = torch.tensor(100.0, device=device)
                # loss_reg_rle = torch.minimum(loss_reg_rle, max_loss_value)
                # loss_reg_bone = torch.minimum(loss_reg_bone, max_loss_value)
                # loss_reg_limbs = torch.minimum(loss_reg_limbs, max_loss_value)
                # loss_lift_l2d = torch.minimum(loss_lift_l2d, max_loss_value)
                # loss_lift_l3d = torch.minimum(loss_lift_l3d, max_loss_value)
                # loss_lift_bone = torch.minimum(loss_lift_bone, max_loss_value)
                # loss_lift_limbs = torch.minimum(loss_lift_limbs, max_loss_value)
                # loss_lift_def = torch.minimum(loss_lift_def, max_loss_value)
                
                 # 各損失が確実にテンソルであることを確認
                loss_reg_rle = torch.minimum(loss_reg_rle if isinstance(loss_reg_rle, torch.Tensor) 
                                            else torch.tensor(loss_reg_rle, device=device), max_loss_value)
                loss_reg_bone = torch.minimum(loss_reg_bone if isinstance(loss_reg_bone, torch.Tensor) 
                                            else torch.tensor(loss_reg_bone, device=device), max_loss_value)
                loss_reg_limbs = torch.minimum(loss_reg_limbs if isinstance(loss_reg_limbs, torch.Tensor) 
                                            else torch.tensor(loss_reg_limbs, device=device), max_loss_value)
                loss_lift_l2d = torch.minimum(loss_lift_l2d if isinstance(loss_lift_l2d, torch.Tensor) 
                                            else torch.tensor(loss_lift_l2d, device=device), max_loss_value)
                loss_lift_l3d = torch.minimum(loss_lift_l3d if isinstance(loss_lift_l3d, torch.Tensor) 
                                            else torch.tensor(loss_lift_l3d, device=device), max_loss_value)
                loss_lift_bone = torch.minimum(loss_lift_bone if isinstance(loss_lift_bone, torch.Tensor) 
                                            else torch.tensor(loss_lift_bone, device=device), max_loss_value)
                loss_lift_limbs = torch.minimum(loss_lift_limbs if isinstance(loss_lift_limbs, torch.Tensor) 
                                            else torch.tensor(loss_lift_limbs, device=device), max_loss_value)
                loss_lift_def = torch.minimum(loss_lift_def if isinstance(loss_lift_def, torch.Tensor) 
                                            else torch.tensor(loss_lift_def, device=device), max_loss_value)
                
                # 重み付き損失の合計
                loss = (
                    loss_weights['rle'] * loss_reg_rle +
                    loss_weights['l2d'] * loss_lift_l2d +
                    loss_weights['l3d'] * loss_lift_l3d +
                    loss_weights['bone'] * ((loss_reg_bone + loss_lift_bone) / 2) +
                    loss_weights['limbs'] * ((loss_reg_limbs + loss_lift_limbs) / 2) +
                    loss_weights['def'] * loss_lift_def
                )
                
                # Normalizing Flow損失（利用可能な場合）
                if rotated_pose_2d_reg is not None and criterion['nf'] is not None:
                    loss += loss_weights['nf'] * loss_nf_reg
                    
                if rotated_pose_2d_lift is not None and criterion['nf'] is not None:
                    loss += loss_weights['nf'] * loss_nf_lift
                    
                
                # 損失を累積
                total_losses['total'] += loss.item()
                total_losses['rle'] += loss_reg_rle.item()
                total_losses['bone'] += (loss_lift_bone.item() + loss_reg_bone.item()) / 2
                total_losses['limbs'] += (loss_lift_limbs.item() + loss_reg_limbs.item()) / 2
                total_losses['l2d'] += loss_lift_l2d.item()
                total_losses['l3d'] += loss_lift_l3d.item()
                total_losses['def'] += loss_lift_def.item()
                
                if rotated_pose_2d_reg is not None and criterion['nf'] is not None:
                    total_losses['nf_reg'] += loss_nf_reg.item() / 2
                if rotated_pose_2d_lift is not None and criterion['nf'] is not None:
                    total_losses['nf_lift'] += loss_nf_lift.item() / 2

                # 評価指標を計算
                # 2Dポーズの平均関節位置誤差（ピクセル単位）
                mpjpe_2d = torch.mean(torch.sqrt(torch.sum((pose_2d_reg - pose_2d_gt) ** 2, dim=-1)))
                
                # 3Dポーズの平均関節位置誤差（mm単位）
                # CPUに転送して計算（NumPy関数を使用するため）
                pose_3d_pred_np = pose_3d_lift.detach().cpu().numpy()
                pose_3d_gt_np = pose_3d_gt.detach().cpu().numpy()
                
                mpjpe_3d = calculate_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
                pa_mpjpe_3d = calculate_pa_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
                n_mpjpe_3d = calculate_n_mpjpe(pose_3d_pred_np, pose_3d_gt_np)

                # 評価指標を累積
                total_metrics['mpjpe_2d'] += mpjpe_2d.item()
                total_metrics['mpjpe_3d'] += mpjpe_3d
                total_metrics['pa_mpjpe_3d'] += pa_mpjpe_3d
                total_metrics['n_mpjpe_3d'] += n_mpjpe_3d
                
                # 進捗バーに現在の損失を表示
                t.set_postfix(loss=loss.item())
    
    # 各損失と評価指標の平均を計算
    for key in total_losses:
        total_losses[key] /= num_batches
    
    for key in total_metrics:
        total_metrics[key] /= num_batches
    
    return total_losses, total_metrics


def main():
    # エラー検出を有効にする
    torch.autograd.set_detect_anomaly(True)
    
    parser = argparse.ArgumentParser(description='EPOCHフレームワークの訓練')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp'], default='mpiinf3dhp',
                        help='使用するデータセット')
    parser.add_argument('--data_path', type=str, default=None,
                        help='使用するデータセットのパス')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=30,
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='学習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='重み減衰')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='データローダーのワーカー数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--output_dir', type=str, default=MODELS_DIR,
                        help='モデル保存ディレクトリ')
    parser.add_argument('--regnet_path', type=str, default=None,
                        help='事前学習済みのRegNetモデルのパス')
    parser.add_argument('--liftnet_path', type=str, default=None,
                        help='事前学習済みのLiftNetモデルのパス')
    parser.add_argument('--nf_path', type=str, default=None,
                        help='事前学習済みのNormalizing Flowモデルのパス')
    parser.add_argument('--resume', type=str, default=None,
                        help='訓練を再開するためのチェックポイント')
    parser.add_argument('--freeze_regnet', action='store_true',
                        help='RegNetの重みを凍結するかどうか')
    parser.add_argument('--freeze_liftnet', action='store_true',
                        help='LiftNetの重みを凍結するかどうか')
    
    args = parser.parse_args()
    
    # 出力ディレクトリを作成
    os.makedirs(args.output_dir, exist_ok=True)
    
    # デバイスを設定
    device = torch.device(args.device)
    print(f"使用するデバイス: {device}")
    
    # データローダーを作成
    print(f"データローダーを作成しています... ({args.dataset})")
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        augment=True
    )
    
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        data_path=args.data_path,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # 関節数を取得
    sample = next(iter(train_loader))
    num_joints = sample['pose_2d'].shape[1]
    
    # Normalizing Flowモデルを読み込む
    normalizing_flow = None
    if args.nf_path is not None and os.path.exists(args.nf_path):
        print(f"Normalizing Flowモデルを読み込んでいます: {args.nf_path}")
        normalizing_flow = NormalizingFlow(
            num_features=num_joints * 2,  # 2Dポーズは各関節(x, y)
            hidden_features=MODEL_CONFIG['normalizing_flow']['hidden_channels'],
            num_blocks=MODEL_CONFIG['normalizing_flow']['num_flow_blocks']
        ).to(device)
        normalizing_flow.load_state_dict(torch.load(args.nf_path, map_location=device))
        normalizing_flow.eval()  # 評価モードに設定
    else:
        print("Normalizing Flowモデルが指定されていないか、見つかりません。NFLossは使用されません。")
    
    # EPOCHモデルを初期化
    model = EPOCH(
        num_joints=num_joints,
        encoder_name='resnet50',
        pretrained=True,
        image_size=(224, 224)
    ).to(device)
    
    # 事前学習済みの重みを読み込む
    if args.regnet_path is not None and os.path.exists(args.regnet_path):
        print(f"RegNetの重みを読み込んでいます: {args.regnet_path}")
        model.load_regnet_weights(args.regnet_path)
    
    if args.liftnet_path is not None and os.path.exists(args.liftnet_path):
        print(f"LiftNetの重みを読み込んでいます: {args.liftnet_path}")
        model.load_liftnet_weights(args.liftnet_path)
    
    # 特定のモジュールの重みを凍結
    if args.freeze_regnet:
        print("RegNetの重みを凍結します")
        for param in model.regnet.parameters():
            param.requires_grad = False
    
    if args.freeze_liftnet:
        print("LiftNetの重みを凍結します")
        for param in model.liftnet.parameters():
            param.requires_grad = False
    
    # 損失関数を初期化
    criterion = {
        'rle': ResidualLogLikelihoodLoss().to(device),
        'l2d': L2DLoss().to(device),
        'l3d': L3DLoss().to(device),
        'bone': BoneLoss().to(device),
        'limbs': LimbsLoss().to(device),
        'def': DeformationLoss().to(device),
        'nf': NFLoss(normalizing_flow).to(device) if normalizing_flow is not None else None
    }
    
    # 損失関数の重み - EPOCHモデル用の設定を使用
    loss_weights = TRAIN_CONFIG['loss_weights']['epoch']
    
    # オプティマイザ
    if TRAIN_CONFIG['epoch']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif TRAIN_CONFIG['epoch']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"未知のオプティマイザ: {TRAIN_CONFIG['epoch']['optimizer']}")
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # TensorBoardのSummaryWriter
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.output_dir, 'logs', f'epoch_{args.dataset}_{timestamp}')
    writer = SummaryWriter(log_dir=log_dir)
    
    # 開始エポックと最良の損失を初期化
    start_epoch = 0
    best_val_loss = float('inf')
    
    # 訓練を再開する場合
    if args.resume is not None:
        if os.path.isfile(args.resume):
            print(f"チェックポイントを読み込んでいます: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['best_val_loss']
            print(f"エポック {start_epoch} から訓練を再開します")
        else:
            print(f"チェックポイントが見つかりません: {args.resume}")
    
    # モデルを訓練
    print(f"EPOCHフレームワークを訓練しています...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"エポック {epoch+1}/{args.epochs}")
        
        # 時間を計測
        start_time = time.time()
        
        # 訓練
        train_losses, train_metrics = train_epoch(model, normalizing_flow, train_loader, optimizer, criterion, device, loss_weights)
        
        # 検証
        val_losses, val_metrics = validate(model, normalizing_flow, val_loader, criterion, device, loss_weights)
        
        # 経過時間を計算
        epoch_time = time.time() - start_time
        
        # 学習率スケジューラを更新
        scheduler.step(val_losses['total'])
        
        # 損失をTensorBoardに記録
        for loss_name, loss_value in train_losses.items():
            writer.add_scalar(f'Loss/train/{loss_name}', loss_value, epoch)
        
        for loss_name, loss_value in val_losses.items():
            writer.add_scalar(f'Loss/val/{loss_name}', loss_value, epoch)
        
        # 評価指標をTensorBoardに記録
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(f'Metrics/train/{metric_name}', metric_value, epoch)
        
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(f'Metrics/val/{metric_name}', metric_value, epoch)
        
        # 学習率をTensorBoardに記録
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 結果を表示
        print(f"訓練損失: {train_losses['total']:.4f}, 検証損失: {val_losses['total']:.4f}, 時間: {epoch_time:.2f}秒")
        print(f"訓練 MPJPE 3D: {train_metrics['mpjpe_3d']:.2f} mm, PA-MPJPE 3D: {train_metrics['pa_mpjpe_3d']:.2f} mm")
        print(f"検証 MPJPE 3D: {val_metrics['mpjpe_3d']:.2f} mm, PA-MPJPE 3D: {val_metrics['pa_mpjpe_3d']:.2f} mm")
        
        # 最良のモデルを保存
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(args.output_dir, f"epoch_{args.dataset}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, best_model_path)
            print(f"最良のモデルを保存しました: {best_model_path}")
        
        # 定期的にチェックポイントを保存
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            checkpoint_path = os.path.join(args.output_dir, f"epoch_{args.dataset}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_metrics': train_metrics,
                'val_metrics': val_metrics
            }, checkpoint_path)
            print(f"チェックポイントを保存しました: {checkpoint_path}")
    
    # 最終モデルを保存
    final_model_path = os.path.join(args.output_dir, f"epoch_{args.dataset}_final.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, final_model_path)
    print(f"最終モデルを保存しました: {final_model_path}")
    
    # TensorBoardのSummaryWriterを閉じる
    writer.close()
    
    print("EPOCHフレームワークの訓練が完了しました!")


if __name__ == '__main__':
    main()