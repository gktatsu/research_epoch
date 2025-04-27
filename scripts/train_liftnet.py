"""
LiftNetモデルの訓練スクリプト
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

from src.config import MODELS_DIR, DATASET_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, OUTPUT_DIR
from src.models import LiftNet, NormalizingFlow
from src.data import get_dataloader
from src.losses import L2DLoss, L3DLoss, BoneLoss, LimbsLoss, DeformationLoss, NFLoss


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
    """
    model.train()
    
    # 損失の合計
    total_losses = {
        'total': 0,
        'l2d': 0,
        'l3d': 0,
        'bone': 0,
        'limbs': 0,
        'def': 0,
        'nf': 0
    }
    
    # バッチ数
    num_batches = len(train_loader)
    
    # 進捗バー
    with tqdm(train_loader, desc='Train', leave=False) as t:
        for batch_idx, batch in enumerate(t):
            # データをデバイスに転送
            pose_2d = batch['pose_2d'].to(device)
            pose_3d = batch['pose_3d'].to(device)
            cam_K = batch['cam_K'].to(device)
            cam_R = batch['cam_R'].to(device)
            cam_t = batch['cam_t'].to(device)
            
            # 勾配をゼロに初期化
            optimizer.zero_grad()
            
            # 順伝播
            outputs = model(pose_2d, cam_K, cam_R, cam_t)
            
            # サイクル一貫性の出力を取得
            pred_pose_2d = outputs['pose_2d_rec']        # 再投影された2Dポーズ
            pred_pose_3d = outputs['pose_3d']            # 推定された3Dポーズ
            pred_pose_3d_rec = outputs['pose_3d_rec']    # 再構築された3Dポーズ
            rotated_pose_2d = outputs['pose_2d_rotated'] # 回転された2Dポーズ
            
            # 損失を計算
            loss_2d = criterion['l2d'](pred_pose_2d, pose_2d)
            loss_3d = criterion['l3d'](pred_pose_3d_rec, pred_pose_3d)
            loss_bone = criterion['bone'](pred_pose_3d)
            loss_limbs = criterion['limbs'](pred_pose_3d)
            loss_def = criterion['def'](pred_pose_3d, pred_pose_3d_rec)
            loss_nf = criterion['nf'](rotated_pose_2d)
            
            # 重み付き損失の合計
            loss = (
                loss_weights['l2d'] * loss_2d +
                loss_weights['l3d'] * loss_3d +
                loss_weights['bone'] * loss_bone +
                loss_weights['limbs'] * loss_limbs +
                loss_weights['def'] * loss_def +
                loss_weights['nf'] * loss_nf
            )
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()
            
            # 損失を累積
            total_losses['total'] += loss.item()
            total_losses['l2d'] += loss_2d.item()
            total_losses['l3d'] += loss_3d.item()
            total_losses['bone'] += loss_bone.item()
            total_losses['limbs'] += loss_limbs.item()
            total_losses['def'] += loss_def.item()
            total_losses['nf'] += loss_nf.item()
            
            # 進捗バーに現在の損失を表示
            t.set_postfix(loss=loss.item())
    
    # 各損失の平均を計算
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def validate(model, normalizing_flow, val_loader, criterion, device, loss_weights):
    """
    検証を実行
    
    Args:
        model: 評価するモデル
        normalizing_flow: 学習済みのNormalizing Flow
        val_loader: 検証データローダー
        criterion: 損失関数の辞書
        device: デバイス
        loss_weights: 損失関数の重み
        
    Returns:
        losses: 各損失の平均値
    """
    model.eval()
    
    # 損失の合計
    total_losses = {
        'total': 0,
        'l2d': 0,
        'l3d': 0,
        'bone': 0,
        'limbs': 0,
        'def': 0,
        'nf': 0
    }
    
    # バッチ数
    num_batches = len(val_loader)
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 進捗バー
        with tqdm(val_loader, desc='Validate', leave=False) as t:
            for batch_idx, batch in enumerate(t):
                # データをデバイスに転送
                pose_2d = batch['pose_2d'].to(device)
                pose_3d = batch['pose_3d'].to(device)
                cam_K = batch['cam_K'].to(device)
                cam_R = batch['cam_R'].to(device)
                cam_t = batch['cam_t'].to(device)
                
                # 順伝播
                outputs = model(pose_2d, cam_K, cam_R, cam_t)
                
                # サイクル一貫性の出力を取得
                pred_pose_2d = outputs['pose_2d_rec']        # 再投影された2Dポーズ
                pred_pose_3d = outputs['pose_3d']            # 推定された3Dポーズ
                pred_pose_3d_rec = outputs['pose_3d_rec']    # 再構築された3Dポーズ
                rotated_pose_2d = outputs['pose_2d_rotated'] # 回転された2Dポーズ
                
                # 損失を計算
                loss_2d = criterion['l2d'](pred_pose_2d, pose_2d)
                loss_3d = criterion['l3d'](pred_pose_3d_rec, pred_pose_3d)
                loss_bone = criterion['bone'](pred_pose_3d)
                loss_limbs = criterion['limbs'](pred_pose_3d)
                loss_def = criterion['def'](pred_pose_3d, pred_pose_3d_rec)
                loss_nf = criterion['nf'](rotated_pose_2d)
                
                # 重み付き損失の合計
                loss = (
                    loss_weights['l2d'] * loss_2d +
                    loss_weights['l3d'] * loss_3d +
                    loss_weights['bone'] * loss_bone +
                    loss_weights['limbs'] * loss_limbs +
                    loss_weights['def'] * loss_def +
                    loss_weights['nf'] * loss_nf
                )
                
                # 損失を累積
                total_losses['total'] += loss.item()
                total_losses['l2d'] += loss_2d.item()
                total_losses['l3d'] += loss_3d.item()
                total_losses['bone'] += loss_bone.item()
                total_losses['limbs'] += loss_limbs.item()
                total_losses['def'] += loss_def.item()
                total_losses['nf'] += loss_nf.item()
                
                # 進捗バーに現在の損失を表示
                t.set_postfix(loss=loss.item())
    
    # 各損失の平均を計算
    for key in total_losses:
        total_losses[key] /= num_batches
    
    return total_losses


def train_normalizing_flow(train_loader, num_features=34, hidden_features=128, num_blocks=6, 
                          num_epochs=50, device='cuda'):
    """
    Normalizing Flowモデルを訓練
    
    Args:
        train_loader: 訓練データローダー
        num_features: 特徴量の次元数（2Dポーズの場合は関節数*2）
        hidden_features: 隠れ層の次元数
        num_blocks: フローブロックの数
        num_epochs: エポック数
        device: デバイス
        
    Returns:
        normalizing_flow: 訓練されたNormalizing Flowモデル
    """
    print("Normalizing Flowモデルを訓練しています...")
    
    # Normalizing Flowモデルを初期化
    normalizing_flow = NormalizingFlow(
        num_features=num_features,
        hidden_features=hidden_features,
        num_blocks=num_blocks
    ).to(device)
    
    # オプティマイザ
    optimizer = optim.Adam(normalizing_flow.parameters(), lr=1e-4)
    
    # 各エポックで訓練
    for epoch in range(num_epochs):
        normalizing_flow.train()
        
        total_loss = 0
        num_batches = 0
        
        # 進捗バー
        with tqdm(train_loader, desc=f'NF Epoch {epoch+1}/{num_epochs}', leave=False) as t:
            for batch_idx, batch in enumerate(t):
                # 2Dポーズデータを取得
                pose_2d = batch['pose_2d'].to(device)
                batch_size = pose_2d.shape[0]
                
                # 2Dポーズを平坦化
                pose_2d_flat = pose_2d.reshape(batch_size, -1)
                
                # 勾配をゼロに初期化
                optimizer.zero_grad()
                
                # 負の対数尤度を計算
                neg_log_likelihood = -normalizing_flow.log_prob(pose_2d_flat)
                loss = torch.mean(neg_log_likelihood)
                
                # 逆伝播
                loss.backward()
                
                # パラメータを更新
                optimizer.step()
                
                # 損失を累積
                total_loss += loss.item()
                num_batches += 1
                
                # 進捗バーに現在の損失を表示
                t.set_postfix(loss=loss.item())
        
        # エポックの平均損失を計算
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}")
    
    print("Normalizing Flowモデルの訓練が完了しました!")
    
    return normalizing_flow


def main():
    parser = argparse.ArgumentParser(description='LiftNetモデルの訓練')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp'], default='human36m',
                        help='使用するデータセット')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['lift_net']['batch_size'],
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['lift_net']['epochs'],
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['lift_net']['lr'],
                        help='学習率')
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['lift_net']['weight_decay'],
                        help='重み減衰')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='データローダーのワーカー数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='モデル保存ディレクトリ')
    parser.add_argument('--nf_epochs', type=int, default=50,
                        help='Normalizing Flowの訓練エポック数')
    parser.add_argument('--resume', type=str, default=None,
                        help='訓練を再開するためのチェックポイント')
    
    args = parser.parse_args()
    
    # 実行日時に基づいたユニークなディレクトリ名を生成
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = os.path.join(OUTPUT_DIR, f'run_{timestamp}')
    
    # サブディレクトリを作成
    models_dir = os.path.join(run_dir, 'models')
    logs_dir = os.path.join(run_dir, 'logs')
    vis_dir = os.path.join(run_dir, 'visualizations')
    
    # ディレクトリを作成
    for dir_path in [run_dir, models_dir, logs_dir, vis_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 出力先をカスタムディレクトリに変更
    if args.output_dir is None:
        args.output_dir = OUTPUT_DIR
    
    # デバイスを設定
    device = torch.device(args.device)
    print(f"使用するデバイス: {device}")
    
    # データローダーを作成
    print(f"データローダーを作成しています... ({args.dataset})")
    train_loader = get_dataloader(
        dataset_name=args.dataset,
        split='train',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        augment=True
    )
    
    val_loader = get_dataloader(
        dataset_name=args.dataset,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # 関節数を取得
    sample = next(iter(train_loader))
    num_joints = sample['pose_2d'].shape[1]
    
    # Normalizing Flowを訓練
    normalizing_flow = train_normalizing_flow(
        train_loader=train_loader,
        num_features=num_joints * 2,  # 2Dポーズは各関節(x, y)
        hidden_features=MODEL_CONFIG['normalizing_flow']['hidden_channels'],
        num_blocks=MODEL_CONFIG['normalizing_flow']['num_flow_blocks'],
        num_epochs=args.nf_epochs,
        device=device
    )
    
    # Normalizing Flowモデルを保存
    nf_path = os.path.join(args.output_dir, f"normalizing_flow_{args.dataset}.pth")
    torch.save(normalizing_flow.state_dict(), nf_path)
    print(f"Normalizing Flowモデルを保存しました: {nf_path}")
    
    # LiftNetモデルを初期化
    model = LiftNet(
        num_joints=num_joints,
        feat_dim=MODEL_CONFIG['lift_net']['dim_l'],
        num_residual_blocks=MODEL_CONFIG['lift_net']['residual_blocks']
    ).to(device)
    
    # 損失関数を初期化
    criterion = {
        'l2d': L2DLoss().to(device),
        'l3d': L3DLoss().to(device),
        'bone': BoneLoss().to(device),
        'limbs': LimbsLoss().to(device),
        'def': DeformationLoss().to(device),
        'nf': NFLoss(normalizing_flow).to(device)
    }
    
    # 損失関数の重み
    loss_weights = TRAIN_CONFIG['loss_weights']['lift_net']
    
    # オプティマイザ
    if TRAIN_CONFIG['lift_net']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif TRAIN_CONFIG['lift_net']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"未知のオプティマイザ: {TRAIN_CONFIG['lift_net']['optimizer']}")
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )
    
    # TensorBoardのSummaryWriter
    log_dir = os.path.join(OUTPUT_DIR, f'liftnet_{args.dataset}')
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
    print(f"LiftNetモデルを訓練しています...")
    
    for epoch in range(start_epoch, args.epochs):
        print(f"エポック {epoch+1}/{args.epochs}")
        
        # 時間を計測
        start_time = time.time()
        
        # 訓練
        train_losses = train_epoch(model, normalizing_flow, train_loader, optimizer, criterion, device, loss_weights)
        
        # 検証
        val_losses = validate(model, normalizing_flow, val_loader, criterion, device, loss_weights)
        
        # 経過時間を計算
        epoch_time = time.time() - start_time
        
        # 学習率スケジューラを更新
        scheduler.step(val_losses['total'])
        
        # 損失をTensorBoardに記録
        writer.add_scalar('Loss/train/total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train/l2d', train_losses['l2d'], epoch)
        writer.add_scalar('Loss/train/l3d', train_losses['l3d'], epoch)
        writer.add_scalar('Loss/train/bone', train_losses['bone'], epoch)
        writer.add_scalar('Loss/train/limbs', train_losses['limbs'], epoch)
        writer.add_scalar('Loss/train/def', train_losses['def'], epoch)
        writer.add_scalar('Loss/train/nf', train_losses['nf'], epoch)
        
        writer.add_scalar('Loss/val/total', val_losses['total'], epoch)
        writer.add_scalar('Loss/val/l2d', val_losses['l2d'], epoch)
        writer.add_scalar('Loss/val/l3d', val_losses['l3d'], epoch)
        writer.add_scalar('Loss/val/bone', val_losses['bone'], epoch)
        writer.add_scalar('Loss/val/limbs', val_losses['limbs'], epoch)
        writer.add_scalar('Loss/val/def', val_losses['def'], epoch)
        writer.add_scalar('Loss/val/nf', val_losses['nf'], epoch)
        
        # 学習率をTensorBoardに記録
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 結果を表示
        print(f"訓練損失: {train_losses['total']:.4f}, 検証損失: {val_losses['total']:.4f}, 時間: {epoch_time:.2f}秒")
        
        # 最良のモデルを保存
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(args.output_dir, f"liftnet_{args.dataset}_best.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, best_model_path)
            print(f"最良のモデルを保存しました: {best_model_path}")
        
        # 定期的にチェックポイントを保存
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_dir, f"liftnet_{args.dataset}_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            print(f"チェックポイントを保存しました: {checkpoint_path}")
    
    # 最終モデルを保存
    final_model_path = os.path.join(args.output_dir, f"liftnet_{args.dataset}_final.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, final_model_path)
    print(f"最終モデルを保存しました: {final_model_path}")
    
    # TensorBoardのSummaryWriterを閉じる
    writer.close()
    
    print("LiftNetモデルの訓練が完了しました!")


if __name__ == '__main__':
    main()