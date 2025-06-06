"""
RegNetモデルの訓練スクリプト
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
from src.models import RegNet, NormalizingFlow
from src.data import get_dataloader
from src.losses import BoneLoss, LimbsLoss, NFLoss, ResidualLogLikelihoodLoss
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
        'nf': 0
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
            
            pose_2d_pred = outputs['pose_2d']         # 推定された2Dポーズ
            pose_3d_pred = outputs['pose_3d']         # 推定された3Dポーズ
            joint_presence = outputs['joint_presence'] # 関節存在確率
            
            # 回転されたポーズと2D投影
            rotated_pose_2d = outputs.get('pose_2d_rotated', None)
            
            # 関節存在確率をシグモイド関数で0~1に変換
            joint_presence = torch.sigmoid(joint_presence)
            
            # 損失を計算
            loss_rle = criterion['rle'](pose_2d_pred, pose_2d_gt, joint_presence)
            loss_bone = criterion['bone'](pose_3d_pred)
            loss_limbs = criterion['limbs'](pose_3d_pred)
            
            # Normalizing Flow損失（回転されたポーズがある場合）
            loss_nf = torch.tensor(0.0, device=device)
            if rotated_pose_2d is not None and criterion['nf'] is not None:
                loss_nf = criterion['nf'](rotated_pose_2d)
            
            # 重み付き損失の合計
            loss = (
                loss_weights['rle'] * loss_rle +
                loss_weights['bone'] * loss_bone +
                loss_weights['limbs'] * loss_limbs
            )
            
            if rotated_pose_2d is not None and criterion['nf'] is not None:
                loss += loss_weights['nf'] * loss_nf
            
            # 逆伝播
            loss.backward()
            
            # パラメータを更新
            optimizer.step()
            
            # 損失を累積
            total_losses['total'] += loss.item()
            total_losses['rle'] += loss_rle.item()
            total_losses['bone'] += loss_bone.item()
            total_losses['limbs'] += loss_limbs.item()
            if rotated_pose_2d is not None and criterion['nf'] is not None:
                total_losses['nf'] += loss_nf.item()
            
            # 評価指標を計算
            # 2Dポーズの平均関節位置誤差（ピクセル単位）
            mpjpe_2d = torch.mean(torch.sqrt(torch.sum((pose_2d_pred - pose_2d_gt) ** 2, dim=-1)))
            
            # 3Dポーズの平均関節位置誤差（mm単位）
            # CPUに転送して計算（NumPy関数を使用するため）
            pose_3d_pred_np = pose_3d_pred.detach().cpu().numpy()
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
        model: 評価するモデル
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
        'nf': 0
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
                
                pose_2d_pred = outputs['pose_2d']         # 推定された2Dポーズ
                pose_3d_pred = outputs['pose_3d']         # 推定された3Dポーズ
                joint_presence = outputs['joint_presence'] # 関節存在確率
                
                # 回転されたポーズと2D投影
                rotated_pose_2d = outputs.get('pose_2d_rotated', None)
                
                # 関節存在確率をシグモイド関数で0~1に変換
                joint_presence = torch.sigmoid(joint_presence)
                
                # 損失を計算
                loss_rle = criterion['rle'](pose_2d_pred, pose_2d_gt, joint_presence)
                loss_bone = criterion['bone'](pose_3d_pred)
                loss_limbs = criterion['limbs'](pose_3d_pred)
                
                # Normalizing Flow損失（回転されたポーズがある場合）
                loss_nf = torch.tensor(0.0, device=device)
                if rotated_pose_2d is not None and criterion['nf'] is not None:
                    loss_nf = criterion['nf'](rotated_pose_2d)
                
                # 重み付き損失の合計
                loss = (
                    loss_weights['rle'] * loss_rle +
                    loss_weights['bone'] * loss_bone +
                    loss_weights['limbs'] * loss_limbs
                )
                
                if rotated_pose_2d is not None and criterion['nf'] is not None:
                    loss += loss_weights['nf'] * loss_nf
                
                # 損失を累積
                total_losses['total'] += loss.item()
                total_losses['rle'] += loss_rle.item()
                total_losses['bone'] += loss_bone.item()
                total_losses['limbs'] += loss_limbs.item()
                if rotated_pose_2d is not None and criterion['nf'] is not None:
                    total_losses['nf'] += loss_nf.item()
                
                # 評価指標を計算
                # 2Dポーズの平均関節位置誤差（ピクセル単位）
                mpjpe_2d = torch.mean(torch.sqrt(torch.sum((pose_2d_pred - pose_2d_gt) ** 2, dim=-1)))
                
                # 3Dポーズの平均関節位置誤差（mm単位）
                # CPUに転送して計算（NumPy関数を使用するため）
                pose_3d_pred_np = pose_3d_pred.detach().cpu().numpy()
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
    parser = argparse.ArgumentParser(description='RegNetモデルの訓練')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp'], default='human36m',
                        help='使用するデータセット')
    parser.add_argument('--batch_size', type=int, default=TRAIN_CONFIG['reg_net']['batch_size'],
                        help='バッチサイズ')
    parser.add_argument('--epochs', type=int, default=TRAIN_CONFIG['reg_net']['epochs'],
                        help='エポック数')
    parser.add_argument('--lr', type=float, default=TRAIN_CONFIG['reg_net']['lr'],
                        help='学習率')
    parser.add_argument('--weight_decay', type=float, default=TRAIN_CONFIG['reg_net']['weight_decay'],
                        help='重み減衰')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='データローダーのワーカー数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR,
                        help='モデル保存ディレクトリ')
    parser.add_argument('--nf_path', type=str, default=None,
                        help='事前学習済みのNormalizing Flowモデルのパス')
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
    
    # RegNetモデルを初期化
    model = RegNet(
        num_joints=num_joints,
        encoder_name='resnet50',
        pretrained=True,
        image_size=(224, 224)
    ).to(device)
    
    # 損失関数を初期化
    criterion = {
        'rle': ResidualLogLikelihoodLoss().to(device),
        'bone': BoneLoss().to(device),
        'limbs': LimbsLoss().to(device),
        'nf': NFLoss(normalizing_flow).to(device) if normalizing_flow is not None else None
    }
    
    # 損失関数の重み
    loss_weights = TRAIN_CONFIG['loss_weights']['reg_net']
    
    # オプティマイザ
    if TRAIN_CONFIG['reg_net']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif TRAIN_CONFIG['reg_net']['optimizer'].lower() == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"未知のオプティマイザ: {TRAIN_CONFIG['reg_net']['optimizer']}")
    
    # 学習率スケジューラ
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=10,
        verbose=True
    )
    
    # TensorBoardのSummaryWriter
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(OUTPUT_DIR, 'logs', f'regnet_{args.dataset}_{timestamp}')
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
    print(f"RegNetモデルを訓練しています...")
    
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
        writer.add_scalar('Loss/train/total', train_losses['total'], epoch)
        writer.add_scalar('Loss/train/rle', train_losses['rle'], epoch)
        writer.add_scalar('Loss/train/bone', train_losses['bone'], epoch)
        writer.add_scalar('Loss/train/limbs', train_losses['limbs'], epoch)
        if normalizing_flow is not None:
            writer.add_scalar('Loss/train/nf', train_losses['nf'], epoch)
        
        writer.add_scalar('Loss/val/total', val_losses['total'], epoch)
        writer.add_scalar('Loss/val/rle', val_losses['rle'], epoch)
        writer.add_scalar('Loss/val/bone', val_losses['bone'], epoch)
        writer.add_scalar('Loss/val/limbs', val_losses['limbs'], epoch)
        if normalizing_flow is not None:
            writer.add_scalar('Loss/val/nf', val_losses['nf'], epoch)
        
        # 評価指標をTensorBoardに記録
        writer.add_scalar('Metrics/train/mpjpe_2d', train_metrics['mpjpe_2d'], epoch)
        writer.add_scalar('Metrics/train/mpjpe_3d', train_metrics['mpjpe_3d'], epoch)
        writer.add_scalar('Metrics/train/pa_mpjpe_3d', train_metrics['pa_mpjpe_3d'], epoch)
        writer.add_scalar('Metrics/train/n_mpjpe_3d', train_metrics['n_mpjpe_3d'], epoch)
        
        writer.add_scalar('Metrics/val/mpjpe_2d', val_metrics['mpjpe_2d'], epoch)
        writer.add_scalar('Metrics/val/mpjpe_3d', val_metrics['mpjpe_3d'], epoch)
        writer.add_scalar('Metrics/val/pa_mpjpe_3d', val_metrics['pa_mpjpe_3d'], epoch)
        writer.add_scalar('Metrics/val/n_mpjpe_3d', val_metrics['n_mpjpe_3d'], epoch)
        
        # 学習率をTensorBoardに記録
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # 結果を表示
        print(f"訓練損失: {train_losses['total']:.4f}, 検証損失: {val_losses['total']:.4f}, 時間: {epoch_time:.2f}秒")
        print(f"訓練 MPJPE 3D: {train_metrics['mpjpe_3d']:.2f} mm, PA-MPJPE 3D: {train_metrics['pa_mpjpe_3d']:.2f} mm")
        print(f"検証 MPJPE 3D: {val_metrics['mpjpe_3d']:.2f} mm, PA-MPJPE 3D: {val_metrics['pa_mpjpe_3d']:.2f} mm")
        
        # 最良のモデルを保存
        if val_losses['total'] < best_val_loss:
            best_val_loss = val_losses['total']
            best_model_path = os.path.join(models_dir, f"regnet_{args.dataset}_best.pth")
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
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(models_dir, f"regnet_{args.dataset}_epoch_{epoch+1}.pth")
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
    final_model_path = os.path.join(OUTPUT_DIR, f"regnet_{args.dataset}_final.pth")
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss
    }, final_model_path)
    print(f"最終モデルを保存しました: {final_model_path}")
    
    # TensorBoardのSummaryWriterを閉じる
    writer.close()
    
    print("RegNetモデルの訓練が完了しました!")


if __name__ == '__main__':
    main()