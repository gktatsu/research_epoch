"""
EPOCHモデルの評価スクリプト

学習済みモデルをロードして、Human3.6MとMPI-INF-3DHPデータセットで評価
"""
import os
import sys
import argparse
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_DIR, MODELS_DIR, DATASET_CONFIG
from src.models import EPOCH, RegNet, LiftNet
from src.data import get_dataloader
from src.utils.metrics import (
    calculate_mpjpe, calculate_pa_mpjpe, calculate_n_mpjpe, 
    calculate_pck, calculate_auc
)


def evaluate_regnet(model, dataloader, device):
    """
    RegNetモデルの評価を実行
    
    Args:
        model: 評価するRegNetモデル
        dataloader: 評価用データローダー
        device: デバイス
        
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    # 評価指標の集計用
    metrics = {
        'mpjpe_2d': [],
        'mpjpe_3d': [],
        'pa_mpjpe_3d': [],
        'n_mpjpe_3d': [],
        'pck': [],
        'auc': []
    }
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 進捗バー
        for batch in tqdm(dataloader, desc='Evaluating RegNet'):
            # データをデバイスに転送
            image = batch['image'].to(device)
            pose_2d_gt = batch['pose_2d'].to(device)
            pose_3d_gt = batch['pose_3d'].to(device)
            
            # 入力画像のバウンディングボックスは画像全体
            batch_size = image.shape[0]
            h, w = image.shape[2], image.shape[3]
            bbox = torch.tensor([[0, 0, w, h]], device=device).repeat(batch_size, 1).float()
            
            # 推論
            outputs = model(image, bbox)
            
            pose_2d_pred = outputs['pose_2d']
            pose_3d_pred = outputs['pose_3d']
            
            # CPUに転送して評価指標を計算
            pose_2d_pred_np = pose_2d_pred.cpu().numpy()
            pose_2d_gt_np = pose_2d_gt.cpu().numpy()
            pose_3d_pred_np = pose_3d_pred.cpu().numpy()
            pose_3d_gt_np = pose_3d_gt.cpu().numpy()
            
            # 評価指標を計算
            # 2D MPJPE (ピクセル単位)
            mpjpe_2d = np.mean(np.sqrt(np.sum((pose_2d_pred_np - pose_2d_gt_np) ** 2, axis=-1)))
            
            # 3D MPJPE (mm単位)
            mpjpe_3d = calculate_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            pa_mpjpe_3d = calculate_pa_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            n_mpjpe_3d = calculate_n_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            
            # PCKとAUC (MPI-INF-3DHPでの標準指標)
            pck = calculate_pck(pose_3d_pred_np, pose_3d_gt_np, threshold=150)
            auc = calculate_auc(pose_3d_pred_np, pose_3d_gt_np)
            
            # バッチの評価指標を集計
            metrics['mpjpe_2d'].append(mpjpe_2d)
            metrics['mpjpe_3d'].append(mpjpe_3d)
            metrics['pa_mpjpe_3d'].append(pa_mpjpe_3d)
            metrics['n_mpjpe_3d'].append(n_mpjpe_3d)
            metrics['pck'].append(pck)
            metrics['auc'].append(auc)
    
    # 全バッチの平均を計算
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return metrics


def evaluate_liftnet(model, dataloader, device):
    """
    LiftNetモデルの評価を実行
    
    Args:
        model: 評価するLiftNetモデル
        dataloader: 評価用データローダー
        device: デバイス
        
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    # 評価指標の集計用
    metrics = {
        'mpjpe_3d': [],
        'pa_mpjpe_3d': [],
        'n_mpjpe_3d': [],
        'pck': [],
        'auc': []
    }
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 進捗バー
        for batch in tqdm(dataloader, desc='Evaluating LiftNet'):
            # データをデバイスに転送
            pose_2d = batch['pose_2d'].to(device)
            pose_3d_gt = batch['pose_3d'].to(device)
            cam_K = batch['cam_K'].to(device)
            cam_R = batch['cam_R'].to(device)
            cam_t = batch['cam_t'].to(device)
            
            # 推論 (簡易バージョンを使用)
            pose_3d_pred = model.estimate_3d_pose(pose_2d, cam_K, cam_R, cam_t)
            
            # CPUに転送して評価指標を計算
            pose_3d_pred_np = pose_3d_pred.cpu().numpy()
            pose_3d_gt_np = pose_3d_gt.cpu().numpy()
            
            # 評価指標を計算
            mpjpe_3d = calculate_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            pa_mpjpe_3d = calculate_pa_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            n_mpjpe_3d = calculate_n_mpjpe(pose_3d_pred_np, pose_3d_gt_np)
            
            # PCKとAUC (MPI-INF-3DHPでの標準指標)
            pck = calculate_pck(pose_3d_pred_np, pose_3d_gt_np, threshold=150)
            auc = calculate_auc(pose_3d_pred_np, pose_3d_gt_np)
            
            # バッチの評価指標を集計
            metrics['mpjpe_3d'].append(mpjpe_3d)
            metrics['pa_mpjpe_3d'].append(pa_mpjpe_3d)
            metrics['n_mpjpe_3d'].append(n_mpjpe_3d)
            metrics['pck'].append(pck)
            metrics['auc'].append(auc)
    
    # 全バッチの平均を計算
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return metrics


def evaluate_epoch(model, dataloader, device):
    """
    EPOCHモデル（RegNetとLiftNetの統合モデル）の評価を実行
    
    Args:
        model: 評価するEPOCHモデル
        dataloader: 評価用データローダー
        device: デバイス
        
    Returns:
        metrics: 評価指標の辞書
    """
    model.eval()
    
    # 評価指標の集計用
    metrics = {
        'mpjpe_2d': [],
        'mpjpe_3d_reg': [],  # RegNetの3D推定結果
        'mpjpe_3d_lift': [], # LiftNetの3D推定結果
        'pa_mpjpe_3d_reg': [],
        'pa_mpjpe_3d_lift': [],
        'n_mpjpe_3d_reg': [],
        'n_mpjpe_3d_lift': [],
        'pck_reg': [],
        'pck_lift': [],
        'auc_reg': [],
        'auc_lift': []
    }
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 進捗バー
        for batch in tqdm(dataloader, desc='Evaluating EPOCH'):
            # データをデバイスに転送
            image = batch['image'].to(device)
            pose_2d_gt = batch['pose_2d'].to(device)
            pose_3d_gt = batch['pose_3d'].to(device)
            
            # 入力画像のバウンディングボックスは画像全体
            batch_size = image.shape[0]
            h, w = image.shape[2], image.shape[3]
            bbox = torch.tensor([[0, 0, w, h]], device=device).repeat(batch_size, 1).float()
            
            # エンドツーエンドで推論
            outputs = model(image, bbox)
            
            pose_2d_pred = outputs['pose_2d']
            pose_3d_reg = outputs['pose_3d_reg']   # RegNetの出力
            pose_3d_lift = outputs['pose_3d_lift'] # LiftNetの出力
            
            # CPUに転送して評価指標を計算
            pose_2d_pred_np = pose_2d_pred.cpu().numpy()
            pose_2d_gt_np = pose_2d_gt.cpu().numpy()
            pose_3d_reg_np = pose_3d_reg.cpu().numpy()
            pose_3d_lift_np = pose_3d_lift.cpu().numpy()
            pose_3d_gt_np = pose_3d_gt.cpu().numpy()
            
            # 評価指標を計算
            # 2D MPJPE (ピクセル単位)
            mpjpe_2d = np.mean(np.sqrt(np.sum((pose_2d_pred_np - pose_2d_gt_np) ** 2, axis=-1)))
            
            # 3D MPJPE (mm単位) for RegNet output
            mpjpe_3d_reg = calculate_mpjpe(pose_3d_reg_np, pose_3d_gt_np)
            pa_mpjpe_3d_reg = calculate_pa_mpjpe(pose_3d_reg_np, pose_3d_gt_np)
            n_mpjpe_3d_reg = calculate_n_mpjpe(pose_3d_reg_np, pose_3d_gt_np)
            
            # 3D MPJPE (mm単位) for LiftNet output
            mpjpe_3d_lift = calculate_mpjpe(pose_3d_lift_np, pose_3d_gt_np)
            pa_mpjpe_3d_lift = calculate_pa_mpjpe(pose_3d_lift_np, pose_3d_gt_np)
            n_mpjpe_3d_lift = calculate_n_mpjpe(pose_3d_lift_np, pose_3d_gt_np)
            
            # PCKとAUC (MPI-INF-3DHPでの標準指標) for RegNet output
            pck_reg = calculate_pck(pose_3d_reg_np, pose_3d_gt_np, threshold=150)
            auc_reg = calculate_auc(pose_3d_reg_np, pose_3d_gt_np)
            
            # PCKとAUC (MPI-INF-3DHPでの標準指標) for LiftNet output
            pck_lift = calculate_pck(pose_3d_lift_np, pose_3d_gt_np, threshold=150)
            auc_lift = calculate_auc(pose_3d_lift_np, pose_3d_gt_np)
            
            # バッチの評価指標を集計
            metrics['mpjpe_2d'].append(mpjpe_2d)
            metrics['mpjpe_3d_reg'].append(mpjpe_3d_reg)
            metrics['mpjpe_3d_lift'].append(mpjpe_3d_lift)
            metrics['pa_mpjpe_3d_reg'].append(pa_mpjpe_3d_reg)
            metrics['pa_mpjpe_3d_lift'].append(pa_mpjpe_3d_lift)
            metrics['n_mpjpe_3d_reg'].append(n_mpjpe_3d_reg)
            metrics['n_mpjpe_3d_lift'].append(n_mpjpe_3d_lift)
            metrics['pck_reg'].append(pck_reg)
            metrics['pck_lift'].append(pck_lift)
            metrics['auc_reg'].append(auc_reg)
            metrics['auc_lift'].append(auc_lift)
    
    # 全バッチの平均を計算
    for key in metrics:
        metrics[key] = float(np.mean(metrics[key]))
    
    return metrics


def print_metrics(metrics, model_type="EPOCH"):
    """
    評価指標を表示
    
    Args:
        metrics: 評価指標の辞書
        model_type: モデルの種類 ("EPOCH", "RegNet", or "LiftNet")
    """
    print(f"\n{model_type}モデルの評価結果:")
    
    if 'mpjpe_2d' in metrics:
        print(f"2D MPJPE: {metrics['mpjpe_2d']:.2f} px")
    
    if model_type == "EPOCH":
        print("\nRegNet出力の評価:")
        print(f"MPJPE: {metrics['mpjpe_3d_reg']:.2f} mm")
        print(f"PA-MPJPE: {metrics['pa_mpjpe_3d_reg']:.2f} mm")
        print(f"N-MPJPE: {metrics['n_mpjpe_3d_reg']:.2f} mm")
        print(f"PCK@150mm: {metrics['pck_reg']:.2f}%")
        print(f"AUC: {metrics['auc_reg']:.4f}")
        
        print("\nLiftNet出力の評価:")
        print(f"MPJPE: {metrics['mpjpe_3d_lift']:.2f} mm")
        print(f"PA-MPJPE: {metrics['pa_mpjpe_3d_lift']:.2f} mm")
        print(f"N-MPJPE: {metrics['n_mpjpe_3d_lift']:.2f} mm")
        print(f"PCK@150mm: {metrics['pck_lift']:.2f}%")
        print(f"AUC: {metrics['auc_lift']:.4f}")
    
    elif model_type in ["RegNet", "LiftNet"]:
        print(f"MPJPE: {metrics['mpjpe_3d']:.2f} mm")
        print(f"PA-MPJPE: {metrics['pa_mpjpe_3d']:.2f} mm")
        print(f"N-MPJPE: {metrics['n_mpjpe_3d']:.2f} mm")
        print(f"PCK@150mm: {metrics['pck']:.2f}%")
        print(f"AUC: {metrics['auc']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='EPOCHモデルの評価')
    parser.add_argument('--model_path', type=str, required=True,
                        help='評価するモデルのパス')
    parser.add_argument('--model_type', type=str, choices=['epoch', 'regnet', 'liftnet'], default='epoch',
                        help='モデルのタイプ')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp'], default='human36m',
                        help='評価に使用するデータセット')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='バッチサイズ')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='データローダーのワーカー数')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--output_file', type=str, default=None,
                        help='評価結果を保存するファイル (JSON形式)')
    
    args = parser.parse_args()
    
    # モデルのタイプを取得
    model_type = args.model_type.lower()
    
    # デバイスを設定
    device = torch.device(args.device)
    print(f"使用するデバイス: {device}")
    
    # データローダーを作成
    print(f"データローダーを作成しています... ({args.dataset})")
    test_loader = get_dataloader(
        dataset_name=args.dataset,
        split='test',
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        augment=False
    )
    
    # モデルをロード
    print(f"モデルをロードしています: {args.model_path}")
    
    if model_type == 'epoch':
        # EPOCHモデルをロード
        sample = next(iter(test_loader))
        num_joints = sample['pose_2d'].shape[1]
        
        model = EPOCH(
            num_joints=num_joints,
            encoder_name='resnet50',
            pretrained=False,
            image_size=(224, 224)
        ).to(device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 評価
        metrics = evaluate_epoch(model, test_loader, device)
        print_metrics(metrics, model_type="EPOCH")
    
    elif model_type == 'regnet':
        # RegNetモデルをロード
        sample = next(iter(test_loader))
        num_joints = sample['pose_2d'].shape[1]
        
        model = RegNet(
            num_joints=num_joints,
            encoder_name='resnet50',
            pretrained=False,
            image_size=(224, 224)
        ).to(device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 評価
        metrics = evaluate_regnet(model, test_loader, device)
        print_metrics(metrics, model_type="RegNet")
    
    elif model_type == 'liftnet':
        # LiftNetモデルをロード
        sample = next(iter(test_loader))
        num_joints = sample['pose_2d'].shape[1]
        
        model = LiftNet(
            num_joints=num_joints,
            feat_dim=1024,
            num_residual_blocks=3
        ).to(device)
        
        checkpoint = torch.load(args.model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 評価
        metrics = evaluate_liftnet(model, test_loader, device)
        print_metrics(metrics, model_type="LiftNet")
    
    # 評価結果を保存
    if args.output_file is not None:
        output_dir = os.path.dirname(args.output_file)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        with open(args.output_file, 'w') as f:
            json.dump({
                'model_type': model_type,
                'dataset': args.dataset,
                'metrics': metrics
            }, f, indent=4)
        
        print(f"評価結果を保存しました: {args.output_file}")


if __name__ == '__main__':
    main()