"""
訓練結果の保存ユーティリティ
"""
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import torch
from .visualization import visualize_2d_pose, visualize_3d_pose, visualize_results

def save_checkpoint(model, optimizer, epoch, best_metric, metrics, losses, save_path):
    """
    モデルのチェックポイントを保存
    
    Args:
        model: 保存するモデル
        optimizer: オプティマイザ
        epoch: 現在のエポック
        best_metric: 最良のメトリック値
        metrics: 評価指標の辞書
        losses: 損失の辞書
        save_path: 保存先パス
    """
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_metric': best_metric,
        'metrics': metrics,
        'losses': losses
    }, save_path)
    print(f"チェックポイントを保存しました: {save_path}")


def save_metrics_to_csv(train_metrics, val_metrics, train_losses, val_losses, save_dir, filename='metrics'):
    """
    訓練中のメトリクスをCSVファイルに保存
    
    Args:
        train_metrics: 訓練メトリクスの履歴 (エポックごとの辞書のリスト)
        val_metrics: 検証メトリクスの履歴 (エポックごとの辞書のリスト)
        train_losses: 訓練損失の履歴 (エポックごとの辞書のリスト)
        val_losses: 検証損失の履歴 (エポックごとの辞書のリスト)
        save_dir: 保存先ディレクトリ
        filename: CSVファイル名 (拡張子なし)
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # CSVファイルのパス
    csv_path = os.path.join(save_dir, f"{filename}.csv")
    
    # 列ヘッダーを作成
    header = ['epoch']
    
    # 訓練損失の列名を追加
    for key in train_losses[0].keys():
        header.append(f'train_loss_{key}')
    
    # 検証損失の列名を追加
    for key in val_losses[0].keys():
        header.append(f'val_loss_{key}')
    
    # 訓練メトリクスの列名を追加
    for key in train_metrics[0].keys():
        header.append(f'train_metric_{key}')
    
    # 検証メトリクスの列名を追加
    for key in val_metrics[0].keys():
        header.append(f'val_metric_{key}')
    
    # CSVファイルに書き込み
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=header)
        writer.writeheader()
        
        for epoch in range(len(train_metrics)):
            row = {'epoch': epoch + 1}
            
            # 訓練損失を追加
            for key, value in train_losses[epoch].items():
                row[f'train_loss_{key}'] = value
            
            # 検証損失を追加
            for key, value in val_losses[epoch].items():
                row[f'val_loss_{key}'] = value
            
            # 訓練メトリクスを追加
            for key, value in train_metrics[epoch].items():
                row[f'train_metric_{key}'] = value
            
            # 検証メトリクスを追加
            for key, value in val_metrics[epoch].items():
                row[f'val_metric_{key}'] = value
            
            writer.writerow(row)
    
    print(f"メトリクスをCSVに保存しました: {csv_path}")


def plot_metrics(train_metrics, val_metrics, save_dir, filename_prefix='metric'):
    """
    メトリクスの推移をプロット
    
    Args:
        train_metrics: 訓練メトリクスの履歴 (エポックごとの辞書のリスト)
        val_metrics: 検証メトリクスの履歴 (エポックごとの辞書のリスト)
        save_dir: 保存先ディレクトリ
        filename_prefix: ファイル名のプレフィックス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # 各メトリクスのキーを取得
    metric_keys = train_metrics[0].keys()
    
    # エポック数
    epochs = range(1, len(train_metrics) + 1)
    
    # 各メトリクスについてプロット
    for key in metric_keys:
        plt.figure(figsize=(10, 6))
        
        # 訓練メトリクスをプロット
        train_values = [metrics[key] for metrics in train_metrics]
        plt.plot(epochs, train_values, 'b-', label=f'Train {key}')
        
        # 検証メトリクスをプロット
        val_values = [metrics[key] for metrics in val_metrics]
        plt.plot(epochs, val_values, 'r-', label=f'Validation {key}')
        
        plt.title(f'{key} vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # プロットを保存
        save_path = os.path.join(save_dir, f"{filename_prefix}_{key}.png")
        plt.savefig(save_path)
        plt.close()
    
    print(f"メトリクスのプロットを保存しました: {save_dir}")


def plot_losses(train_losses, val_losses, save_dir, filename_prefix='loss'):
    """
    損失の推移をプロット
    
    Args:
        train_losses: 訓練損失の履歴 (エポックごとの辞書のリスト)
        val_losses: 検証損失の履歴 (エポックごとの辞書のリスト)
        save_dir: 保存先ディレクトリ
        filename_prefix: ファイル名のプレフィックス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # 各損失のキーを取得
    loss_keys = train_losses[0].keys()
    
    # エポック数
    epochs = range(1, len(train_losses) + 1)
    
    # 各損失についてプロット
    for key in loss_keys:
        plt.figure(figsize=(10, 6))
        
        # 訓練損失をプロット
        train_values = [losses[key] for losses in train_losses]
        plt.plot(epochs, train_values, 'b-', label=f'Train {key}')
        
        # 検証損失をプロット
        val_values = [losses[key] for losses in val_losses]
        plt.plot(epochs, val_values, 'r-', label=f'Validation {key}')
        
        plt.title(f'{key} vs. Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(key)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # プロットを保存
        save_path = os.path.join(save_dir, f"{filename_prefix}_{key}.png")
        plt.savefig(save_path)
        plt.close()
    
    print(f"損失のプロットを保存しました: {save_dir}")


def save_pose_visualizations(image, pose_2d, pose_3d, save_dir, filename_prefix, epoch=None, index=0):
    """
    ポーズの可視化結果を保存
    
    Args:
        image: 入力画像 [H, W, 3]
        pose_2d: 2Dポーズ [関節数, 2]
        pose_3d: 3Dポーズ [関節数, 3]
        save_dir: 保存先ディレクトリ
        filename_prefix: ファイル名のプレフィックス
        epoch: エポック番号 (Noneの場合は含めない)
        index: バッチ内のインデックス
    """
    # ディレクトリが存在しない場合は作成
    os.makedirs(save_dir, exist_ok=True)
    
    # ファイル名のエポック部分を設定
    epoch_str = f"_epoch{epoch}" if epoch is not None else ""
    
    # 2Dポーズの可視化と保存
    fig_2d = visualize_2d_pose(pose_2d, image)
    save_path_2d = os.path.join(save_dir, f"{filename_prefix}_2d{epoch_str}_{index}.png")
    fig_2d.savefig(save_path_2d)
    plt.close(fig_2d)
    
    # 3Dポーズの可視化と保存
    fig_3d = visualize_3d_pose(pose_3d)
    save_path_3d = os.path.join(save_dir, f"{filename_prefix}_3d{epoch_str}_{index}.png")
    fig_3d.savefig(save_path_3d)
    plt.close(fig_3d)
    
    # 総合的な可視化と保存
    fig_all = visualize_results(image, pose_2d, pose_3d)
    save_path_all = os.path.join(save_dir, f"{filename_prefix}_all{epoch_str}_{index}.png")
    fig_all.savefig(save_path_all)
    plt.close(fig_all)