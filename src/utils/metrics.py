"""
3Dポーズ推定の評価指標の実装
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation


def calculate_mpjpe(predicted, target):
    """
    平均関節位置誤差 (MPJPE) を計算
    
    Args:
        predicted: 予測3Dポーズ [バッチサイズ, 関節数, 3]
        target: 目標3Dポーズ [バッチサイズ, 関節数, 3]
        
    Returns:
        mpjpe: 平均関節位置誤差 (mm)
    """
    if isinstance(predicted, np.ndarray):
        return np.mean(np.sqrt(np.sum((predicted - target) ** 2, axis=-1)))
    elif isinstance(predicted, torch.Tensor):
        return torch.mean(torch.sqrt(torch.sum((predicted - target) ** 2, dim=-1)))
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")


def procrustes_alignment(X, Y):
    """
    プロクラステス分析を使用して姿勢を整合
    
    Args:
        X: ソース3Dポーズ [関節数, 3]
        Y: ターゲット3Dポーズ [関節数, 3]
        
    Returns:
        X_aligned: Yに整合させたX
    """
    if isinstance(X, torch.Tensor):
        X = X.detach().cpu().numpy()
    if isinstance(Y, torch.Tensor):
        Y = Y.detach().cpu().numpy()
    
    # NaNや無限大の値をチェック
    if np.isnan(X).any() or np.isnan(Y).any() or np.isinf(X).any() or np.isinf(Y).any():
        # 問題がある場合は元のポーズをそのまま返す
        return X.copy()
        
    # 中心を原点に移動
    X_centered = X - X.mean(axis=0)
    Y_centered = Y - Y.mean(axis=0)
    
    # ゼロ除算を防ぐために小さな値を追加
    eps = 1e-7
    
    # 分散をそろえる
    X_scale = np.sqrt(np.sum(X_centered ** 2)) + eps
    Y_scale = np.sqrt(np.sum(Y_centered ** 2)) + eps
    
    X_normalized = X_centered / X_scale
    Y_normalized = Y_centered / Y_scale
    
    # 最適な回転行列を計算
    H = X_normalized.T @ Y_normalized
    
    # SVDの計算で発散を防ぐために条件処理
    try:
        U, S, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        
        # 反射変換を防ぐ
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
            
        # スケール、回転、平行移動を適用
        scale = Y_scale / X_scale
        t = Y.mean(axis=0) - scale * (R @ X.mean(axis=0))
        
        X_aligned = scale * (X @ R.T) + t
    except np.linalg.LinAlgError:
        # SVDが収束しない場合は単純な中心合わせと尺度調整のみを行う
        scale = Y_scale / X_scale
        X_aligned = scale * X_centered + Y.mean(axis=0)
    
    return X_aligned


def calculate_pa_mpjpe(predicted, target):
    """
    プロクラステス分析後の平均関節位置誤差 (PA-MPJPE) を計算
    
    Args:
        predicted: 予測3Dポーズ [バッチサイズ, 関節数, 3]
        target: 目標3Dポーズ [バッチサイズ, 関節数, 3]
        
    Returns:
        pa_mpjpe: プロクラステス分析後の平均関節位置誤差 (mm)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    errors = []
    for i in range(predicted.shape[0]):
        aligned = procrustes_alignment(predicted[i], target[i])
        errors.append(np.mean(np.sqrt(np.sum((aligned - target[i]) ** 2, axis=-1))))
    
    return np.mean(errors)


def calculate_n_mpjpe(predicted, target):
    """
    正規化された平均関節位置誤差 (N-MPJPE) を計算
    
    Args:
        predicted: 予測3Dポーズ [バッチサイズ, 関節数, 3]
        target: 目標3Dポーズ [バッチサイズ, 関節数, 3]
        
    Returns:
        n_mpjpe: 正規化された平均関節位置誤差 (mm)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # NaNや無限大の値をチェック
    if np.isnan(predicted).any() or np.isnan(target).any() or np.isinf(predicted).any() or np.isinf(target).any():
        # 問題がある場合は大きな値を返す
        return 1000.0
    
    # バッチごとに正規化して計算
    errors = []
    for i in range(predicted.shape[0]):
        try:
            # 中心を原点に移動
            pred_centered = predicted[i] - predicted[i].mean(axis=0)
            target_centered = target[i] - target[i].mean(axis=0)
            
            # スケールを合わせる (ゼロ除算を防ぐ)
            eps = 1e-7
            pred_scale = np.sqrt(np.sum(pred_centered ** 2)) + eps
            target_scale = np.sqrt(np.sum(target_centered ** 2)) + eps
            
            pred_normalized = pred_centered * (target_scale / pred_scale)
            
            # 誤差を計算
            error = np.mean(np.sqrt(np.sum((pred_normalized - target_centered) ** 2, axis=-1)))
            errors.append(error)
        except Exception as e:
            # エラーが発生した場合は大きな値を追加
            print(f"N-MPJPE計算中のサンプル{i}でエラー: {e}")
            errors.append(1000.0)
    
    if len(errors) == 0:
        return 1000.0
    
    return np.mean(errors)


def calculate_pck(predicted, target, threshold=150):
    """
    正しいキーポイントの割合 (PCK) を計算 
    - threshold mm以内の予測を正解とみなす
    
    Args:
        predicted: 予測3Dポーズ [バッチサイズ, 関節数, 3]
        target: 目標3Dポーズ [バッチサイズ, 関節数, 3]
        threshold: 正解とみなす距離閾値 (mm)
        
    Returns:
        pck: 正しいキーポイントの割合 (%)
    """
    if isinstance(predicted, torch.Tensor):
        predicted = predicted.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()
    
    # 関節ごとの誤差を計算
    distances = np.sqrt(np.sum((predicted - target) ** 2, axis=-1))
    
    # 閾値以下の関節の割合を計算
    pck = np.mean(distances <= threshold) * 100.0
    
    return pck


def calculate_auc(predicted, target, thresholds=None):
    """
    PCKのAUC (Area Under Curve) を計算
    
    Args:
        predicted: 予測3Dポーズ [バッチサイズ, 関節数, 3]
        target: 目標3Dポーズ [バッチサイズ, 関節数, 3]
        thresholds: PCKの閾値リスト (mm)
        
    Returns:
        auc: PCKのAUC値
    """
    if thresholds is None:
        thresholds = np.linspace(0, 150, 31)
    
    pck_values = []
    for threshold in thresholds:
        pck = calculate_pck(predicted, target, threshold)
        pck_values.append(pck)
    
    # 正規化されたAUCを計算
    auc = np.trapz(pck_values, thresholds) / (thresholds[-1] - thresholds[0])
    auc = auc / 100.0  # PCKは%で表現されているため、正規化
    
    return auc