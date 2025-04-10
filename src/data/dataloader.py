"""
データローダーの実装
"""
import torch
from torch.utils.data import DataLoader

from .datasets import Human36MDataset, MPIInf3DHPDataset


def get_human36m_loader(data_path, batch_size=32, subjects=None, actions=None, 
                       split="train", image_size=(224, 224), augment=False, 
                       num_workers=4, shuffle=True):
    """
    Human3.6Mデータセットのデータローダーを取得
    
    Args:
        data_path: Human3.6Mデータセットのルートディレクトリ
        batch_size: バッチサイズ
        subjects: 使用する被験者のリスト
        actions: 使用するアクションのリスト
        split: データセット分割 ("train" または "test")
        image_size: 画像サイズ (width, height)
        augment: データ拡張を行うかどうか
        num_workers: データ読み込みに使用するワーカー数
        shuffle: データをシャッフルするかどうか
        
    Returns:
        dataloader: Human3.6Mデータセットのデータローダー
    """
    dataset = Human36MDataset(
        data_path=data_path,
        subjects=subjects,
        actions=actions,
        split=split,
        image_size=image_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_mpiinf3dhp_loader(data_path, batch_size=32, sequences=None, 
                         split="train", image_size=(224, 224), augment=False, 
                         num_workers=4, shuffle=True):
    """
    MPI-INF-3DHPデータセットのデータローダーを取得
    
    Args:
        data_path: MPI-INF-3DHPデータセットのルートディレクトリ
        batch_size: バッチサイズ
        sequences: 使用するシーケンス番号のリスト
        split: データセット分割 ("train" または "test")
        image_size: 画像サイズ (width, height)
        augment: データ拡張を行うかどうか
        num_workers: データ読み込みに使用するワーカー数
        shuffle: データをシャッフルするかどうか
        
    Returns:
        dataloader: MPI-INF-3DHPデータセットのデータローダー
    """
    dataset = MPIInf3DHPDataset(
        data_path=data_path,
        sequences=sequences,
        split=split,
        image_size=image_size,
        augment=augment
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    return dataloader


def get_dataloader(dataset_name, **kwargs):
    """
    指定したデータセットのデータローダーを取得
    
    Args:
        dataset_name: データセット名 ("human36m" または "mpiinf3dhp")
        **kwargs: 各データセットのデータローダー関数に渡す引数
        
    Returns:
        dataloader: 指定したデータセットのデータローダー
    """
    if dataset_name.lower() == "human36m":
        return get_human36m_loader(**kwargs)
    elif dataset_name.lower() == "mpiinf3dhp":
        return get_mpiinf3dhp_loader(**kwargs)
    else:
        raise ValueError(f"未知のデータセット名: {dataset_name}")