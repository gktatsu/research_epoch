"""
Human3.6MとMPI-INF-3DHPデータセットの実装
"""
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import h5py
from PIL import Image
import random

from ..config import HUMAN36M_DIR, MPI_INF_3DHP_DIR, DATASET_CONFIG


class Human36MDataset(Dataset):
    """
    Human3.6Mデータセットのカスタムデータセットクラス
    """
    def __init__(self, data_path=HUMAN36M_DIR, subjects=None, actions=None, 
                 split="train", image_size=(224, 224), augment=False):
        """
        Args:
            data_path: Human3.6Mデータセットのルートディレクトリ
            subjects: 使用する被験者のリスト（Noneの場合はsplitによる）
            actions: 使用するアクションのリスト（Noneの場合は全て）
            split: データセット分割 ("train" または "test")
            image_size: 画像サイズ (width, height)
            augment: データ拡張を行うかどうか
        """
        self.data_path = data_path
        self.image_size = image_size
        self.augment = augment
        
        # 被験者が指定されていなければ、分割に基づいて設定
        if subjects is None:
            if split == "train":
                self.subjects = DATASET_CONFIG['human36m']['train_subjects']
            else:
                self.subjects = DATASET_CONFIG['human36m']['test_subjects']
        else:
            self.subjects = subjects
        
        # アクションが指定されていなければ、全てのアクションを使用
        if actions is None:
            self.actions = DATASET_CONFIG['human36m']['actions']
        else:
            self.actions = actions
        
        # データセットの読み込み
        self.data = self._load_data()
    
    def _load_data(self):
        """
        H5ファイルからデータを読み込み
        
        Returns:
            data: 画像パス、2Dポーズ、3Dポーズ、カメラパラメータのリスト
        """
        # 注意: 実際の実装ではデータセットの構造に合わせた読み込みが必要
        # ここでは簡易的な実装を示す
        data = []
        
        for subject in self.subjects:
            subject_path = os.path.join(self.data_path, subject)
            
            for action in self.actions:
                # H5ファイルが存在する場合
                action_file = os.path.join(subject_path, f"{action}.h5")
                if os.path.exists(action_file):
                    with h5py.File(action_file, 'r') as f:
                        # フレーム数
                        n_frames = f['poses_2d'].shape[0]
                        
                        for i in range(n_frames):
                            # 画像パス
                            img_path = os.path.join(subject_path, 'images', f"{action}_{i:06d}.jpg")
                            
                            # 2Dポーズ [17, 2]
                            pose_2d = f['poses_2d'][i]
                            
                            # 3Dポーズ [17, 3]
                            pose_3d = f['poses_3d'][i]
                            
                            # カメラパラメータ
                            cam_param = {
                                'R': f['camera'][i, :9].reshape(3, 3),  # 回転行列
                                't': f['camera'][i, 9:12],               # 平行移動ベクトル
                                'K': f['camera'][i, 12:21].reshape(3, 3) # カメラ内部パラメータ
                            }
                            
                            data.append({
                                'img_path': img_path,
                                'pose_2d': pose_2d,
                                'pose_3d': pose_3d,
                                'cam_param': cam_param,
                                'subject': subject,
                                'action': action,
                                'frame': i
                            })
        
        return data
    
    def __len__(self):
        """
        データセットの長さを返す
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        指定したインデックスのデータを返す
        
        Args:
            idx: データインデックス
            
        Returns:
            sample: 画像、2Dポーズ、3Dポーズ、カメラパラメータを含む辞書
        """
        item = self.data[idx]
        
        # 画像を読み込む
        try:
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            # 画像が見つからない場合は黒画像を返す
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # 画像のリサイズ
        img = cv2.resize(img, self.image_size)
        
        # データ拡張（必要な場合）
        if self.augment:
            img, pose_2d, pose_3d = self._augment_data(img, item['pose_2d'], item['pose_3d'])
        else:
            pose_2d = item['pose_2d'].copy()
            pose_3d = item['pose_3d'].copy()
        
        # カメラパラメータ
        cam_param = item['cam_param'].copy()
        
        # NumPy配列からTensorに変換
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # [3, H, W]
        pose_2d_tensor = torch.from_numpy(pose_2d).float()  # [17, 2]
        pose_3d_tensor = torch.from_numpy(pose_3d).float()  # [17, 3]
        
        # カメラパラメータをTensorに変換
        cam_R = torch.from_numpy(cam_param['R']).float()  # [3, 3]
        cam_t = torch.from_numpy(cam_param['t']).float()  # [3]
        cam_K = torch.from_numpy(cam_param['K']).float()  # [3, 3]
        
        return {
            'image': img_tensor,
            'pose_2d': pose_2d_tensor,
            'pose_3d': pose_3d_tensor,
            'cam_R': cam_R,
            'cam_t': cam_t,
            'cam_K': cam_K,
            'subject': item['subject'],
            'action': item['action'],
            'frame': item['frame']
        }
    
    def _augment_data(self, img, pose_2d, pose_3d):
        """
        データ拡張を行う
        
        Args:
            img: 入力画像 [H, W, 3]
            pose_2d: 2Dポーズ [17, 2]
            pose_3d: 3Dポーズ [17, 3]
            
        Returns:
            img_aug: 拡張された画像
            pose_2d_aug: 拡張された2Dポーズ
            pose_3d_aug: 拡張された3Dポーズ
        """
        # 水平方向の反転（50%の確率で）
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # 水平方向に反転
            h, w = img.shape[:2]
            
            # 2Dポーズの反転
            pose_2d = pose_2d.copy()
            pose_2d[:, 0] = w - pose_2d[:, 0]  # x座標を反転
            
            # 左右の関節を入れ替え
            # 脚: 左(4,5,6) <-> 右(7,8,9)
            # 腕: 左(10,11,12) <-> 右(13,14,15)
            pairs = [(4, 7), (5, 8), (6, 9), (10, 13), (11, 14), (12, 15)]
            for i, j in pairs:
                pose_2d[[i, j]] = pose_2d[[j, i]]
                pose_3d[[i, j]] = pose_3d[[j, i]]
            
            # 3Dポーズの反転（x軸方向）
            pose_3d = pose_3d.copy()
            pose_3d[:, 0] = -pose_3d[:, 0]  # x座標を反転
        
        # 明るさ、コントラスト、彩度のランダム変更
        img = img.astype(np.float32) / 255.0
        
        # 明るさ: ±50%
        brightness = np.random.uniform(0.5, 1.5)
        img = img * brightness
        
        # コントラスト: ±50%
        contrast = np.random.uniform(0.5, 1.5)
        img = (img - 0.5) * contrast + 0.5
        
        # 彩度: ±50%
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = np.random.uniform(0.5, 1.5)
        img[:, :, 1] = img[:, :, 1] * saturation
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
        # 値の範囲を[0, 1]に制限
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        return img, pose_2d, pose_3d


class MPIInf3DHPDataset(Dataset):
    """
    MPI-INF-3DHPデータセットのカスタムデータセットクラス
    """
    def __init__(self, data_path=MPI_INF_3DHP_DIR, subjects=None, 
                 split="train", image_size=(224, 224), augment=False):
        """
        Args:
            data_path: MPI-INF-3DHPデータセットのルートディレクトリ
            subjects: 使用するサブジェクト番号のリスト（Noneの場合はsplitによる）
            split: データセット分割 ("train" または "test")
            image_size: 画像サイズ (width, height)
            augment: データ拡張を行うかどうか
        """
        self.data_path = data_path
        self.image_size = image_size
        self.augment = augment
        
        # シーケンスが指定されていなければ、分割に基づいて設定
        if subjects is None:
            if split == "train":
                self.subjects = DATASET_CONFIG['mpi_inf_3dhp']['train_subjects']
            else:
                self.subjects = DATASET_CONFIG['mpi_inf_3dhp']['test_subjects']
        else:
            self.subjects = subjects
        
        # データセットの読み込み
        self.data = self._load_data()
    
    def _load_data(self):
        """
        H5ファイルからデータを読み込み
        
        Returns:
            data: 画像パス、2Dポーズ、3Dポーズのリスト
        """
        # 注意: 実際の実装ではデータセットの構造に合わせた読み込みが必要
        # ここでは簡易的な実装を示す
        data = []
        print("data_path:", self.data_path)
        for subject in self.subjects:
            subject_path = os.path.join(self.data_path, f"S{subject}")
            seqs = os.listdir(subject_path)
            # シーケンスのリスト
            sequences = [seq for seq in seqs if os.path.isdir(os.path.join(subject_path, seq))]
            print("Subject:", subject_path,",Sequences:", sequences)
            
            for seq in sequences:
                seq_path = os.path.join(subject_path, f"Seq{seq}")
                print("Sequence:", seq)
                # 各カメラの視点について
                for cam in range(1, 15):  # MPI-INF-3DHPデータセットには通常14台のカメラがある
                    # H5ファイルが存在する場合
                    ann_file = os.path.join(seq_path, f"annot_cam{cam}.h5")
                    if os.path.exists(ann_file):
                        with h5py.File(ann_file, 'r') as f:
                            # フレーム数
                            n_frames = f['poses_2d'].shape[0]
                            
                            for i in range(0, n_frames, 5):  # サンプリングレートを下げる
                                # 画像パス
                                img_path = os.path.join(seq_path, f"images/img_cam{cam}_{i:06d}.jpg")
                                
                                # 2Dポーズ [17, 2]
                                pose_2d = f['poses_2d'][i]
                                
                                # 3Dポーズ [17, 3]
                                pose_3d = f['poses_3d'][i]
                                
                                # カメラパラメータ（MPI-INF-3DHPではカメラパラメータも提供されている）
                                cam_param = {
                                    'R': f['camera'][i, :9].reshape(3, 3),  # 回転行列
                                    't': f['camera'][i, 9:12],               # 平行移動ベクトル
                                    'K': f['camera'][i, 12:21].reshape(3, 3) # カメラ内部パラメータ
                                }
                                
                                data.append({
                                    'img_path': img_path,
                                    'pose_2d': pose_2d,
                                    'pose_3d': pose_3d,
                                    'cam_param': cam_param,
                                    'sequence': seq,
                                    'camera': cam,
                                    'frame': i
                                })
        print("Loaded data:", len(data), "samples")
        return data
    
    def __len__(self):
        """
        データセットの長さを返す
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        指定したインデックスのデータを返す
        
        Args:
            idx: データインデックス
            
        Returns:
            sample: 画像、2Dポーズ、3Dポーズ、カメラパラメータを含む辞書
        """
        item = self.data[idx]
        
        # 画像を読み込む
        try:
            img = cv2.imread(item['img_path'])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except:
            # 画像が見つからない場合は黒画像を返す
            img = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
        
        # 画像のリサイズ
        img = cv2.resize(img, self.image_size)
        
        # データ拡張（必要な場合）
        if self.augment:
            img, pose_2d, pose_3d = self._augment_data(img, item['pose_2d'], item['pose_3d'])
        else:
            pose_2d = item['pose_2d'].copy()
            pose_3d = item['pose_3d'].copy()
        
        # カメラパラメータ
        cam_param = item['cam_param'].copy()
        
        # NumPy配列からTensorに変換
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0  # [3, H, W]
        pose_2d_tensor = torch.from_numpy(pose_2d).float()  # [17, 2]
        pose_3d_tensor = torch.from_numpy(pose_3d).float()  # [17, 3]
        
        # カメラパラメータをTensorに変換
        cam_R = torch.from_numpy(cam_param['R']).float()  # [3, 3]
        cam_t = torch.from_numpy(cam_param['t']).float()  # [3]
        cam_K = torch.from_numpy(cam_param['K']).float()  # [3, 3]
        
        return {
            'image': img_tensor,
            'pose_2d': pose_2d_tensor,
            'pose_3d': pose_3d_tensor,
            'cam_R': cam_R,
            'cam_t': cam_t,
            'cam_K': cam_K,
            'sequence': item['sequence'],
            'camera': item['camera'],
            'frame': item['frame']
        }
    
    def _augment_data(self, img, pose_2d, pose_3d):
        """
        データ拡張を行う
        
        Args:
            img: 入力画像 [H, W, 3]
            pose_2d: 2Dポーズ [17, 2]
            pose_3d: 3Dポーズ [17, 3]
            
        Returns:
            img_aug: 拡張された画像
            pose_2d_aug: 拡張された2Dポーズ
            pose_3d_aug: 拡張された3Dポーズ
        """
        # Human36MDatasetと同様のデータ拡張を行う
        # 水平方向の反転（50%の確率で）
        if random.random() > 0.5:
            img = cv2.flip(img, 1)  # 水平方向に反転
            h, w = img.shape[:2]
            
            # 2Dポーズの反転
            pose_2d = pose_2d.copy()
            pose_2d[:, 0] = w - pose_2d[:, 0]  # x座標を反転
            
            # 左右の関節を入れ替え
            # 脚: 左(4,5,6) <-> 右(7,8,9)
            # 腕: 左(10,11,12) <-> 右(13,14,15)
            pairs = [(4, 7), (5, 8), (6, 9), (10, 13), (11, 14), (12, 15)]
            for i, j in pairs:
                pose_2d[[i, j]] = pose_2d[[j, i]]
                pose_3d[[i, j]] = pose_3d[[j, i]]
            
            # 3Dポーズの反転（x軸方向）
            pose_3d = pose_3d.copy()
            pose_3d[:, 0] = -pose_3d[:, 0]  # x座標を反転
        
        # 明るさ、コントラスト、彩度のランダム変更
        img = img.astype(np.float32) / 255.0
        
        # 明るさ: ±50%
        brightness = np.random.uniform(0.5, 1.5)
        img = img * brightness
        
        # コントラスト: ±50%
        contrast = np.random.uniform(0.5, 1.5)
        img = (img - 0.5) * contrast + 0.5
        
        # 彩度: ±50%
        img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        saturation = np.random.uniform(0.5, 1.5)
        img[:, :, 1] = img[:, :, 1] * saturation
        img = cv2.cvtColor(img, cv2.COLOR_HSV2RGB)
        
        # 値の範囲を[0, 1]に制限
        img = np.clip(img, 0, 1)
        img = (img * 255).astype(np.uint8)
        
        return img, pose_2d, pose_3d