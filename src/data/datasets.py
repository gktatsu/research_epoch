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
    def __init__(self, data_path=MPI_INF_3DHP_DIR, subjects=None, sequences=None, 
                 split="train", image_size=(224, 224), augment=False):
        """
        Args:
            data_path: MPI-INF-3DHPデータセットのルートディレクトリ
            subjects: 使用するサブジェクト番号のリスト（Noneの場合はsplitによる）
            sequences: 使用するシーケンス番号のリスト（Noneの場合は全て）
            split: データセット分割 ("train" または "test")
            image_size: 画像サイズ (width, height)
            augment: データ拡張を行うかどうか
        """
        self.data_path = data_path
        self.image_size = image_size
        self.augment = augment
        
        # サブジェクトが指定されていなければ、分割に基づいて設定
        if subjects is None:
            if split == "train":
                self.subjects = [1, 2, 3, 4, 5, 6, 7]  # S1-S7は訓練用
            else:
                self.subjects = [8]  # S8はテスト用
        else:
            self.subjects = subjects
        
        # シーケンスが指定されていなければ、すべてのシーケンスを使用
        if sequences is None:
            self.sequences = [1, 2]  # Seq1, Seq2
        else:
            self.sequences = sequences
        
        print(f"データセットの初期化: {data_path}")
        print(f"サブジェクト: {self.subjects}")
        print(f"シーケンス: {self.sequences}")
        
        # データセットの読み込み
        self.data = self._load_data()
        print(f"読み込まれたデータ数: {len(self.data)}")
    
    def _load_data(self):
        """
        アノテーションファイルからデータを読み込み
        
        Returns:
            data: 画像パス、2Dポーズ、3Dポーズのリスト
        """
        data = []
        
        for subject in self.subjects:
            for seq in self.sequences:
                subject_seq_path = os.path.join(self.data_path, f"S{subject}", f"Seq{seq}")
                
                # ディレクトリが存在しない場合はスキップ
                if not os.path.exists(subject_seq_path):
                    print(f"ディレクトリが見つかりません: {subject_seq_path}")
                    continue
                
                # すべてのカメラについて処理
                for cam in range(14):  # cam0 to cam13
                    cam_path = os.path.join(subject_seq_path, f"cam{cam}")
                    
                    # カメラディレクトリが存在しない場合はスキップ
                    if not os.path.exists(cam_path):
                        continue
                    
                    annot_file = os.path.join(cam_path, f"annot_cam{cam}.h5")
                    
                    if not os.path.exists(annot_file):
                        print(f"アノテーションファイルが見つかりません: {annot_file}")
                        continue
                    
                    print(f"アノテーションファイルを読み込み中: {annot_file}")
                    
                    try:
                        with h5py.File(annot_file, 'r') as f:
                            # データセットの構造を確認
                            print(f"H5ファイルのキー: {list(f.keys())}")
                            
                            # 存在するデータセットに基づいて処理を行う
                            if 'poses_2d' in f:
                                pose_2d_data = f['poses_2d']
                                n_frames = pose_2d_data.shape[0]
                                print(f"フレーム数: {n_frames}")
                                
                                # サンプリングレートを下げる（すべてのフレームを使用するとデータが多すぎる）
                                for i in range(0, n_frames, 5):
                                    if i >= n_frames:
                                        break
                                    
                                    # 画像パス（画像ファイル名の形式を確認）
                                    # いくつかの形式を試してみる
                                    img_paths = [
                                        os.path.join(cam_path, "images", f"img_cam{cam}_{i:06d}.jpg"),
                                        os.path.join(cam_path, "images", f"img_{i:06d}.jpg"),
                                        os.path.join(cam_path, "images", f"frame_{i:06d}.jpg"),
                                        os.path.join(cam_path, "images", f"frame{i:06d}.jpg")
                                    ]
                                    
                                    img_path = None
                                    for path in img_paths:
                                        if os.path.exists(path):
                                            img_path = path
                                            break
                                    
                                    # 画像が見つからない場合はスキップ
                                    if img_path is None:
                                        if i == 0:  # 最初のフレームの場合のみメッセージを表示
                                            print(f"画像が見つかりません: {img_paths[0]}")
                                        continue
                                    
                                    # 2Dポーズ [17, 2]
                                    pose_2d = pose_2d_data[i]
                                    
                                    # 3Dポーズ [17, 3]
                                    pose_3d = f['poses_3d'][i] if 'poses_3d' in f else np.zeros((17, 3))
                                    
                                    # カメラパラメータ
                                    # 注: カメラパラメータの構造はデータセットごとに異なる場合がある
                                    if 'camera' in f:
                                        if f['camera'].shape[1] >= 21:  # カメラパラメータが十分な次元を持っている場合
                                            cam_param = {
                                                'R': f['camera'][i, :9].reshape(3, 3),  # 回転行列
                                                't': f['camera'][i, 9:12],               # 平行移動ベクトル
                                                'K': f['camera'][i, 12:21].reshape(3, 3) # カメラ内部パラメータ
                                            }
                                        else:
                                            # 不十分な次元の場合、デフォルト値で補完
                                            cam_param = {
                                                'R': np.eye(3),  # 単位行列
                                                't': np.zeros(3),  # ゼロベクトル
                                                'K': np.array([  # デフォルトの内部パラメータ
                                                    [1000, 0, self.image_size[0]/2],
                                                    [0, 1000, self.image_size[1]/2],
                                                    [0, 0, 1]
                                                ])
                                            }
                                    else:
                                        # カメラパラメータが提供されていない場合のデフォルト値
                                        cam_param = {
                                            'R': np.eye(3),  # 単位行列
                                            't': np.zeros(3),  # ゼロベクトル
                                            'K': np.array([  # デフォルトの内部パラメータ
                                                [1000, 0, self.image_size[0]/2],
                                                [0, 1000, self.image_size[1]/2],
                                                [0, 0, 1]
                                            ])
                                        }
                                    
                                    data.append({
                                        'img_path': img_path,
                                        'pose_2d': pose_2d,
                                        'pose_3d': pose_3d,
                                        'cam_param': cam_param,
                                        'subject': subject,
                                        'sequence': seq,
                                        'camera': cam,
                                        'frame': i
                                    })
                            else:
                                print(f"'poses_2d'データセットが見つかりません: {annot_file}")
                    except Exception as e:
                        print(f"エラー: {annot_file}の読み込み中にエラーが発生しました: {e}")
        
        # データがない場合は警告
        if len(data) == 0:
            print("警告: データが読み込まれませんでした。ファイルパスとデータセット構造を確認してください。")
        
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
        except Exception as e:
            print(f"画像の読み込みエラー: {item['img_path']}: {e}")
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
            'sequence': item['sequence'],
            'camera': item['camera'],
            'frame': item['frame']
        }
    
    # _augment_dataメソッドはそのまま
    
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