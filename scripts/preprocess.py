"""
Human3.6MとMPI-INF-3DHPデータセットの前処理スクリプト
"""
import os
import sys
import argparse
import numpy as np
import cv2
import h5py
from tqdm import tqdm
import torch

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HUMAN36M_DIR, MPI_INF_3DHP_DIR, DATASET_CONFIG


def preprocess_human36m(output_dir=None, image_size=(224, 224), debug=False):
    """
    Human3.6Mデータセットを前処理
    
    Args:
        output_dir: 出力ディレクトリ
        image_size: 処理後の画像サイズ
        debug: デバッグモード（少量のデータのみ処理）
    """
    if output_dir is None:
        output_dir = HUMAN36M_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Human3.6Mデータセットを前処理しています...")
    print(f"入力ディレクトリ: {HUMAN36M_DIR}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # 被験者とアクションのリスト
    subjects = DATASET_CONFIG['human36m']['train_subjects'] + DATASET_CONFIG['human36m']['test_subjects']
    actions = DATASET_CONFIG['human36m']['actions']
    
    # 各被験者とアクションについて処理
    for subject in subjects:
        subject_dir = os.path.join(HUMAN36M_DIR, subject)
        subject_out_dir = os.path.join(output_dir, subject)
        os.makedirs(subject_out_dir, exist_ok=True)
        
        # 画像の出力ディレクトリ
        images_out_dir = os.path.join(subject_out_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        
        for action in actions:
            print(f"処理中: {subject} - {action}")
            
            # アクションデータのパス
            action_file = os.path.join(subject_dir, f"{action}.h5")
            if not os.path.exists(action_file):
                print(f"ファイルが見つかりません: {action_file}")
                continue
            
            # H5ファイルを読み込む
            with h5py.File(action_file, 'r') as f:
                # データの形状を取得
                n_frames = f['poses_3d'].shape[0]
                
                # デバッグモードの場合は少量のデータのみ処理
                if debug:
                    n_frames = min(n_frames, 10)
                
                # 前処理結果を格納するH5ファイル
                out_file = os.path.join(subject_out_dir, f"{action}.h5")
                with h5py.File(out_file, 'w') as out_f:
                    # データセットを作成
                    out_f.create_dataset('poses_2d', shape=(n_frames, 17, 2), dtype=np.float32)
                    out_f.create_dataset('poses_3d', shape=(n_frames, 17, 3), dtype=np.float32)
                    out_f.create_dataset('camera', shape=(n_frames, 21), dtype=np.float32)
                    
                    # 各フレームを処理
                    for i in tqdm(range(n_frames), desc=f"{subject} - {action}"):
                        # 画像を読み込み（存在する場合）
                        img_path = os.path.join(subject_dir, 'images', f"{action}_{i:06d}.jpg")
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                # 画像をリサイズ
                                img_resized = cv2.resize(img, image_size)
                                # 保存
                                out_img_path = os.path.join(images_out_dir, f"{action}_{i:06d}.jpg")
                                cv2.imwrite(out_img_path, img_resized)
                        
                        # 2Dポーズデータ
                        pose_2d = f['poses_2d'][i]
                        
                        # 3Dポーズデータ
                        pose_3d = f['poses_3d'][i]
                        
                        # カメラパラメータ
                        # カメラ行列、回転行列、平行移動ベクトルの形式で保存
                        camera_data = np.zeros(21, dtype=np.float32)
                        camera_data[:9] = f['camera'][i, :9].reshape(-1)  # 回転行列 (3x3)
                        camera_data[9:12] = f['camera'][i, 9:12]          # 平行移動ベクトル (3)
                        camera_data[12:21] = f['camera'][i, 12:21].reshape(-1)  # カメラ行列 (3x3)
                        
                        # データを保存
                        out_f['poses_2d'][i] = pose_2d
                        out_f['poses_3d'][i] = pose_3d
                        out_f['camera'][i] = camera_data
    
    print(f"Human3.6Mデータセットの前処理が完了しました!")


def preprocess_mpiinf3dhp(output_dir=None, image_size=(224, 224), debug=False):
    """
    MPI-INF-3DHPデータセットを前処理
    
    Args:
        output_dir: 出力ディレクトリ
        image_size: 処理後の画像サイズ
        debug: デバッグモード（少量のデータのみ処理）
    """
    if output_dir is None:
        output_dir = MPI_INF_3DHP_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"MPI-INF-3DHPデータセットを前処理しています...")
    print(f"入力ディレクトリ: {MPI_INF_3DHP_DIR}")
    print(f"出力ディレクトリ: {output_dir}")
    
    # 訓練用と評価用のシーケンス
    train_sequences = DATASET_CONFIG['mpi_inf_3dhp']['train_sequences']
    test_sequences = DATASET_CONFIG['mpi_inf_3dhp']['test_sequences']
    sequences = train_sequences + test_sequences
    
    # 各シーケンスについて処理
    for seq in sequences:
        seq_dir = os.path.join(MPI_INF_3DHP_DIR, f"S{seq}")
        seq_out_dir = os.path.join(output_dir, f"S{seq}")
        os.makedirs(seq_out_dir, exist_ok=True)
        
        # 画像の出力ディレクトリ
        images_out_dir = os.path.join(seq_out_dir, 'images')
        os.makedirs(images_out_dir, exist_ok=True)
        
        print(f"処理中: シーケンス {seq}")
        
        # 各カメラビューについて処理
        for cam in range(1, 15):  # 通常14台のカメラがある
            # アノテーションファイルのパス
            annot_file = os.path.join(seq_dir, f"annot_cam{cam}.h5")
            if not os.path.exists(annot_file):
                print(f"ファイルが見つかりません: {annot_file}")
                continue
            
            # H5ファイルを読み込む
            with h5py.File(annot_file, 'r') as f:
                # データの形状を取得
                if 'poses_3d' not in f:
                    print(f"'poses_3d'データが見つかりません: {annot_file}")
                    continue
                
                n_frames = f['poses_3d'].shape[0]
                
                # デバッグモードの場合は少量のデータのみ処理
                if debug:
                    n_frames = min(n_frames, 10)
                
                # 前処理結果を格納するH5ファイル
                out_file = os.path.join(seq_out_dir, f"annot_cam{cam}.h5")
                with h5py.File(out_file, 'w') as out_f:
                    # データセットを作成
                    out_f.create_dataset('poses_2d', shape=(n_frames, 17, 2), dtype=np.float32)
                    out_f.create_dataset('poses_3d', shape=(n_frames, 17, 3), dtype=np.float32)
                    out_f.create_dataset('camera', shape=(n_frames, 21), dtype=np.float32)
                    
                    # 各フレームを処理
                    for i in tqdm(range(0, n_frames, 5), desc=f"シーケンス {seq} - カメラ {cam}"):  # サンプリングレート低減
                        if i >= n_frames:
                            break
                            
                        # 画像を読み込み（存在する場合）
                        img_path = os.path.join(seq_dir, 'images', f"img_cam{cam}_{i:06d}.jpg")
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path)
                            if img is not None:
                                # 画像をリサイズ
                                img_resized = cv2.resize(img, image_size)
                                # 保存
                                out_img_path = os.path.join(images_out_dir, f"img_cam{cam}_{i:06d}.jpg")
                                cv2.imwrite(out_img_path, img_resized)
                        
                        # 2Dポーズデータ（存在する場合）
                        if 'poses_2d' in f:
                            pose_2d = f['poses_2d'][i]
                            # MPII形式に変換（必要な場合）
                            # ここでは単純化のため、そのまま使用
                        else:
                            # 3Dポーズから2Dポーズを計算
                            pose_3d = f['poses_3d'][i]
                            camera_params = f['camera'][i] if 'camera' in f else None
                            
                            if camera_params is not None:
                                # カメラパラメータを使用して3Dから2Dに投影
                                # 単純な直交投影を例として使用
                                pose_2d = pose_3d[:, :2]
                            else:
                                # カメラパラメータがない場合は単純な直交投影
                                pose_2d = pose_3d[:, :2]
                        
                        # 3Dポーズデータ
                        pose_3d = f['poses_3d'][i]
                        
                        # カメラパラメータ（存在する場合）
                        if 'camera' in f:
                            camera_data = f['camera'][i]
                        else:
                            # カメラパラメータがない場合はデフォルト値
                            camera_data = np.zeros(21, dtype=np.float32)
                            # 回転行列を単位行列に
                            camera_data[:9] = np.eye(3).reshape(-1)
                            # 内部パラメータを適当な値に設定
                            camera_data[12:21] = np.array([
                                1000, 0, image_size[0]/2,  # fx, 0, cx
                                0, 1000, image_size[1]/2,  # 0, fy, cy
                                0, 0, 1                   # 0, 0, 1
                            ])
                        
                        # データを保存
                        out_f['poses_2d'][i//5] = pose_2d
                        out_f['poses_3d'][i//5] = pose_3d
                        out_f['camera'][i//5] = camera_data
    
    print(f"MPI-INF-3DHPデータセットの前処理が完了しました!")


def main():
    parser = argparse.ArgumentParser(description='Human3.6MとMPI-INF-3DHPデータセットの前処理')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp', 'all'], default='all',
                        help='前処理するデータセット')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='出力ディレクトリ')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='処理後の画像サイズ')
    parser.add_argument('--debug', action='store_true',
                        help='デバッグモード（少量のデータのみ処理）')
    
    args = parser.parse_args()
    
    if args.dataset in ['human36m', 'all']:
        preprocess_human36m(args.output_dir, tuple(args.image_size), args.debug)
    
    if args.dataset in ['mpiinf3dhp', 'all']:
        preprocess_mpiinf3dhp(args.output_dir, tuple(args.image_size), args.debug)


if __name__ == '__main__':
    main()