"""
MATファイルをH5形式に変換するスクリプト

Human3.6MやMPI-INF-3DHPデータセットのMAT形式のデータをH5形式に変換
"""
import os
import sys
import argparse
import numpy as np
import h5py
from scipy.io import loadmat
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import HUMAN36M_DIR, MPI_INF_3DHP_DIR


def convert_mat_to_h5(input_path, output_path, dataset_type='human36m'):
    """
    MATファイルをH5形式に変換

    Args:
        input_path: 入力MATファイルのパス
        output_path: 出力H5ファイルのパス
        dataset_type: データセットの種類 ('human36m' または 'mpiinf3dhp')
    """
    print(f"変換中: {input_path} -> {output_path}")
    
    try:
        # MATファイルを読み込む
        if h5py.is_hdf5(input_path):
            # v7.3 MAT形式の場合はh5pyで直接読み込む
            mat_data = h5py.File(input_path, 'r')
            is_v73 = True
        else:
            # 古いMAT形式の場合はscipy.ioで読み込む
            mat_data = loadmat(input_path)
            is_v73 = False
        
        # 出力用のH5ファイルを作成
        with h5py.File(output_path, 'w') as h5_file:
            # データセットの種類に応じて変換処理を実行
            if dataset_type == 'human36m':
                _convert_human36m(mat_data, h5_file, is_v73)
            elif dataset_type == 'mpiinf3dhp':
                _convert_mpiinf3dhp(mat_data, h5_file, is_v73)
            else:
                raise ValueError(f"未知のデータセット種類: {dataset_type}")
        
        print(f"変換完了: {output_path}")
        
    except Exception as e:
        print(f"変換エラー: {e}")
        if os.path.exists(output_path):
            os.remove(output_path)


def _convert_human36m(mat_data, h5_file, is_v73):
    """
    Human3.6MデータセットのMAT形式からH5形式への変換処理

    Args:
        mat_data: MATファイルのデータ
        h5_file: 出力H5ファイル
        is_v73: v7.3形式のMATファイルかどうか
    """
    if is_v73:
        # v7.3 MAT形式の場合の変換処理
        # Human3.6MのMAT構造に合わせて処理
        
        # 必要なデータを取得
        if 'poses_3d' in mat_data:
            poses_3d = np.array(mat_data['poses_3d'])
            # v7.3形式では転置が必要な場合がある
            if poses_3d.shape[0] < poses_3d.shape[1]:
                poses_3d = poses_3d.T
            h5_file.create_dataset('poses_3d', data=poses_3d)
        
        if 'poses_2d' in mat_data:
            poses_2d = np.array(mat_data['poses_2d'])
            if poses_2d.shape[0] < poses_2d.shape[1]:
                poses_2d = poses_2d.T
            h5_file.create_dataset('poses_2d', data=poses_2d)
        
        if 'camera' in mat_data:
            camera = np.array(mat_data['camera'])
            if camera.shape[0] < camera.shape[1]:
                camera = camera.T
            h5_file.create_dataset('camera', data=camera)
        
        # その他のメタデータがあれば追加
        if 'frame_rate' in mat_data:
            h5_file.create_dataset('frame_rate', data=np.array(mat_data['frame_rate']))
        
        if 'subject' in mat_data:
            # 文字列の場合はエンコーディングに注意
            subject = np.array(mat_data['subject'])
            h5_file.create_dataset('subject', data=subject)
    
    else:
        # 古いMAT形式の場合の変換処理
        # 主要なデータを取得して保存
        for key in ['poses_3d', 'poses_2d', 'camera']:
            if key in mat_data:
                h5_file.create_dataset(key, data=mat_data[key])
        
        # その他のメタデータを保存
        for key in ['frame_rate', 'subject', 'action']:
            if key in mat_data:
                h5_file.create_dataset(key, data=mat_data[key])


def _convert_mpiinf3dhp(mat_data, h5_file, is_v73):
    """
    MPI-INF-3DHPデータセットのMAT形式からH5形式への変換処理

    Args:
        mat_data: MATファイルのデータ
        h5_file: 出力H5ファイル
        is_v73: v7.3形式のMATファイルかどうか
    """
    if is_v73:
        # v7.3 MAT形式の場合の変換処理
        # MPI-INF-3DHPのMAT構造に合わせて処理
        
        # MPI-INF-3DHPの場合、'annot'フィールド内にデータが格納されている場合がある
        if 'annot' in mat_data:
            annot = mat_data['annot']
            
            # 3Dポーズデータ
            if 'univ_annot3' in annot:
                poses_3d = np.array(annot['univ_annot3'])
                # 形状を整形（必要な場合）
                if len(poses_3d.shape) > 2:
                    poses_3d = poses_3d.reshape(poses_3d.shape[0], -1, 3)
                h5_file.create_dataset('poses_3d', data=poses_3d)
            
            # 2Dポーズデータ
            if 'annot2' in annot:
                poses_2d = np.array(annot['annot2'])
                # 形状を整形（必要な場合）
                if len(poses_2d.shape) > 2:
                    poses_2d = poses_2d.reshape(poses_2d.shape[0], -1, 2)
                h5_file.create_dataset('poses_2d', data=poses_2d)
            
            # カメラデータ
            if 'cam' in annot:
                camera_data = np.array(annot['cam'])
                # カメラパラメータの形式変換（K, R, tを統合）
                camera_params = _process_camera_params(camera_data)
                h5_file.create_dataset('camera', data=camera_params)
        
        else:
            # データが直接ルートに格納されている場合
            if 'poses_3d' in mat_data:
                poses_3d = np.array(mat_data['poses_3d'])
                h5_file.create_dataset('poses_3d', data=poses_3d)
            
            if 'poses_2d' in mat_data:
                poses_2d = np.array(mat_data['poses_2d'])
                h5_file.create_dataset('poses_2d', data=poses_2d)
            
            if 'camera' in mat_data:
                camera = np.array(mat_data['camera'])
                h5_file.create_dataset('camera', data=camera)
    
    else:
        # 古いMAT形式の場合の変換処理
        # MPI-INF-3DHPの古いMAT形式のデータ構造に合わせて処理
        if 'annot' in mat_data:
            annot = mat_data['annot']
            
            # 3Dポーズデータ
            if 'univ_annot3' in annot:
                h5_file.create_dataset('poses_3d', data=annot['univ_annot3'])
            elif 'annot3' in annot:
                h5_file.create_dataset('poses_3d', data=annot['annot3'])
            
            # 2Dポーズデータ
            if 'annot2' in annot:
                h5_file.create_dataset('poses_2d', data=annot['annot2'])
            
            # カメラデータ
            if 'cam' in annot:
                camera_params = _process_camera_params(annot['cam'])
                h5_file.create_dataset('camera', data=camera_params)
        
        else:
            # 直接ルートに格納されている場合
            for key in ['poses_3d', 'poses_2d', 'camera']:
                if key in mat_data:
                    h5_file.create_dataset(key, data=mat_data[key])


def _process_camera_params(camera_data):
    """
    カメラパラメータを処理して統一形式に変換

    Args:
        camera_data: 元のカメラパラメータデータ

    Returns:
        camera_params: 統一形式のカメラパラメータ [回転行列(9) + 平行移動ベクトル(3) + カメラ行列(9)]
    """
    # カメラパラメータの構造は、データセットによって異なる場合がある
    # ここでは一般的な処理を実装し、特定のケースに合わせて調整する
    
    # カメラデータの形状を確認
    if isinstance(camera_data, dict):
        # 辞書形式の場合
        K = camera_data.get('K', np.eye(3))
        R = camera_data.get('R', np.eye(3))
        t = camera_data.get('t', np.zeros(3))
    elif isinstance(camera_data, np.ndarray):
        # 配列形式の場合
        if camera_data.shape[-1] == 21:
            # すでに統一形式になっている場合
            return camera_data
        elif camera_data.shape[-1] == 3 and camera_data.shape[-2] == 4:
            # [3, 4]の射影行列の場合
            P = camera_data
            # 射影行列からKとRとtを分解（簡易版）
            K = np.eye(3)
            R = P[:, :3]
            t = P[:, 3]
        else:
            # その他の形式の場合はそのまま返す
            return camera_data
    else:
        # その他の形式の場合は単位行列で初期化
        K = np.eye(3)
        R = np.eye(3)
        t = np.zeros(3)
    
    # 統一形式に変換 [R(9) + t(3) + K(9)]
    camera_params = np.concatenate([
        R.flatten(),
        t.flatten(),
        K.flatten()
    ])
    
    return camera_params


def convert_directory(input_dir, output_dir, dataset_type, recursive=False):
    """
    ディレクトリ内のすべてのMATファイルをH5形式に変換

    Args:
        input_dir: 入力ディレクトリ
        output_dir: 出力ディレクトリ
        dataset_type: データセットの種類 ('human36m' または 'mpiinf3dhp')
        recursive: サブディレクトリも再帰的に処理するかどうか
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # ディレクトリ内のファイルを取得
    if recursive:
        # 再帰的にすべてのファイルを取得
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith('.mat'):
                    # 入力ファイルの相対パスを取得
                    rel_path = os.path.relpath(os.path.join(root, file), input_dir)
                    
                    # 出力ファイルのパスを構築
                    output_subdir = os.path.dirname(os.path.join(output_dir, rel_path))
                    os.makedirs(output_subdir, exist_ok=True)
                    
                    # 拡張子を.h5に変更
                    output_file = os.path.splitext(os.path.join(output_dir, rel_path))[0] + '.h5'
                    
                    # 変換を実行
                    convert_mat_to_h5(
                        os.path.join(root, file),
                        output_file,
                        dataset_type
                    )
    else:
        # 現在のディレクトリのみ処理
        for file in os.listdir(input_dir):
            if file.endswith('.mat'):
                # 拡張子を.h5に変更
                output_file = os.path.splitext(os.path.join(output_dir, file))[0] + '.h5'
                
                # 変換を実行
                convert_mat_to_h5(
                    os.path.join(input_dir, file),
                    output_file,
                    dataset_type
                )


def main():
    parser = argparse.ArgumentParser(description='MATファイルをH5形式に変換')
    parser.add_argument('--input', type=str, required=True,
                        help='入力MATファイルまたはディレクトリのパス')
    parser.add_argument('--output', type=str, required=True,
                        help='出力H5ファイルまたはディレクトリのパス')
    parser.add_argument('--dataset', type=str, choices=['human36m', 'mpiinf3dhp'], default='human36m',
                        help='データセットの種類')
    parser.add_argument('--recursive', action='store_true',
                        help='ディレクトリを再帰的に処理するかどうか')
    
    args = parser.parse_args()
    
    # 入力がディレクトリかファイルかを判断
    if os.path.isdir(args.input):
        # ディレクトリの場合
        convert_directory(args.input, args.output, args.dataset, args.recursive)
    else:
        # 単一ファイルの場合
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        convert_mat_to_h5(args.input, args.output, args.dataset)


if __name__ == '__main__':
    main()