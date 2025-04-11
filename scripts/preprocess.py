"""
MPI-INF-3DHPデータセットの前処理スクリプト
"""
import os
import sys
import argparse
import numpy as np
import cv2
import h5py
import scipy.io as sio
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def preprocess_mpiinf3dhp(dataset_path, output_path, image_size=(224, 224), debug=False):
    """
    MPI-INF-3DHPデータセットを前処理し、各カメラを独立したデータとして扱います
    
    Args:
        dataset_path: 元のデータセットのパス
        output_path: 出力ディレクトリのパス
        image_size: 処理後の画像サイズ (width, height)
        debug: デバッグモード（少量のデータのみ処理）
    """
    os.makedirs(output_path, exist_ok=True)
    
    print(f"MPI-INF-3DHPデータセットを前処理しています...")
    print(f"入力ディレクトリ: {dataset_path}")
    print(f"出力ディレクトリ: {output_path}")
    
    # 被験者とシーケンスのリスト
    subjects = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
    
    for subject in subjects:
        subject_path = os.path.join(dataset_path, subject)
        if not os.path.exists(subject_path):
            print(f"被験者ディレクトリが見つかりません: {subject_path}")
            continue
        
        # 被験者の出力ディレクトリを作成
        subject_out_path = os.path.join(output_path, subject)
        os.makedirs(subject_out_path, exist_ok=True)
        
        # 各シーケンスについて処理
        sequences = [d for d in os.listdir(subject_path) if d.startswith('Seq') and os.path.isdir(os.path.join(subject_path, d))]
        
        for seq in sequences:
            seq_path = os.path.join(subject_path, seq)
            seq_out_path = os.path.join(subject_out_path, seq)
            os.makedirs(seq_out_path, exist_ok=True)
            
            print(f"処理中: {subject}/{seq}")
            
            # アノテーションファイルを読み込む
            annot_file = os.path.join(seq_path, 'annot.mat')
            if not os.path.exists(annot_file):
                print(f"アノテーションファイルが見つかりません: {annot_file}")
                continue
            
            # カメラキャリブレーションファイルを読み込む
            calib_file = os.path.join(seq_path, 'camera.calibration')
            if not os.path.exists(calib_file):
                print(f"キャリブレーションファイルが見つかりません: {calib_file}")
                continue
            
            try:
                # アノテーションを読み込む
                annot_data = sio.loadmat(annot_file)
                
                # カメラキャリブレーションを解析
                camera_params = parse_camera_calibration(calib_file)
            except Exception as e:
                print(f"ファイル読み込みエラー: {e}")
                continue
            
            # 各カメラについて処理
            num_cameras = 14  # MPI-INF-3DHPには通常14台のカメラがある
            
            for cam_idx in range(num_cameras):
                # ビデオファイルをチェック
                video_file = os.path.join(seq_path, 'imageSequence', f'video_{cam_idx}.avi')
                if not os.path.exists(video_file):
                    print(f"  カメラ{cam_idx}のビデオファイルが見つかりません: {video_file}")
                    continue
                
                # カメラの出力ディレクトリを作成
                cam_out_path = os.path.join(seq_out_path, f'cam{cam_idx}')
                os.makedirs(cam_out_path, exist_ok=True)
                images_out_path = os.path.join(cam_out_path, 'images')
                os.makedirs(images_out_path, exist_ok=True)
                
                print(f"  カメラ{cam_idx}を処理中...")
                
                # カメラのアノテーションデータを取得
                poses_3d = extract_3d_poses(annot_data, cam_idx)
                poses_2d = extract_2d_poses(annot_data, cam_idx)
                
                if poses_3d is None or poses_2d is None:
                    print(f"  カメラ{cam_idx}のアノテーションデータがありません")
                    continue
                
                # カメラパラメータを取得
                camera_param = camera_params.get(f"{cam_idx}", None)
                if camera_param is None:
                    print(f"  カメラ{cam_idx}のキャリブレーションデータがありません")
                    continue
                
                # 利用可能なフレーム数を確認
                n_frames = min(poses_3d.shape[0], poses_2d.shape[0])
                
                # デバッグモードの場合はフレーム数を制限
                if debug:
                    n_frames = min(n_frames, 10)
                
                # ビデオからフレームを抽出
                print(f"  ビデオからフレームを抽出中...")
                actual_frames = extract_frames(video_file, images_out_path, cam_idx, n_frames, image_size, debug)
                
                if actual_frames == 0:
                    print(f"  フレームを抽出できませんでした")
                    continue
                
                # 実際に抽出できたフレーム数に合わせる
                n_frames = min(n_frames, actual_frames)
                
                # H5ファイルを作成
                print(f"  アノテーションデータをH5形式に変換中...")
                out_file = os.path.join(cam_out_path, f"annot_cam{cam_idx}.h5")
                
                with h5py.File(out_file, 'w') as f:
                    # データセットを作成
                    f.create_dataset('poses_2d', data=poses_2d[:n_frames])
                    f.create_dataset('poses_3d', data=poses_3d[:n_frames])
                    
                    # カメラパラメータを保存
                    camera_data = np.zeros((n_frames, 21), dtype=np.float32)
                    
                    # 回転行列 (3x3) を平坦化して保存
                    R_flat = camera_param['R'].reshape(9)
                    camera_data[:, :9] = np.tile(R_flat, (n_frames, 1))
                    
                    # 平行移動ベクトル (3x1) を保存
                    t_flat = camera_param['t'].reshape(3)
                    camera_data[:, 9:12] = np.tile(t_flat, (n_frames, 1))
                    
                    # 内部パラメータ行列 (3x3) を平坦化して保存
                    K_flat = camera_param['K'].reshape(9)
                    camera_data[:, 12:21] = np.tile(K_flat, (n_frames, 1))
                    
                    f.create_dataset('camera', data=camera_data)
                
                print(f"  カメラ{cam_idx}の処理が完了しました: {out_file}")
    
    print(f"MPI-INF-3DHPデータセットの前処理が完了しました!")


def parse_camera_calibration(calib_file):
    """カメラキャリブレーションファイルを解析します"""
    cameras = {}
    
    with open(calib_file, 'r') as f:
        lines = f.readlines()
    
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if line.startswith('name'):
            # カメラ名を取得
            parts = line.split()
            if len(parts) >= 2:
                camera_id = parts[1].strip()
                camera_params = {}
                
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith('name'):
                    param_line = lines[i].strip()
                    if param_line.startswith('intrinsic'):
                        # 内部パラメータを解析
                        values = [float(v) for v in param_line.split()[1:]]
                        K = np.array(values).reshape(4, 4)[:3, :3]  # 3x3行列を抽出
                        camera_params['K'] = K
                    elif param_line.startswith('extrinsic'):
                        # 外部パラメータを解析
                        values = [float(v) for v in param_line.split()[1:]]
                        RT = np.array(values).reshape(4, 4)
                        R = RT[:3, :3]  # 回転行列
                        t = RT[:3, 3]   # 平行移動ベクトル
                        camera_params['R'] = R
                        camera_params['t'] = t
                    
                    i += 1
                
                cameras[camera_id] = camera_params
            else:
                i += 1
        else:
            i += 1
    
    return cameras


def extract_3d_poses(annot_data, cam_idx):
    """
    指定したカメラの3Dポーズを抽出します
    
    Args:
        annot_data: アノテーションデータ
        cam_idx: カメラインデックス
        
    Returns:
        poses_3d: 3Dポーズデータ [フレーム数, 関節数, 3]
    """
    # annot3はカメラごとに格納されている
    if 'annot3' in annot_data and isinstance(annot_data['annot3'], np.ndarray):
        if annot_data['annot3'].shape[0] > cam_idx:
            # (14, 1)の形状で、各要素が実際のデータを含む
            cam_data = annot_data['annot3'][cam_idx, 0]
            if isinstance(cam_data, np.ndarray):
                # 関節数を取得（通常は28だが、ここでは17に変換）
                if cam_data.shape[1] % 3 == 0:
                    n_joints_orig = cam_data.shape[1] // 3
                    # 17関節に変換
                    return convert_to_17_joints_3d(cam_data)
                return cam_data
    
    # univ_annot3も同様の形式で格納されている
    if 'univ_annot3' in annot_data and isinstance(annot_data['univ_annot3'], np.ndarray):
        if annot_data['univ_annot3'].shape[0] > cam_idx:
            cam_data = annot_data['univ_annot3'][cam_idx, 0]
            if isinstance(cam_data, np.ndarray):
                if cam_data.shape[1] % 3 == 0:
                    n_joints_orig = cam_data.shape[1] // 3
                    return convert_to_17_joints_3d(cam_data)
                return cam_data
    
    print(f"警告: カメラ{cam_idx}の3Dポーズデータが見つかりません。")
    return None


def extract_2d_poses(annot_data, cam_idx):
    """
    指定したカメラの2Dポーズを抽出します
    
    Args:
        annot_data: アノテーションデータ
        cam_idx: カメラインデックス
        
    Returns:
        poses_2d: 2Dポーズデータ [フレーム数, 関節数, 2]
    """
    # annot2はカメラごとに格納されている
    if 'annot2' in annot_data and isinstance(annot_data['annot2'], np.ndarray):
        if annot_data['annot2'].shape[0] > cam_idx:
            # (14, 1)の形状で、各要素が実際のデータを含む
            cam_data = annot_data['annot2'][cam_idx, 0]
            if isinstance(cam_data, np.ndarray):
                # 関節数を取得（通常は28だが、ここでは17に変換）
                if cam_data.shape[1] % 2 == 0:
                    n_joints_orig = cam_data.shape[1] // 2
                    # 17関節に変換
                    return convert_to_17_joints_2d(cam_data)
                return cam_data
    
    print(f"警告: カメラ{cam_idx}の2Dポーズデータが見つかりません。")
    return None


def convert_to_17_joints_3d(poses):
    """
    元のMPI-INF-3DHP形式（通常28関節）から17関節形式に変換します
    
    Args:
        poses: 元の3Dポーズデータ [フレーム数, 関節数*3]
        
    Returns:
        poses_17: 17関節形式の3Dポーズデータ [フレーム数, 17, 3]
    """
    n_frames = poses.shape[0]
    n_joints_orig = poses.shape[1] // 3
    
    # 元のポーズを[フレーム数, 関節数, 3]形式に変換
    poses_reshaped = poses.reshape(n_frames, n_joints_orig, 3)
    
    # MPI-INF-3DHPのインデックスマッピング（例）
    # 実際のデータセットの関節定義に合わせて調整が必要
    if n_joints_orig == 28:
        # 28関節から17関節へのマッピング
        # 以下のマッピングは例示であり、実際のデータセットに合わせて調整が必要です
        joint_map = [0,  # 骨盤
                    4,  # 胸部
                    7,  # 首
                    8,  # 頭
                    9,  # 左股関節
                    10, # 左膝
                    11, # 左足首
                    14, # 右股関節
                    15, # 右膝
                    16, # 右足首
                    18, # 左肩
                    19, # 左肘
                    20, # 左手首
                    23, # 右肩
                    24, # 右肘
                    25, # 右手首
                    27] # 中心
        poses_17 = poses_reshaped[:, joint_map, :]
    else:
        # 関節数が不明な場合、可能な限り先頭の17関節を使用
        poses_17 = poses_reshaped[:, :17, :] if n_joints_orig >= 17 else poses_reshaped
    
    return poses_17


def convert_to_17_joints_2d(poses):
    """
    元のMPI-INF-3DHP形式（通常28関節）から17関節形式に変換します
    
    Args:
        poses: 元の2Dポーズデータ [フレーム数, 関節数*2]
        
    Returns:
        poses_17: 17関節形式の2Dポーズデータ [フレーム数, 17, 2]
    """
    n_frames = poses.shape[0]
    n_joints_orig = poses.shape[1] // 2
    
    # 元のポーズを[フレーム数, 関節数, 2]形式に変換
    poses_reshaped = poses.reshape(n_frames, n_joints_orig, 2)
    
    # MPI-INF-3DHPのインデックスマッピング（例）
    if n_joints_orig == 28:
        # 28関節から17関節へのマッピング
        joint_map = [0,  # 骨盤
                    4,  # 胸部
                    7,  # 首
                    8,  # 頭
                    9,  # 左股関節
                    10, # 左膝
                    11, # 左足首
                    14, # 右股関節
                    15, # 右膝
                    16, # 右足首
                    18, # 左肩
                    19, # 左肘
                    20, # 左手首
                    23, # 右肩
                    24, # 右肘
                    25, # 右手首
                    27] # 中心
        poses_17 = poses_reshaped[:, joint_map, :]
    else:
        # 関節数が不明な場合、可能な限り先頭の17関節を使用
        poses_17 = poses_reshaped[:, :17, :] if n_joints_orig >= 17 else poses_reshaped
    
    return poses_17


def extract_frames(video_file, output_dir, cam_idx, n_frames, image_size=(224, 224), debug=False):
    """
    ビデオからフレームを抽出します
    
    Args:
        video_file: ビデオファイルのパス
        output_dir: 出力ディレクトリ
        cam_idx: カメラインデックス
        n_frames: 抽出するフレーム数
        image_size: 出力画像のサイズ (width, height)
        debug: デバッグモード
        
    Returns:
        actual_frames: 実際に抽出されたフレーム数
    """
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        print(f"ビデオを開けませんでした: {video_file}")
        return 0
    
    video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # デバッグモードの場合はさらにフレーム数を制限
    if debug:
        n_frames = min(n_frames, 10)
    
    # 適切なサンプリングレートを計算
    if video_frame_count > n_frames:
        sampling_rate = video_frame_count // n_frames
    else:
        sampling_rate = 1
    
    print(f"  ビデオフレーム数: {video_frame_count}, 抽出フレーム数: {n_frames}, サンプリングレート: {sampling_rate}")
    
    frame_idx = 0
    saved_count = 0
    
    for i in tqdm(range(min(n_frames, video_frame_count)), desc=f"  カメラ{cam_idx}のフレーム抽出"):
        # 指定位置にシーク
        target_frame = i * sampling_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # 画像をリサイズ
        frame_resized = cv2.resize(frame, image_size)
        
        # フレームを保存
        out_path = os.path.join(output_dir, f'img_cam{cam_idx}_{saved_count:06d}.jpg')
        cv2.imwrite(out_path, frame_resized)
        
        saved_count += 1
    
    cap.release()
    print(f"  {saved_count}フレームを抽出しました")
    return saved_count


def main():
    parser = argparse.ArgumentParser(description='MPI-INF-3DHPデータセットの前処理')
    parser.add_argument('--dataset_path', type=str, required=True, help='元のデータセットのパス')
    parser.add_argument('--output_path', type=str, required=True, help='出力ディレクトリのパス')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224], help='処理後の画像サイズ')
    parser.add_argument('--debug', action='store_true', help='デバッグモード（少量のデータのみ処理）')
    
    args = parser.parse_args()
    
    preprocess_mpiinf3dhp(args.dataset_path, args.output_path, tuple(args.image_size), args.debug)


if __name__ == "__main__":
    main()
    
    