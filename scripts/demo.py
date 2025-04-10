"""
EPOCHモデルの対話型デモスクリプト

学習済みモデルを使用して、画像またはビデオから3Dポーズとカメラパラメータを推定し、
結果を視覚化するためのデモプログラム
"""
import os
import sys
import argparse
import glob
import time
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tqdm import tqdm

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_DIR, MODELS_DIR, VISUALIZATIONS_DIR, EXAMPLES_DIR
from src.models import EPOCH, RegNet, LiftNet
from src.utils.visualization import (
    visualize_2d_pose, 
    visualize_3d_pose, 
    visualize_results,
    create_video_visualization
)


def preprocess_image(image, image_size=(224, 224)):
    """
    入力画像を前処理
    
    Args:
        image: 入力画像 [H, W, 3]
        image_size: 画像サイズ (幅, 高さ)
        
    Returns:
        image_tensor: 前処理された画像テンソル [1, 3, H, W]
        bbox: 人物のバウンディングボックス [1, 4] (左, 上, 幅, 高さ)
    """
    h, w = image.shape[:2]
    
    # バウンディングボックスを定義（ここでは画像全体を使用）
    # 実際のアプリケーションでは人物検出器を使用するとよい
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
    
    # 画像をリサイズ
    image_resized = cv2.resize(image, image_size)
    
    # NumPy配列からTensorに変換 [H, W, 3] -> [1, 3, H, W]
    image_tensor = torch.from_numpy(image_resized.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # バッチ次元を追加
    
    # バウンディングボックスをテンソルに変換
    bbox_tensor = torch.from_numpy(bbox).float()
    
    return image_tensor, bbox_tensor


def process_image(model, image, device, image_size=(224, 224)):
    """
    モデルを使用して1枚の画像を処理
    
    Args:
        model: 使用するモデル
        image: 入力画像 [H, W, 3]
        device: デバイス
        image_size: 処理する画像サイズ
        
    Returns:
        results: 2Dポーズと3Dポーズを含む辞書
    """
    # 画像を前処理
    image_tensor, bbox_tensor = preprocess_image(image, image_size)
    
    # データをデバイスに転送
    image_tensor = image_tensor.to(device)
    bbox_tensor = bbox_tensor.to(device)
    
    # 勾配計算を無効化
    with torch.no_grad():
        # 推論を実行
        outputs = model(image_tensor, bbox_tensor)
        
        # 結果を取得
        pose_2d = outputs['pose_2d'][0].cpu().numpy()
        pose_3d_reg = outputs['pose_3d_reg'][0].cpu().numpy()
        pose_3d_lift = outputs['pose_3d_lift'][0].cpu().numpy()
        
        return {
            'pose_2d': pose_2d,
            'pose_3d_reg': pose_3d_reg,
            'pose_3d_lift': pose_3d_lift
        }


def process_video(model, video_path, output_path, device, image_size=(224, 224), max_frames=None):
    """
    モデルを使用してビデオを処理
    
    Args:
        model: 使用するモデル
        video_path: 入力ビデオのパス
        output_path: 出力ビデオのパス
        device: デバイス
        image_size: 処理する画像サイズ
        max_frames: 処理する最大フレーム数（Noneの場合は全て）
        
    Returns:
        None（ビデオファイルに結果を保存）
    """
    # ビデオキャプチャを開く
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"ビデオを開けませんでした: {video_path}")
    
    # ビデオの情報を取得
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 処理するフレーム数を決定
    if max_frames is not None and max_frames < total_frames:
        frames_to_process = max_frames
    else:
        frames_to_process = total_frames
    
    print(f"ビデオ情報: {width}x{height}, {fps}fps, 全{total_frames}フレーム")
    print(f"処理するフレーム数: {frames_to_process}")
    
    # 結果を格納するリスト
    frames = []
    poses_2d = []
    poses_3d = []
    
    # 各フレームを処理
    with tqdm(total=frames_to_process, desc="ビデオフレーム処理中") as pbar:
        for frame_idx in range(frames_to_process):
            ret, frame = cap.read()
            if not ret:
                break
            
            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # フレームを処理
            results = process_image(model, frame_rgb, device, image_size)
            
            # 結果を保存
            frames.append(frame_rgb)
            poses_2d.append(results['pose_2d'])
            poses_3d.append(results['pose_3d_lift'])  # LiftNetの出力を使用
            
            pbar.update(1)
    
    # ビデオキャプチャを解放
    cap.release()
    
    # 結果をビデオとして保存
    create_video_visualization(frames, poses_2d, poses_3d, output_path, fps=fps)
    print(f"処理結果を保存しました: {output_path}")


def process_image_folder(model, folder_path, output_dir, device, image_size=(224, 224)):
    """
    フォルダ内の画像を処理
    
    Args:
        model: 使用するモデル
        folder_path: 入力画像フォルダのパス
        output_dir: 出力ディレクトリのパス
        device: デバイス
        image_size: 処理する画像サイズ
        
    Returns:
        None（画像ファイルに結果を保存）
    """
    # 出力ディレクトリを作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 画像ファイルのリストを取得
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        image_files.extend(glob.glob(os.path.join(folder_path, ext.upper())))
    
    if not image_files:
        print(f"画像が見つかりませんでした: {folder_path}")
        return
    
    print(f"処理する画像数: {len(image_files)}")
    
    # 各画像を処理
    for image_path in tqdm(image_files, desc="画像処理中"):
        # 画像ファイル名
        image_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # 画像を読み込む
        image = cv2.imread(image_path)
        if image is None:
            print(f"画像を読み込めませんでした: {image_path}")
            continue
        
        # BGRからRGBに変換
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 画像を処理
        results = process_image(model, image_rgb, device, image_size)
        
        # 結果を可視化して保存
        output_path = os.path.join(output_dir, f"{image_name}_result.png")
        fig = visualize_results(image_rgb, results['pose_2d'], results['pose_3d_lift'])
        plt.savefig(output_path)
        plt.close(fig)
        
        # 3Dポーズの側面図も保存
        side_view_path = os.path.join(output_dir, f"{image_name}_side_view.png")
        fig, ax = plt.subplots(figsize=(8, 8))
        visualize_3d_pose(results['pose_3d_lift'], ax=ax, title='3Dポーズ（側面図）', elev=20, azim=90)
        plt.savefig(side_view_path)
        plt.close(fig)
    
    print(f"処理結果を保存しました: {output_dir}")


def interactive_demo(model, device, image_size=(224, 224)):
    """
    カメラを使用した対話型デモ
    
    Args:
        model: 使用するモデル
        device: デバイス
        image_size: 処理する画像サイズ
        
    Returns:
        None
    """
    # カメラキャプチャを開く
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise ValueError("カメラを開けませんでした")
    
    # 表示ウィンドウを作成
    cv2.namedWindow('EPOCH Demo', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('EPOCH Demo', 1280, 720)
    
    print("対話型デモを開始します（終了するには 'q' キーを押してください）")
    print("処理中...")
    
    # フレームレート計測用
    frame_times = []
    
    try:
        while True:
            # フレームをキャプチャ
            ret, frame = cap.read()
            if not ret:
                break
            
            start_time = time.time()
            
            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # フレームを処理
            results = process_image(model, frame_rgb, device, image_size)
            
            # 2Dポーズを描画
            frame_with_pose = frame.copy()
            pose_2d = results['pose_2d']
            
            # 骨を描画
            from src.utils.pose_utils import JOINT_CONNECTIONS
            for joint1, joint2 in JOINT_CONNECTIONS:
                pt1 = (int(pose_2d[joint1, 0]), int(pose_2d[joint1, 1]))
                pt2 = (int(pose_2d[joint2, 0]), int(pose_2d[joint2, 1]))
                cv2.line(frame_with_pose, pt1, pt2, (0, 255, 0), 2)
            
            # 関節を描画
            for i in range(pose_2d.shape[0]):
                cv2.circle(frame_with_pose, (int(pose_2d[i, 0]), int(pose_2d[i, 1])), 5, (0, 0, 255), -1)
            
            # 処理時間を計算
            end_time = time.time()
            processing_time = end_time - start_time
            frame_times.append(processing_time)
            
            # 平均FPSを計算（直近30フレーム）
            if len(frame_times) > 30:
                frame_times.pop(0)
            avg_fps = 1.0 / (sum(frame_times) / len(frame_times))
            
            # 処理時間とFPSを描画
            cv2.putText(
                frame_with_pose, 
                f"Processing time: {processing_time*1000:.1f} ms (FPS: {avg_fps:.1f})", 
                (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                (0, 0, 255), 
                2
            )
            
            # 結果を表示
            cv2.imshow('EPOCH Demo', frame_with_pose)
            
            # 'q' キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    finally:
        # カメラキャプチャを解放
        cap.release()
        cv2.destroyAllWindows()
        print("デモを終了しました")


def main():
    parser = argparse.ArgumentParser(description='EPOCHモデルのデモ')
    parser.add_argument('--model_path', type=str, default=None,
                        help='学習済みモデルのパス')
    parser.add_argument('--mode', type=str, choices=['interactive', 'video', 'folder'], default='interactive',
                        help='デモモード')
    parser.add_argument('--input', type=str, default=None,
                        help='入力ビデオまたはフォルダのパス')
    parser.add_argument('--output', type=str, default=None,
                        help='出力ビデオまたはフォルダのパス')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--num_joints', type=int, default=17,
                        help='関節の数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='処理する画像サイズ (幅 高さ)')
    parser.add_argument('--max_frames', type=int, default=None,
                        help='ビデオモードで処理する最大フレーム数')
    
    args = parser.parse_args()
    
    # モデルパスのデフォルト値を設定
    if args.model_path is None:
        # ディレクトリからモデルファイルを検索
        model_files = glob.glob(os.path.join(MODELS_DIR, "epoch_*.pth"))
        if model_files:
            args.model_path = model_files[0]
            print(f"モデルが自動的に選択されました: {args.model_path}")
        else:
            raise ValueError("モデルが見つかりません。--model_pathオプションで指定してください。")
    
    # 出力パスのデフォルト値を設定
    if args.mode in ['video', 'folder'] and args.output is None:
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        if args.mode == 'video':
            args.output = os.path.join(VISUALIZATIONS_DIR, "output_video.mp4")
        else:
            args.output = os.path.join(VISUALIZATIONS_DIR, "output_images")
    
    # 入力のデフォルト値を設定（必要な場合）
    if args.mode in ['video', 'folder'] and args.input is None:
        if args.mode == 'video':
            # ビデオファイルのリストを取得
            video_files = glob.glob(os.path.join(EXAMPLES_DIR, "*.mp4"))
            video_files.extend(glob.glob(os.path.join(EXAMPLES_DIR, "*.avi")))
            if video_files:
                args.input = video_files[0]
                print(f"入力ビデオが自動的に選択されました: {args.input}")
            else:
                raise ValueError("入力ビデオが見つかりません。--inputオプションで指定してください。")
        else:
            # フォルダのデフォルト値
            args.input = EXAMPLES_DIR
            print(f"入力フォルダが自動的に選択されました: {args.input}")
    
    # デバイスを設定
    device = torch.device(args.device)
    print(f"使用するデバイス: {device}")
    
    # モデルをロード
    print(f"モデルをロードしています: {args.model_path}")
    model = EPOCH(
        num_joints=args.num_joints,
        encoder_name='resnet50',
        pretrained=False,
        image_size=tuple(args.image_size)
    ).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 選択したモードでデモを実行
    if args.mode == 'interactive':
        interactive_demo(model, device, image_size=tuple(args.image_size))
    
    elif args.mode == 'video':
        print(f"ビデオを処理しています: {args.input}")
        process_video(
            model, 
            args.input, 
            args.output, 
            device, 
            image_size=tuple(args.image_size),
            max_frames=args.max_frames
        )
    
    elif args.mode == 'folder':
        print(f"フォルダ内の画像を処理しています: {args.input}")
        process_image_folder(
            model, 
            args.input, 
            args.output, 
            device, 
            image_size=tuple(args.image_size)
        )


if __name__ == '__main__':
    main()