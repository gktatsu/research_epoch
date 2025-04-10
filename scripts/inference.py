"""
EPOCHモデルの単一画像に対する推論スクリプト

学習済みモデルを使用して、単一の入力画像から3Dポーズとカメラパラメータを推定
"""
import os
import sys
import argparse
import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt

# プロジェクトのルートディレクトリをパスに追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_DIR, MODELS_DIR, VISUALIZATIONS_DIR
from src.models import EPOCH, RegNet, LiftNet
from src.utils.visualization import visualize_2d_pose, visualize_3d_pose, visualize_results


def preprocess_image(image_path, image_size=(224, 224)):
    """
    入力画像を前処理
    
    Args:
        image_path: 入力画像のパス
        image_size: 画像サイズ (幅, 高さ)
        
    Returns:
        image_tensor: 前処理された画像テンソル [1, 3, H, W]
        orig_image: 元の画像 [H, W, 3]
        bbox: 人物のバウンディングボックス [1, 4] (左, 上, 幅, 高さ)
    """
    # 画像を読み込む
    orig_image = cv2.imread(image_path)
    if orig_image is None:
        raise ValueError(f"画像を読み込めませんでした: {image_path}")
    
    orig_image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB)
    h, w = orig_image.shape[:2]
    
    # バウンディングボックスを定義（ここでは画像全体を使用）
    # 実際のアプリケーションでは人物検出器を使用するとよい
    bbox = np.array([[0, 0, w, h]], dtype=np.float32)
    
    # 画像をリサイズ
    image = cv2.resize(orig_image, image_size)
    
    # NumPy配列からTensorに変換 [H, W, 3] -> [1, 3, H, W]
    image_tensor = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
    image_tensor = image_tensor.unsqueeze(0)  # バッチ次元を追加
    
    # バウンディングボックスをテンソルに変換
    bbox_tensor = torch.from_numpy(bbox).float()
    
    return image_tensor, orig_image, bbox_tensor


def run_inference(model, image_tensor, bbox_tensor, device, model_type='epoch'):
    """
    モデルを使用して推論を実行
    
    Args:
        model: 推論に使用するモデル
        image_tensor: 入力画像テンソル [1, 3, H, W]
        bbox_tensor: バウンディングボックステンソル [1, 4]
        device: デバイス
        model_type: モデルのタイプ ('epoch', 'regnet', or 'liftnet')
        
    Returns:
        outputs: 推論結果の辞書
    """
    # データをデバイスに転送
    image_tensor = image_tensor.to(device)
    bbox_tensor = bbox_tensor.to(device)
    
    # 推論モードに設定
    model.eval()
    
    # 勾配計算を無効化
    with torch.no_grad():
        if model_type == 'epoch':
            # EPOCHモデルの推論
            outputs = model(image_tensor, bbox_tensor)
            
            # 必要な出力を取得
            pose_2d = outputs['pose_2d'][0].cpu().numpy()  # [関節数, 2]
            pose_3d = outputs['pose_3d_lift'][0].cpu().numpy()  # [関節数, 3]
            
            # RegNetとLiftNetの両方の出力を含む辞書を返す
            return {
                'pose_2d': pose_2d,
                'pose_3d_reg': outputs['pose_3d_reg'][0].cpu().numpy(),
                'pose_3d_lift': pose_3d,
                'cam_K': outputs['cam_K'][0].cpu().numpy(),
                'cam_R': outputs['cam_R'][0].cpu().numpy(),
                'cam_t': outputs['cam_t'][0].cpu().numpy()
            }
        
        elif model_type == 'regnet':
            # RegNetモデルの推論
            outputs = model(image_tensor, bbox_tensor)
            
            # 必要な出力を取得
            pose_2d = outputs['pose_2d'][0].cpu().numpy()  # [関節数, 2]
            pose_3d = outputs['pose_3d'][0].cpu().numpy()  # [関節数, 3]
            
            return {
                'pose_2d': pose_2d,
                'pose_3d': pose_3d,
                'cam_K': outputs['cam_K'][0].cpu().numpy(),
                'cam_R': outputs['cam_R'][0].cpu().numpy(),
                'cam_t': outputs['cam_t'][0].cpu().numpy()
            }
        
        elif model_type == 'liftnet':
            # LiftNetモデルは画像から直接推論できないため、
            # 2Dポーズとカメラパラメータが必要
            raise ValueError("LiftNetモデルは単独での推論をサポートしていません。EPOCHまたはRegNetモデルを使用してください。")


def main():
    parser = argparse.ArgumentParser(description='EPOCHモデルの単一画像に対する推論')
    parser.add_argument('--image_path', type=str, required=True,
                        help='入力画像のパス')
    parser.add_argument('--model_path', type=str, required=True,
                        help='学習済みモデルのパス')
    parser.add_argument('--model_type', type=str, choices=['epoch', 'regnet'], default='epoch',
                        help='モデルのタイプ')
    parser.add_argument('--output_path', type=str, default=None,
                        help='出力画像のパス')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='使用するデバイス')
    parser.add_argument('--num_joints', type=int, default=17,
                        help='関節の数')
    parser.add_argument('--image_size', type=int, nargs=2, default=[224, 224],
                        help='入力画像のサイズ (幅 高さ)')
    
    args = parser.parse_args()
    
    # 出力パスのデフォルト値を設定
    if args.output_path is None:
        os.makedirs(VISUALIZATIONS_DIR, exist_ok=True)
        output_filename = os.path.splitext(os.path.basename(args.image_path))[0] + "_result.png"
        args.output_path = os.path.join(VISUALIZATIONS_DIR, output_filename)
    
    # 出力ディレクトリを作成
    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # デバイスを設定
    device = torch.device(args.device)
    print(f"使用するデバイス: {device}")
    
    # モデルをロード
    print(f"モデルをロードしています: {args.model_path}")
    model_type = args.model_type.lower()
    
    if model_type == 'epoch':
        model = EPOCH(
            num_joints=args.num_joints,
            encoder_name='resnet50',
            pretrained=False,
            image_size=tuple(args.image_size)
        ).to(device)
    elif model_type == 'regnet':
        model = RegNet(
            num_joints=args.num_joints,
            encoder_name='resnet50',
            pretrained=False,
            image_size=tuple(args.image_size)
        ).to(device)
    else:
        raise ValueError(f"未対応のモデルタイプ: {model_type}")
    
    # モデルの重みをロード
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 画像を前処理
    print(f"画像を前処理しています: {args.image_path}")
    image_tensor, orig_image, bbox_tensor = preprocess_image(
        args.image_path, 
        image_size=tuple(args.image_size)
    )
    
    # 推論を実行
    print("推論を実行しています...")
    outputs = run_inference(model, image_tensor, bbox_tensor, device, model_type)
    
    # 結果を可視化
    print("結果を可視化しています...")
    if model_type == 'epoch':
        # EPOCHモデルの場合はRegNetとLiftNetの両方の結果を表示
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 元画像と2Dポーズの可視化
        fig_2d = visualize_2d_pose(outputs['pose_2d'], image=orig_image, title='2Dポーズ推定')
        plt.close(fig_2d)  # 個別の図は閉じる
        axes[0].imshow(orig_image)
        visualize_2d_pose(outputs['pose_2d'], image=orig_image, ax=axes[0], title='2Dポーズ推定')
        
        # RegNetの3Dポーズ推定
        visualize_3d_pose(outputs['pose_3d_reg'], ax=axes[1], title='RegNetの3Dポーズ推定', elev=20, azim=30)
        
        # LiftNetの3Dポーズ推定
        visualize_3d_pose(outputs['pose_3d_lift'], ax=axes[2], title='LiftNetの3Dポーズ推定', elev=20, azim=30)
        
        plt.tight_layout()
        plt.savefig(args.output_path)
        print(f"結果を保存しました: {args.output_path}")
        
        # もう一つの視点からの3Dポーズも保存
        side_view_path = os.path.splitext(args.output_path)[0] + "_side_view.png"
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        visualize_3d_pose(outputs['pose_3d_reg'], ax=axes[0], title='RegNetの3Dポーズ（側面図）', elev=20, azim=90)
        visualize_3d_pose(outputs['pose_3d_lift'], ax=axes[1], title='LiftNetの3Dポーズ（側面図）', elev=20, azim=90)
        plt.tight_layout()
        plt.savefig(side_view_path)
        print(f"側面図を保存しました: {side_view_path}")
    
    else:
        # RegNetモデルの場合は2Dと3Dポーズのみを表示
        fig = visualize_results(orig_image, outputs['pose_2d'], outputs['pose_3d'])
        plt.savefig(args.output_path)
        print(f"結果を保存しました: {args.output_path}")
    
    print("完了しました!")


if __name__ == '__main__':
    main()