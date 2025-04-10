"""
ポーズ推定の可視化のためのユーティリティ関数
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2
import torch
from .pose_utils import JOINT_CONNECTIONS

# MPIIデータセット形式の関節の名前
JOINT_NAMES = [
    'Pelvis', 'Spine', 'Neck', 'Head',
    'L_Hip', 'L_Knee', 'L_Ankle',
    'R_Hip', 'R_Knee', 'R_Ankle',
    'L_Shoulder', 'L_Elbow', 'L_Wrist',
    'R_Shoulder', 'R_Elbow', 'R_Wrist'
]

# 関節とリンクの色設定
JOINT_COLORS = {
    'trunk': 'blue',          # 胴体 (骨盤, 脊椎, 首, 頭)
    'left_leg': 'green',      # 左脚 (左腰, 左膝, 左足首)
    'right_leg': 'red',       # 右脚 (右腰, 右膝, 右足首)
    'left_arm': 'orange',     # 左腕 (左肩, 左肘, 左手首)
    'right_arm': 'purple'     # 右腕 (右肩, 右肘, 右手首)
}

# 関節のグループ化
JOINT_GROUPS = {
    'trunk': [0, 1, 2, 3],              # 胴体
    'left_leg': [4, 5, 6],              # 左脚
    'right_leg': [7, 8, 9],             # 右脚
    'left_arm': [10, 11, 12],           # 左腕
    'right_arm': [13, 14, 15]           # 右腕
}


def visualize_2d_pose(pose_2d, image=None, figsize=(8, 8), title=None, save_path=None):
    """
    2Dポーズを可視化
    
    Args:
        pose_2d: 2Dポーズ座標 [関節数, 2]
        image: (オプション) 背景画像 [H, W, 3]
        figsize: 図のサイズ
        title: 図のタイトル
        save_path: 保存先パス
        
    Returns:
        fig: Matplotlibの図オブジェクト
    """
    if isinstance(pose_2d, torch.Tensor):
        pose_2d = pose_2d.detach().cpu().numpy()
    
    # 図を作成
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    
    # 画像が提供された場合は表示
    if image is not None:
        ax.imshow(image)
    
    # 関節グループごとに描画
    for group_name, joint_indices in JOINT_GROUPS.items():
        color = JOINT_COLORS[group_name]
        
        # グループ内の関節を描画
        for j in joint_indices:
            ax.plot(pose_2d[j, 0], pose_2d[j, 1], 'o', color=color, markersize=8)
            
            # 関節名を表示（オプション）
            # ax.text(pose_2d[j, 0], pose_2d[j, 1], JOINT_NAMES[j], fontsize=8)
    
    # 骨を描画
    for joint1, joint2 in JOINT_CONNECTIONS:
        # 接続する関節のグループを特定
        group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
        group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
        
        # 両方の関節が同じグループに属している場合はそのグループの色を使用
        if group1 == group2:
            color = JOINT_COLORS[group1]
        else:
            # 異なるグループをつなぐ場合（例：脊椎と肩）は胴体の色を使用
            color = JOINT_COLORS['trunk']
        
        ax.plot([pose_2d[joint1, 0], pose_2d[joint2, 0]],
                [pose_2d[joint1, 1], pose_2d[joint2, 1]], '-', color=color, linewidth=2)
    
    # 軸の設定
    if image is None:
        # 画像がない場合は座標を反転（原点が左上）
        ax.invert_yaxis()
    
    # タイトルを設定
    if title:
        ax.set_title(title)
    
    # 軸ラベルを消す
    ax.set_xticks([])
    ax.set_yticks([])
    
    # 図を保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def visualize_3d_pose(pose_3d, title=None, figsize=(10, 10), save_path=None, elev=10, azim=45):
    """
    3Dポーズを可視化
    
    Args:
        pose_3d: 3Dポーズ座標 [関節数, 3]
        title: 図のタイトル
        figsize: 図のサイズ
        save_path: 保存先パス
        elev: 仰角 (度数法)
        azim: 方位角 (度数法)
        
    Returns:
        fig: Matplotlibの図オブジェクト
    """
    if isinstance(pose_3d, torch.Tensor):
        pose_3d = pose_3d.detach().cpu().numpy()
    
    # 図を作成
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')
    
    # 視点の設定
    ax.view_init(elev=elev, azim=azim)
    
    # 関節グループごとに描画
    for group_name, joint_indices in JOINT_GROUPS.items():
        color = JOINT_COLORS[group_name]
        
        # グループ内の関節を描画
        for j in joint_indices:
            ax.scatter(pose_3d[j, 0], pose_3d[j, 1], pose_3d[j, 2], color=color, s=50)
            
            # 関節名を表示（オプション）
            # ax.text(pose_3d[j, 0], pose_3d[j, 1], pose_3d[j, 2], JOINT_NAMES[j], fontsize=8)
    
    # 骨を描画
    for joint1, joint2 in JOINT_CONNECTIONS:
        # 接続する関節のグループを特定
        group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
        group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
        
        # 両方の関節が同じグループに属している場合はそのグループの色を使用
        if group1 == group2:
            color = JOINT_COLORS[group1]
        else:
            # 異なるグループをつなぐ場合（例：脊椎と肩）は胴体の色を使用
            color = JOINT_COLORS['trunk']
        
        ax.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                [pose_3d[joint1, 2], pose_3d[joint2, 2]], '-', color=color, linewidth=2)
    
    # 軸の設定
    # 統一したスケールで表示
    max_range = np.max(np.abs(pose_3d.max(axis=0) - pose_3d.min(axis=0)))
    mid_x = 0.5 * (pose_3d[:, 0].max() + pose_3d[:, 0].min())
    mid_y = 0.5 * (pose_3d[:, 1].max() + pose_3d[:, 1].min())
    mid_z = 0.5 * (pose_3d[:, 2].max() + pose_3d[:, 2].min())
    
    ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    # 軸ラベル
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    # タイトルを設定
    if title:
        ax.set_title(title)
    
    # 図を保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    plt.tight_layout()
    return fig


def visualize_results(image, pose_2d, pose_3d, figsize=(16, 8), save_path=None):
    """
    入力画像、2Dポーズ、3Dポーズを並べて可視化
    
    Args:
        image: 入力画像 [H, W, 3]
        pose_2d: 2Dポーズ座標 [関節数, 2]
        pose_3d: 3Dポーズ座標 [関節数, 3]
        figsize: 図のサイズ
        save_path: 保存先パス
        
    Returns:
        fig: Matplotlibの図オブジェクト
    """
    if isinstance(pose_2d, torch.Tensor):
        pose_2d = pose_2d.detach().cpu().numpy()
    if isinstance(pose_3d, torch.Tensor):
        pose_3d = pose_3d.detach().cpu().numpy()
    
    # 図を作成
    fig = plt.figure(figsize=figsize)
    
    # 元画像
    ax1 = fig.add_subplot(131)
    ax1.imshow(image)
    ax1.set_title('入力画像')
    ax1.set_xticks([])
    ax1.set_yticks([])
    
    # 2Dポーズ
    ax2 = fig.add_subplot(132)
    for group_name, joint_indices in JOINT_GROUPS.items():
        color = JOINT_COLORS[group_name]
        for j in joint_indices:
            ax2.plot(pose_2d[j, 0], pose_2d[j, 1], 'o', color=color, markersize=8)
    
    for joint1, joint2 in JOINT_CONNECTIONS:
        group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
        group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
        
        if group1 == group2:
            color = JOINT_COLORS[group1]
        else:
            color = JOINT_COLORS['trunk']
        
        ax2.plot([pose_2d[joint1, 0], pose_2d[joint2, 0]],
                 [pose_2d[joint1, 1], pose_2d[joint2, 1]], '-', color=color, linewidth=2)
    
    ax2.imshow(image, alpha=0.5)  # 画像を半透明で表示
    ax2.set_title('2Dポーズ')
    ax2.set_xticks([])
    ax2.set_yticks([])
    
    # 3Dポーズ
    ax3 = fig.add_subplot(133, projection='3d')
    for group_name, joint_indices in JOINT_GROUPS.items():
        color = JOINT_COLORS[group_name]
        for j in joint_indices:
            ax3.scatter(pose_3d[j, 0], pose_3d[j, 1], pose_3d[j, 2], color=color, s=50)
    
    for joint1, joint2 in JOINT_CONNECTIONS:
        group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
        group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
        
        if group1 == group2:
            color = JOINT_COLORS[group1]
        else:
            color = JOINT_COLORS['trunk']
        
        ax3.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                 [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                 [pose_3d[joint1, 2], pose_3d[joint2, 2]], '-', color=color, linewidth=2)
    
    # 3D表示の設定
    max_range = np.max(np.abs(pose_3d.max(axis=0) - pose_3d.min(axis=0)))
    mid_x = 0.5 * (pose_3d[:, 0].max() + pose_3d[:, 0].min())
    mid_y = 0.5 * (pose_3d[:, 1].max() + pose_3d[:, 1].min())
    mid_z = 0.5 * (pose_3d[:, 2].max() + pose_3d[:, 2].min())
    
    ax3.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
    ax3.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
    ax3.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
    
    ax3.view_init(elev=10, azim=45)
    ax3.set_title('3Dポーズ')
    
    plt.tight_layout()
    
    # 図を保存
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    
    return fig


def create_video_visualization(frames, poses_2d, poses_3d, output_path, fps=30):
    """
    ビデオフレームとポーズ推定の結果を可視化した動画を作成
    
    Args:
        frames: ビデオフレームのリスト [フレーム数, H, W, 3]
        poses_2d: 2Dポーズ座標のリスト [フレーム数, 関節数, 2]
        poses_3d: 3Dポーズ座標のリスト [フレーム数, 関節数, 3]
        output_path: 出力動画のパス
        fps: フレームレート
    """
    if isinstance(frames, list):
        frames = np.array(frames)
    if isinstance(poses_2d, list):
        poses_2d = np.array(poses_2d)
    if isinstance(poses_3d, list):
        poses_3d = np.array(poses_3d)
    
    if isinstance(poses_2d, torch.Tensor):
        poses_2d = poses_2d.detach().cpu().numpy()
    if isinstance(poses_3d, torch.Tensor):
        poses_3d = poses_3d.detach().cpu().numpy()
    
    num_frames = len(frames)
    height, width = frames[0].shape[:2]
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    for i in range(num_frames):
        # 現在のフレームとポーズ
        frame = frames[i]
        pose_2d = poses_2d[i]
        pose_3d = poses_3d[i]
        
        # 2Dポーズを描画したフレーム
        frame_with_pose = frame.copy()
        
        # 骨を描画
        for joint1, joint2 in JOINT_CONNECTIONS:
            # 接続する関節のグループを特定
            group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
            group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
            
            # 両方の関節が同じグループに属している場合はそのグループの色を使用
            if group1 == group2:
                color_name = group1
            else:
                # 異なるグループをつなぐ場合（例：脊椎と肩）は胴体の色を使用
                color_name = 'trunk'
            
            # Matplotlibの色名をOpenCVのBGR形式に変換
            if color_name == 'blue':
                color = (255, 0, 0)
            elif color_name == 'green':
                color = (0, 255, 0)
            elif color_name == 'red':
                color = (0, 0, 255)
            elif color_name == 'orange':
                color = (0, 165, 255)
            elif color_name == 'purple':
                color = (128, 0, 128)
            else:
                color = (255, 0, 0)  # デフォルトは青
            
            # 2Dポーズの線を描画
            pt1 = (int(pose_2d[joint1, 0]), int(pose_2d[joint1, 1]))
            pt2 = (int(pose_2d[joint2, 0]), int(pose_2d[joint2, 1]))
            cv2.line(frame_with_pose, pt1, pt2, color, 2)
        
        # 関節を描画
        for group_name, joint_indices in JOINT_GROUPS.items():
            # Matplotlibの色名をOpenCVのBGR形式に変換
            if group_name == 'trunk':
                color = (255, 0, 0)  # 青
            elif group_name == 'left_leg':
                color = (0, 255, 0)  # 緑
            elif group_name == 'right_leg':
                color = (0, 0, 255)  # 赤
            elif group_name == 'left_arm':
                color = (0, 165, 255)  # オレンジ
            elif group_name == 'right_arm':
                color = (128, 0, 128)  # 紫
            
            for j in joint_indices:
                cv2.circle(frame_with_pose, (int(pose_2d[j, 0]), int(pose_2d[j, 1])), 5, color, -1)
        
        # 3Dポーズを描画
        fig = plt.figure(figsize=(width/100, height/100), dpi=100)
        ax = fig.add_subplot(111, projection='3d')
        
        for group_name, joint_indices in JOINT_GROUPS.items():
            color = JOINT_COLORS[group_name]
            for j in joint_indices:
                ax.scatter(pose_3d[j, 0], pose_3d[j, 1], pose_3d[j, 2], color=color, s=50)
        
        for joint1, joint2 in JOINT_CONNECTIONS:
            group1 = next((g for g, indices in JOINT_GROUPS.items() if joint1 in indices), None)
            group2 = next((g for g, indices in JOINT_GROUPS.items() if joint2 in indices), None)
            
            if group1 == group2:
                color = JOINT_COLORS[group1]
            else:
                color = JOINT_COLORS['trunk']
            
            ax.plot([pose_3d[joint1, 0], pose_3d[joint2, 0]],
                    [pose_3d[joint1, 1], pose_3d[joint2, 1]],
                    [pose_3d[joint1, 2], pose_3d[joint2, 2]], '-', color=color, linewidth=2)
        
        # 3D表示の設定
        max_range = np.max(np.abs(pose_3d.max(axis=0) - pose_3d.min(axis=0)))
        mid_x = 0.5 * (pose_3d[:, 0].max() + pose_3d[:, 0].min())
        mid_y = 0.5 * (pose_3d[:, 1].max() + pose_3d[:, 1].min())
        mid_z = 0.5 * (pose_3d[:, 2].max() + pose_3d[:, 2].min())
        
        ax.set_xlim(mid_x - max_range/2, mid_x + max_range/2)
        ax.set_ylim(mid_y - max_range/2, mid_y + max_range/2)
        ax.set_zlim(mid_z - max_range/2, mid_z + max_range/2)
        
        ax.view_init(elev=10, azim=45)
        ax.set_axis_off()
        
        # Matplotlibの図をOpenCVの画像に変換
        fig.canvas.draw()
        pose_3d_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        pose_3d_img = pose_3d_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        pose_3d_img = cv2.resize(pose_3d_img, (width, height))
        pose_3d_img = cv2.cvtColor(pose_3d_img, cv2.COLOR_RGB2BGR)
        
        plt.close(fig)
        
        # 2Dポーズと3Dポーズの可視化を横に並べる
        combined_img = np.hstack((frame_with_pose, pose_3d_img))
        
        # ビデオに書き込み
        out.write(combined_img)
    
    out.release()
    print(f"ビデオを保存しました: {output_path}")