"""
3Dポーズの操作と変換のためのユーティリティ関数
"""
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# MPII形式での関節接続定義 (17関節)
JOINT_CONNECTIONS = [
    (0, 1),   # 骨盤 → 胸
    (1, 2),   # 胸 → 首
    (2, 3),   # 首 → 頭
    (0, 4),   # 骨盤 → 左股関節
    (4, 5),   # 左股関節 → 左膝
    (5, 6),   # 左膝 → 左足首
    (0, 7),   # 骨盤 → 右股関節
    (7, 8),   # 右股関節 → 右膝
    (8, 9),   # 右膝 → 右足首
    (1, 10),  # 胸 → 左肩
    (10, 11), # 左肩 → 左肘
    (11, 12), # 左肘 → 左手首
    (1, 13),  # 胸 → 右肩
    (13, 14), # 右肩 → 右肘
    (14, 15), # 右肘 → 右手首
]

# ヒップとスパインを使って計算する平面の法線に対して制約を適用する関節リスト
CONSTRAINED_LIMBS = [
    (4, 5),   # 左股関節 → 左膝
    (5, 6),   # 左膝 → 左足首
    (7, 8),   # 右股関節 → 右膝
    (8, 9),   # 右膝 → 右足首
    (10, 11), # 左肩 → 左肘
    (11, 12), # 左肘 → 左手首
    (13, 14), # 右肩 → 右肘
    (14, 15), # 右肘 → 右手首
]

# 骨長比率の計算に使用する骨の組み合わせ
BONE_PAIRS = [
    ((0, 1), (1, 2)),    # (骨盤→胸) と (胸→首)
    ((0, 4), (0, 7)),    # (骨盤→左股関節) と (骨盤→右股関節)
    ((4, 5), (7, 8)),    # (左股関節→左膝) と (右股関節→右膝)
    ((5, 6), (8, 9)),    # (左膝→左足首) と (右膝→右足首)
    ((1, 10), (1, 13)),  # (胸→左肩) と (胸→右肩)
    ((10, 11), (13, 14)), # (左肩→左肘) と (右肩→右肘)
    ((11, 12), (14, 15)), # (左肘→左手首) と (右肘→右手首)
]


def normalize_pose(pose):
    """
    ポーズを中心化して正規化
    
    Args:
        pose: 3Dポーズ [バッチサイズ, 関節数, 3] または [関節数, 3]
        
    Returns:
        normalized_pose: 正規化されたポーズ (同じ形状)
    """
    if isinstance(pose, np.ndarray):
        if pose.ndim == 2:  # [関節数, 3]
            # 骨盤（ルート関節）を原点に
            root_joint = pose[0:1]
            centered_pose = pose - root_joint
            
            # スケール正規化（骨盤から頭までの距離を1に）
            scale = np.linalg.norm(centered_pose[3] - centered_pose[0])
            normalized_pose = centered_pose / (scale + 1e-8)
            
            return normalized_pose
        else:  # [バッチサイズ, 関節数, 3]
            normalized_poses = []
            for i in range(pose.shape[0]):
                normalized_poses.append(normalize_pose(pose[i]))
            return np.stack(normalized_poses)
    
    elif isinstance(pose, torch.Tensor):
        if pose.dim() == 2:  # [関節数, 3]
            # 骨盤（ルート関節）を原点に
            root_joint = pose[0:1]
            centered_pose = pose - root_joint
            
            # スケール正規化（骨盤から頭までの距離を1に）
            scale = torch.norm(centered_pose[3] - centered_pose[0])
            normalized_pose = centered_pose / (scale + 1e-8)
            
            return normalized_pose
        else:  # [バッチサイズ, 関節数, 3]
            # バッチ処理
            root_joint = pose[:, 0:1]
            centered_pose = pose - root_joint
            
            # スケール正規化（骨盤から頭までの距離を1に）
            scale = torch.norm(centered_pose[:, 3] - centered_pose[:, 0], dim=-1, keepdim=True).unsqueeze(1)
            normalized_pose = centered_pose / (scale + 1e-8)
            
            return normalized_pose
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")


def rotate_pose(pose, angle_degrees, axis='y'):
    """
    指定した軸の周りでポーズを回転
    
    Args:
        pose: 3Dポーズ [バッチサイズ, 関節数, 3] または [関節数, 3]
        angle_degrees: 回転角度 (度数法)
        axis: 回転軸 ('x', 'y', または 'z')
        
    Returns:
        rotated_pose: 回転後のポーズ (同じ形状)
    """
    angle_rad = np.deg2rad(angle_degrees)
    
    # 回転行列を作成
    if axis.lower() == 'x':
        rotation_vector = [angle_rad, 0, 0]
    elif axis.lower() == 'y':
        rotation_vector = [0, angle_rad, 0]
    elif axis.lower() == 'z':
        rotation_vector = [0, 0, angle_rad]
    else:
        raise ValueError("軸は 'x', 'y', または 'z' である必要があります")
    
    rotation = Rotation.from_rotvec(rotation_vector)
    rotation_matrix = rotation.as_matrix()
    
    if isinstance(pose, np.ndarray):
        if pose.ndim == 2:  # [関節数, 3]
            return np.matmul(pose, rotation_matrix.T)
        else:  # [バッチサイズ, 関節数, 3]
            rotated_poses = []
            for i in range(pose.shape[0]):
                rotated_poses.append(np.matmul(pose[i], rotation_matrix.T))
            return np.stack(rotated_poses)
    
    elif isinstance(pose, torch.Tensor):
        rotation_matrix = torch.tensor(rotation_matrix, dtype=pose.dtype, device=pose.device)
        
        if pose.dim() == 2:  # [関節数, 3]
            return torch.matmul(pose, rotation_matrix.T)
        else:  # [バッチサイズ, 関節数, 3]
            return torch.matmul(pose, rotation_matrix.T)
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")


def project_pose(pose_3d, K, R, t):
    """
    透視投影による3Dポーズの2Dポーズへの変換
    
    Args:
        pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3]
        K: カメラ内部パラメータ行列 [バッチサイズ, 3, 3]
        R: 回転行列 [バッチサイズ, 3, 3]
        t: 平行移動ベクトル [バッチサイズ, 3, 1] または [バッチサイズ, 3]
        
    Returns:
        pose_2d: 投影された2Dポーズ [バッチサイズ, 関節数, 2]
    """
    batch_size, num_joints = pose_3d.shape[0], pose_3d.shape[1]
    
    # データ型とデバイスを取得
    dtype = pose_3d.dtype
    device = pose_3d.device
    
    # 平行移動ベクトルの形状を調整
    if t.dim() == 2:
        t = t.unsqueeze(-1)  # [B, 3] -> [B, 3, 1]
    
    # 同次座標に変換
    pose_homogeneous = torch.ones(batch_size, num_joints, 4, dtype=dtype, device=device)
    pose_homogeneous[:, :, :3] = pose_3d
    
    # 各バッチのポーズに変換行列を適用
    pose_2d_homogeneous = []
    for i in range(batch_size):
        # 外部パラメータを適用 [3, 4] @ [J, 4].T -> [3, J]
        RT = torch.cat([R[i], t[i]], dim=1)
        pose_camera = torch.matmul(RT, pose_homogeneous[i].transpose(0, 1))
        
        # 内部パラメータを適用 [3, 3] @ [3, J] -> [3, J]
        pose_image = torch.matmul(K[i], pose_camera)
        
        # 同次座標から2D座標へ
        pose_image = pose_image.transpose(0, 1)  # [J, 3]
        pose_2d_i = pose_image[:, :2] / pose_image[:, 2:3]
        
        pose_2d_homogeneous.append(pose_2d_i)
    
    pose_2d = torch.stack(pose_2d_homogeneous)
    
    return pose_2d


def back_project_pose(pose_2d, depth, K, R, t):
    """
    2Dポーズと深度から3Dポーズを復元（逆透視投影）
    
    Args:
        pose_2d: 2Dポーズ [バッチサイズ, 関節数, 2]
        depth: 各関節の深度 [バッチサイズ, 関節数, 1]
        K: カメラ内部パラメータ行列 [バッチサイズ, 3, 3]
        R: 回転行列 [バッチサイズ, 3, 3]
        t: 平行移動ベクトル [バッチサイズ, 3, 1] または [バッチサイズ, 3]
        
    Returns:
        pose_3d: 復元された3Dポーズ [バッチサイズ, 関節数, 3]
    """
    batch_size, num_joints = pose_2d.shape[0], pose_2d.shape[1]
    
    # データ型とデバイスを取得
    dtype = pose_2d.dtype
    device = pose_2d.device
    
    # 平行移動ベクトルの形状を調整
    if isinstance(t, torch.Tensor) and t.dim() == 2:
        t = t.unsqueeze(-1)  # [B, 3] -> [B, 3, 1]
    
    # 深度方向も含めた同次座標に変換
    pose_image = torch.ones(batch_size, num_joints, 3, dtype=dtype, device=device)
    pose_image[:, :, :2] = pose_2d
    pose_image = pose_image * depth
    
    # 逆変換して3D座標を復元
    pose_3d = []
    for i in range(batch_size):
        # 内部パラメータの逆変換を適用
        K_inv = torch.inverse(K[i])
        pose_camera = torch.matmul(K_inv, pose_image[i].transpose(0, 1))  # [3, 3] @ [3, J] -> [3, J]
        
        # 外部パラメータの逆変換を適用
        RT = torch.cat([R[i], t[i]], dim=1)
        RT_inv = torch.zeros(4, 3, dtype=dtype, device=device)
        R_inv = torch.inverse(R[i])
        RT_inv[:3, :] = R_inv
        RT_inv[3, :] = -torch.matmul(R_inv.transpose(0, 1), t[i]).squeeze()
        
        pose_world = torch.matmul(RT_inv, pose_camera)  # [4, 3] @ [3, J] -> [4, J]
        pose_3d.append(pose_world.transpose(0, 1)[:, :3])  # [J, 3]
    
    pose_3d = torch.stack(pose_3d)
    
    return pose_3d


def get_bone_lengths(pose_3d):
    """
    3Dポーズから骨の長さを計算
    
    Args:
        pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3] または [関節数, 3]
        
    Returns:
        bone_lengths: 各骨の長さ [バッチサイズ, 骨数] または [骨数]
    """
    if isinstance(pose_3d, np.ndarray):
        if pose_3d.ndim == 2:  # [関節数, 3]
            bone_lengths = []
            for joint1, joint2 in JOINT_CONNECTIONS:
                bone_vec = pose_3d[joint2] - pose_3d[joint1]
                bone_length = np.linalg.norm(bone_vec)
                bone_lengths.append(bone_length)
            return np.array(bone_lengths)
        else:  # [バッチサイズ, 関節数, 3]
            batch_bone_lengths = []
            for i in range(pose_3d.shape[0]):
                batch_bone_lengths.append(get_bone_lengths(pose_3d[i]))
            return np.stack(batch_bone_lengths)
    
    elif isinstance(pose_3d, torch.Tensor):
        if pose_3d.dim() == 2:  # [関節数, 3]
            bone_lengths = []
            for joint1, joint2 in JOINT_CONNECTIONS:
                bone_vec = pose_3d[joint2] - pose_3d[joint1]
                bone_length = torch.norm(bone_vec)
                bone_lengths.append(bone_length)
            return torch.stack(bone_lengths)
        else:  # [バッチサイズ, 関節数, 3]
            bone_lengths = []
            for joint1, joint2 in JOINT_CONNECTIONS:
                bone_vec = pose_3d[:, joint2] - pose_3d[:, joint1]
                bone_length = torch.norm(bone_vec, dim=-1)
                bone_lengths.append(bone_length)
            return torch.stack(bone_lengths, dim=1)
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")


def calculate_bone_ratios(pose_3d):
    """
    BONE_PAIRSで指定された骨のペアの長さの比率を計算
    
    Args:
        pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3] または [関節数, 3]
        
    Returns:
        bone_ratios: 骨の長さの比率 [バッチサイズ, ペア数] または [ペア数]
    """
    if isinstance(pose_3d, np.ndarray):
        if pose_3d.ndim == 2:  # [関節数, 3]
            ratios = []
            for (joint1a, joint1b), (joint2a, joint2b) in BONE_PAIRS:
                bone1_vec = pose_3d[joint1b] - pose_3d[joint1a]
                bone2_vec = pose_3d[joint2b] - pose_3d[joint2a]
                
                bone1_length = np.linalg.norm(bone1_vec)
                bone2_length = np.linalg.norm(bone2_vec)
                
                # 0除算を防ぐ
                ratio = bone1_length / (bone2_length + 1e-8)
                ratios.append(ratio)
            
            return np.array(ratios)
        else:  # [バッチサイズ, 関節数, 3]
            batch_ratios = []
            for i in range(pose_3d.shape[0]):
                batch_ratios.append(calculate_bone_ratios(pose_3d[i]))
            return np.stack(batch_ratios)
    
    elif isinstance(pose_3d, torch.Tensor):
        if pose_3d.dim() == 2:  # [関節数, 3]
            ratios = []
            for (joint1a, joint1b), (joint2a, joint2b) in BONE_PAIRS:
                bone1_vec = pose_3d[joint1b] - pose_3d[joint1a]
                bone2_vec = pose_3d[joint2b] - pose_3d[joint2a]
                
                bone1_length = torch.norm(bone1_vec)
                bone2_length = torch.norm(bone2_vec)
                
                # 0除算を防ぐ
                ratio = bone1_length / (bone2_length + 1e-8)
                ratios.append(ratio)
            
            return torch.stack(ratios)
        else:  # [バッチサイズ, 関節数, 3]
            ratios = []
            for (joint1a, joint1b), (joint2a, joint2b) in BONE_PAIRS:
                bone1_vec = pose_3d[:, joint1b] - pose_3d[:, joint1a]
                bone2_vec = pose_3d[:, joint2b] - pose_3d[:, joint2a]
                
                bone1_length = torch.norm(bone1_vec, dim=-1)
                bone2_length = torch.norm(bone2_vec, dim=-1)
                
                # 0除算を防ぐ
                ratio = bone1_length / (bone2_length + 1e-8)
                ratios.append(ratio)
            
            return torch.stack(ratios, dim=1)
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")


def calculate_limbs_angles(pose_3d):
    """
    ポーズの肢の角度を計算し、不自然な曲げを検出
    
    Args:
        pose_3d: 3Dポーズ [バッチサイズ, 関節数, 3] または [関節数, 3]
        
    Returns:
        is_valid: 各肢の曲げが自然かどうかのブール値 [バッチサイズ, 制約肢数] または [制約肢数]
    """
    def compute_plane_normal(pose):
        # 骨盤と脊椎のベクトルから平面の法線を計算
        hip_left = pose[4] - pose[0]
        hip_right = pose[7] - pose[0]
        normal = np.cross(hip_left, hip_right)
        normal = normal / (np.linalg.norm(normal) + 1e-8)  # 正規化
        return normal
    
    if isinstance(pose_3d, np.ndarray):
        if pose_3d.ndim == 2:  # [関節数, 3]
            normal = compute_plane_normal(pose_3d)
            
            is_valid = []
            for proximal, distal in CONSTRAINED_LIMBS:
                # 近位部と遠位部のベクトル
                proximal_vec = pose_3d[proximal] - pose_3d[0]  # 関節から骨盤へ
                distal_vec = pose_3d[distal] - pose_3d[proximal]  # 次の関節へ
                
                # 法線との内積
                proximal_proj = np.dot(normal, proximal_vec)
                distal_proj = np.dot(normal, distal_vec)
                
                # 正の内積は自然な曲げを示す
                valid = proximal_proj <= distal_proj
                is_valid.append(valid)
            
            return np.array(is_valid)
        else:  # [バッチサイズ, 関節数, 3]
            batch_is_valid = []
            for i in range(pose_3d.shape[0]):
                batch_is_valid.append(calculate_limbs_angles(pose_3d[i]))
            return np.stack(batch_is_valid)
    
    elif isinstance(pose_3d, torch.Tensor):
        # PyTorchバージョンの実装（バッチ処理対応）
        if pose_3d.dim() == 2:  # [関節数, 3]
            pose_np = pose_3d.detach().cpu().numpy()
            is_valid_np = calculate_limbs_angles(pose_np)
            return torch.tensor(is_valid_np, dtype=torch.bool, device=pose_3d.device)
        else:  # [バッチサイズ, 関節数, 3]
            batch_size = pose_3d.shape[0]
            device = pose_3d.device
            
            # 各ポーズの法線を計算
            hip_left = pose_3d[:, 4] - pose_3d[:, 0]  # [B, 3]
            hip_right = pose_3d[:, 7] - pose_3d[:, 0]  # [B, 3]
            
            # バッチ化されたクロス積
            normal_x = hip_left[:, 1] * hip_right[:, 2] - hip_left[:, 2] * hip_right[:, 1]
            normal_y = hip_left[:, 2] * hip_right[:, 0] - hip_left[:, 0] * hip_right[:, 2]
            normal_z = hip_left[:, 0] * hip_right[:, 1] - hip_left[:, 1] * hip_right[:, 0]
            normal = torch.stack([normal_x, normal_y, normal_z], dim=1)  # [B, 3]
            
            # 正規化
            normal_norm = torch.norm(normal, dim=1, keepdim=True)
            normal = normal / (normal_norm + 1e-8)  # [B, 3]
            
            is_valid = []
            for proximal, distal in CONSTRAINED_LIMBS:
                # 近位部と遠位部のベクトル
                proximal_vec = pose_3d[:, proximal] - pose_3d[:, 0]  # [B, 3]
                distal_vec = pose_3d[:, distal] - pose_3d[:, proximal]  # [B, 3]
                
                # 法線との内積
                proximal_proj = torch.sum(normal * proximal_vec, dim=1)  # [B]
                distal_proj = torch.sum(normal * distal_vec, dim=1)  # [B]
                
                # 正の内積は自然な曲げを示す
                valid = proximal_proj <= distal_proj
                is_valid.append(valid)
            
            return torch.stack(is_valid, dim=1)  # [B, 制約肢数]
    else:
        raise TypeError("入力はnumpy.ndarrayまたはtorch.Tensorである必要があります")