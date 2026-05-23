import open3d as o3d
import numpy as np
import cv2
import os
import math
import json
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


def load_camera_matrix(filepath):
    K = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            K.append([float(x) for x in line.split()])
    return np.array(K, dtype=np.float64)


def euler_to_matrix(rx, ry, rz):
    rx, ry, rz = math.radians(rx), math.radians(ry), math.radians(rz)
    R_x = np.array([[1, 0, 0], [0, math.cos(rx), -math.sin(rx)], [0, math.sin(rx), math.cos(rx)]], dtype=np.float32)
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)], [0, 1, 0], [-math.sin(ry), 0, math.cos(ry)]], dtype=np.float32)
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0], [math.sin(rz), math.cos(rz), 0], [0, 0, 1]], dtype=np.float32)
    return R_z @ R_y @ R_x


def generate_rays(K, T_c2w, w, h):
    """生成非单位化 pinhole rays，使 t_hit 表示 Z-depth。"""
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)

    R = T_c2w[:3, :3]
    dirs_world = dirs @ R.T
    origins = np.broadcast_to(T_c2w[:3, 3], dirs_world.shape)

    rays = np.concatenate([origins, dirs_world], axis=-1).astype(np.float32)
    return o3d.core.Tensor(rays)


def elastic_transform(image, alpha=4, sigma=3):
    random_state = np.random.RandomState(None)
    shape = image.shape
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)
    return cv2.remap(image, map_x, map_y, interpolation=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def augment_mask(mask):
    if np.random.rand() > 0.5:
        return mask
    op = np.random.choice([cv2.MORPH_ERODE, cv2.MORPH_DILATE])
    k_size = np.random.choice([3, 5])
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    return cv2.morphologyEx(mask, op, kernel)


def extract_and_augment_contour(mask, is_base_pose=False):
    kernel = np.ones((3, 3), np.uint8)
    contour = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel)

    if is_base_pose:
        cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        contour = cv2.dilate(contour, cross_kernel, iterations=1)
    else:
        contour = elastic_transform(contour)
        if np.random.rand() < 0.4:
            h, w = contour.shape
            bw, bh = np.random.randint(10, 40), np.random.randint(10, 40)
            bx, by = np.random.randint(0, max(1, w - bw)), np.random.randint(0, max(1, h - bh))
            contour[by:by + bh, bx:bx + bw] = 0
        if np.random.rand() < 0.5:
            cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
            contour = cv2.dilate(contour, cross_kernel, iterations=np.random.randint(1, 3))
    return contour


def augment_and_normalize_depth(depth_raw, mask):
    valid_pixels = mask > 127
    if not np.any(valid_pixels):
        return np.ones_like(depth_raw, dtype=np.float32), np.zeros((depth_raw.shape[0], depth_raw.shape[1], 3), dtype=np.uint8)

    target_depths = depth_raw[valid_pixels].copy()
    if np.random.rand() < 0.6:
        s = np.random.uniform(0.8, 1.2)
        delta = np.random.uniform(-15, 15)
        noise = np.random.normal(0, np.random.uniform(0.01, 0.03) * 100, size=target_depths.shape)
        target_depths = s * target_depths + delta + noise

    d_min, d_max = target_depths.min(), target_depths.max()
    if d_max - d_min < 1e-3:
        norm_depths = np.zeros_like(target_depths)
    else:
        norm_depths = (target_depths - d_min) / (d_max - d_min)

    depth_norm = np.ones_like(depth_raw, dtype=np.float32)
    depth_norm[valid_pixels] = norm_depths

    if np.random.rand() < 0.3:
        h, w = depth_raw.shape
        bw, bh = np.random.randint(20, 60), np.random.randint(20, 60)
        bx, by = np.random.randint(0, max(1, w - bw)), np.random.randint(0, max(1, h - bh))
        depth_norm[by:by + bh, bx:bx + bw] = 1.0

    depth_uint8 = (depth_norm * 255).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    depth_vis[depth_norm == 1.0] = [0, 0, 0]
    return depth_norm, depth_vis


def generate_dataset():
    print("=== 第五阶段：离线生成 A/B Pair 数据集 ===")

    K_PATH = r"data/test/camera_new.txt"
    OBJ_PATH = r"data/ori/Model_obj.ply"
    TVIEW_PATH = r"data/ori/T_view.txt"

    BASE_DIR = r"data/base_pose"
    TRAIN_DIR = r"data/sample/train/images"
    VAL_DIR = r"data/sample/val/images"
    os.makedirs(BASE_DIR, exist_ok=True)
    os.makedirs(TRAIN_DIR, exist_ok=True)
    os.makedirs(VAL_DIR, exist_ok=True)

    w, h = 960, 540
    total_pixels = w * h
    K = load_camera_matrix(K_PATH)
    T_view = np.loadtxt(TVIEW_PATH).astype(np.float32)

    mesh_legacy = o3d.io.read_triangle_mesh(OBJ_PATH)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_legacy)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    NUM_PAIRS = 10000
    TRAIN_RATIO = 0.9
    MIN_RATIO = 0.20
    MAX_RATIO = 0.90
    FIRST_ROUND_RATIO = 0.30

    train_labels, val_labels = {}, {}
    rejected_count = 0

    def render_state_from_delta(delta_T):
        T_render = T_view @ delta_T.astype(np.float32)
        T_c2w = np.linalg.inv(T_render)
        rays = generate_rays(K, T_c2w, w, h)
        ans = scene.cast_rays(rays)
        depth = ans['t_hit'].numpy()
        mask = (depth < np.inf).astype(np.uint8) * 255
        depth_raw = np.where(depth < np.inf, depth, 0.0).astype(np.float32)
        return T_render, depth_raw, mask

    def sample_global_delta():
        rx, ry, rz = np.random.uniform(-20, 20, 3)
        tx, ty = np.random.uniform(-30, 30, 2)
        tz = np.random.uniform(-30, 30)
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = euler_to_matrix(rx, ry, rz)
        delta[:3, 3] = [tx, ty, tz]
        return delta

    def sample_local_delta():
        rx, ry, rz = np.random.uniform(-3, 3, 3)
        tx, ty, tz = np.random.uniform(-5, 5, 3)
        delta = np.eye(4, dtype=np.float32)
        delta[:3, :3] = euler_to_matrix(rx, ry, rz)
        delta[:3, 3] = [tx, ty, tz]
        return delta

    def valid_ratio(mask):
        ratio = np.count_nonzero(mask) / total_pixels
        return MIN_RATIO <= ratio <= MAX_RATIO, ratio

    def save_triplet(save_dir, prefix, role, mask, contour, depth, depth_vis=None):
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{role}_mask.png"), mask)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_{role}_contour.png"), contour)
        np.save(os.path.join(save_dir, f"{prefix}_{role}_depth.npy"), depth)
        if depth_vis is not None:
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_{role}_depth_vis.png"), depth_vis)

    print("\n[1/3] 生成并保存 Base Pose...")
    base_delta = np.eye(4, dtype=np.float32)
    _, base_depth_raw, base_mask = render_state_from_delta(base_delta)
    ok_base, base_ratio = valid_ratio(base_mask)
    print(f"  -> Base Pose 画面占比: {base_ratio * 100:.2f}%")
    if not ok_base:
        print("  -> [警告] Base Pose 占比超出 20%~90%，请检查 T_view。")
    base_contour = extract_and_augment_contour(base_mask, is_base_pose=True)
    base_depth, base_vis = augment_and_normalize_depth(base_depth_raw, base_mask)
    cv2.imwrite(os.path.join(BASE_DIR, "mask.png"), base_mask)
    cv2.imwrite(os.path.join(BASE_DIR, "contour.png"), base_contour)
    np.save(os.path.join(BASE_DIR, "depth.npy"), base_depth)
    cv2.imwrite(os.path.join(BASE_DIR, "depth_vis.png"), base_vis)

    print(f"\n[2/3] 离线生成 Pair 数据集 (目标: {NUM_PAIRS} 对)...")
    pbar = tqdm(total=NUM_PAIRS, desc="Generating Pairs")
    i = 0
    while i < NUM_PAIRS:
        pair_type = "base_to_target" if np.random.rand() < FIRST_ROUND_RATIO else "local_refine"

        if pair_type == "base_to_target":
            delta_A = np.eye(4, dtype=np.float32)
            delta_B = sample_global_delta()
        else:
            delta_B = sample_global_delta()
            delta_small = sample_local_delta()
            delta_A = delta_B @ delta_small

        T_render_A, depth_raw_A, mask_raw_A = render_state_from_delta(delta_A)
        T_render_B, depth_raw_B, mask_raw_B = render_state_from_delta(delta_B)

        ok_A, ratio_A = valid_ratio(mask_raw_A)
        ok_B, ratio_B = valid_ratio(mask_raw_B)
        if not (ok_A and ok_B):
            rejected_count += 1
            continue

        mask_A = base_mask.copy() if pair_type == "base_to_target" else augment_mask(mask_raw_A)
        contour_A = base_contour.copy() if pair_type == "base_to_target" else extract_and_augment_contour(mask_A, is_base_pose=False)
        depth_A, vis_A = base_depth.copy(), base_vis.copy()
        if pair_type != "base_to_target":
            depth_A, vis_A = augment_and_normalize_depth(depth_raw_A, mask_A)

        mask_B = augment_mask(mask_raw_B)
        contour_B = extract_and_augment_contour(mask_B, is_base_pose=False)
        depth_B, vis_B = augment_and_normalize_depth(depth_raw_B, mask_B)

        is_train = np.random.rand() < TRAIN_RATIO
        save_dir = TRAIN_DIR if is_train else VAL_DIR
        labels_dict = train_labels if is_train else val_labels

        prefix = f"{i:05d}"
        save_triplet(save_dir, prefix, "A", mask_A, contour_A, depth_A, vis_A if i % 1000 == 0 else None)
        save_triplet(save_dir, prefix, "B", mask_B, contour_B, depth_B, vis_B if i % 1000 == 0 else None)

        labels_dict[prefix] = {
            "pair_type": pair_type,
            "Delta_A": delta_A.tolist(),
            "Delta_B": delta_B.tolist(),
            "T_render_A": T_render_A.tolist(),
            "T_render_B": T_render_B.tolist(),
            "mask_ratio_A": float(ratio_A),
            "mask_ratio_B": float(ratio_B)
        }

        i += 1
        pbar.update(1)

    pbar.close()

    print("\n[3/3] 保存标签文件...")
    os.makedirs(r"data/sample/train", exist_ok=True)
    os.makedirs(r"data/sample/val", exist_ok=True)
    with open(r"data/sample/train/labels.json", "w", encoding="utf-8") as f:
        json.dump(train_labels, f)
    with open(r"data/sample/val/labels.json", "w", encoding="utf-8") as f:
        json.dump(val_labels, f)

    print("\n[大功告成] Pair 数据集构建完毕！")
    print(f"  -> 成功采纳: {NUM_PAIRS} 对样本")
    print(f"  -> 拒绝拦截: {rejected_count} 对")
    print(f"  -> 训练集: {len(train_labels)} 对")
    print(f"  -> 验证集: {len(val_labels)} 对")


if __name__ == "__main__":
    generate_dataset()
