import os
import cv2
import open3d as o3d
import numpy as np
import copy
import torch
import torch.nn.functional as F

from learning.models.refine_network import RefineNet

class Config:
    # ==========================================
    # 📂 相对路径快速配置区 (修改这里即可全局生效)
    # ==========================================
    SRC_DATA_ROOT = "data"            # 数据输入根目录
    DST_OUT_ROOT = "output_mse_adam"       # 推理输出根目录

    TEST_DIR = os.path.join(SRC_DATA_ROOT, "test")
    MODEL_PLY = os.path.join(SRC_DATA_ROOT, "ori/Model_obj.ply")
    T_VIEW_PATH = os.path.join(SRC_DATA_ROOT, "ori/T_view.txt")
    CAM_INTRINSICS_PATH = os.path.join(TEST_DIR, "camera_new.txt")

    WEIGHT_PATH = os.path.join(DST_OUT_ROOT, "best.pth")
    OUT_PRED_DIR = os.path.join(DST_OUT_ROOT, "pred")
    OUT_DEBUG_DIR = os.path.join(DST_OUT_ROOT, "debug")

    # ==========================================
    # ⚙️ 迭代控制与终止策略参数
    # ==========================================
    DAMPING_T = 1.0
    DAMPING_R = 1.0

    MAX_ITER = 15            # 最大迭代上限
    MIN_ITER = 10            # 强制最少迭代下限 (防止过早停止)
    
    # 【双重安全阈值】必须同时满足才允许停止
    STOP_TRANS_THRES = 1.0   # 平移变化小于 1.0 mm
    STOP_ROT_THRES = 1.0     # 旋转变化小于 1.0 度

    SAVE_TOP_K = 3           # 自动保存收敛最好的前 K 个姿态
    
    TRANS_SCALE = 50.0
    NETWORK_IN_SIZE = (480, 270)
    RENDER_SIZE = (960, 540)

os.makedirs(Config.OUT_PRED_DIR, exist_ok=True)
os.makedirs(Config.OUT_DEBUG_DIR, exist_ok=True)


def compute_rotation_matrix_from_6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = F.normalize(x_raw, p=2, dim=-1)
    z_dot_x = (y_raw * x).sum(dim=-1, keepdim=True)
    y = F.normalize(y_raw - z_dot_x * x, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)

def load_matrix(path, shape=(3, 3)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"致命错误：找不到关键矩阵文件 {path}！")
    mat = np.loadtxt(path)
    if mat.size == np.prod(shape):
        mat = mat.reshape(shape)
    return mat

def pack_to_tensor(contour, depth, mask, target_size):
    c = cv2.resize(contour, target_size, interpolation=cv2.INTER_NEAREST)
    m = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    d = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)

    c = (c / 255.0).astype(np.float32)
    m = (m / 255.0).astype(np.float32)
    d = d.astype(np.float32)
    return torch.from_numpy(np.stack([c, d, m], axis=0)).unsqueeze(0)

def render_kidney(mesh, K, T, width, height):
    mesh_copy = copy.deepcopy(mesh)
    mesh_copy.transform(T)
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh_copy)

    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    cx, cy = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]

    u, v = np.meshgrid(np.arange(width), np.arange(height))
    dirs = np.stack([(u - cx) / fx, (v - cy) / fy, np.ones_like(u)], axis=-1)

    rays = np.zeros((height, width, 6), dtype=np.float32)
    rays[:, :, 3:] = dirs

    rays_t = o3d.core.Tensor(rays)
    ans = scene.cast_rays(rays_t)
    depth = ans['t_hit'].numpy()
    depth[depth == np.inf] = 0

    mask = (depth > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = np.zeros_like(mask)
    cv2.drawContours(contour_img, contours, -1, 255, 2)

    # 深度背景值初始化为 1
    depth_norm = np.ones_like(depth, dtype=np.float32)
    valid = depth > 0
    if valid.any():
        depth_norm[valid] = (depth[valid] - depth[valid].min()) / (depth[valid].max() - depth[valid].min() + 1e-5)

    return contour_img, mask, depth_norm


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 初始化 MSE 权重推理引擎，设备: {device}")

    class DummyCfg:
        use_BN = True
        rot_rep = "6d"

    model = RefineNet(cfg=DummyCfg(), c_in=3).to(device)
    model.load_state_dict(torch.load(Config.WEIGHT_PATH, map_location=device))
    model.eval()

    mesh = o3d.io.read_triangle_mesh(Config.MODEL_PLY)
    K = load_matrix(Config.CAM_INTRINSICS_PATH, shape=(3, 3))
    T_curr = load_matrix(Config.T_VIEW_PATH, shape=(4, 4)).astype(np.float32)

    b_contour = cv2.imread(os.path.join(Config.TEST_DIR, "contour.png"), cv2.IMREAD_GRAYSCALE)
    b_mask = cv2.imread(os.path.join(Config.TEST_DIR, "mask.png"), cv2.IMREAD_GRAYSCALE)
    b_depth = np.load(os.path.join(Config.TEST_DIR, "depth_normalized.npy"))
    tensor_B = pack_to_tensor(b_contour, b_depth, b_mask, Config.NETWORK_IN_SIZE).to(device)

    a_contour, a_mask, a_depth = render_kidney(
        mesh, K, T_curr,
        width=Config.RENDER_SIZE[0],
        height=Config.RENDER_SIZE[1]
    )

    print("\n" + "=" * 65)
    print("🚀 开始使用 MSE 训练权重执行 3D-2D 术中位姿迭代细化")
    print(f"📁 数据源: {Config.SRC_DATA_ROOT} | 输出目录: {Config.DST_OUT_ROOT}")
    print("=" * 65)

    # 用于记录历次迭代的收敛状态
    pose_history = []

    with torch.no_grad():
        for i in range(Config.MAX_ITER):
            tensor_A = pack_to_tensor(a_contour, a_depth, a_mask, Config.NETWORK_IN_SIZE).to(device)
            out = model(tensor_A, tensor_B)

            R_pred_raw = compute_rotation_matrix_from_6d(out['rot'])[0].cpu().numpy()
            t_pred_raw = (out['trans'][0].cpu().numpy() * Config.TRANS_SCALE).reshape(3, 1)

            t_pred_damped = t_pred_raw * Config.DAMPING_T
            rvec, _ = cv2.Rodrigues(R_pred_raw)
            rvec_damped = rvec * Config.DAMPING_R
            R_pred_damped, _ = cv2.Rodrigues(rvec_damped)

            delta_T = np.eye(4, dtype=np.float32)
            delta_T[:3, :3] = R_pred_damped
            delta_T[:3, 3:4] = t_pred_damped

            trans_mag = np.linalg.norm(t_pred_damped)
            rot_angle_deg = np.linalg.norm(rvec_damped) * 180.0 / np.pi
            
            print(f"[Iter {i:02d}] 实际执行修正 -> 平移: {trans_mag:6.3f} mm | 旋转: {rot_angle_deg:5.2f} 度")

            T_curr = T_curr @ delta_T

            # --- 记录当前轮次信息 ---
            # 收敛分数：用平移(mm) + 旋转(度) 的简单求和作为“不稳定度”。越小说明网络认为越贴合。
            convergence_score = trans_mag + rot_angle_deg 
            pose_history.append({
                'iter': i,
                'pose': T_curr.copy(),
                'trans_mag': trans_mag,
                'rot_deg': rot_angle_deg,
                'score': convergence_score
            })

            # --- 渲染 Debug 图片 ---
            iter_dir = os.path.join(Config.OUT_DEBUG_DIR, f"iter_{i:02d}")
            os.makedirs(iter_dir, exist_ok=True)

            a_contour, a_mask, a_depth = render_kidney(
                mesh, K, T_curr,
                width=Config.RENDER_SIZE[0],
                height=Config.RENDER_SIZE[1]
            )

            cv2.imwrite(os.path.join(iter_dir, "contour.png"), a_contour)
            cv2.imwrite(os.path.join(iter_dir, "mask.png"), a_mask)
            np.save(os.path.join(iter_dir, "depth.npy"), a_depth)

            overlay = cv2.cvtColor(b_mask, cv2.COLOR_GRAY2BGR)
            overlay[a_contour > 0] = [0, 0, 255]
            cv2.imwrite(os.path.join(iter_dir, "overlay_visual.png"), overlay)

            # --- 全新的双重条件停止逻辑 ---
            if i >= (Config.MIN_ITER - 1): # 确保执行至少 MIN_ITER 次 (因为 i 从 0 开始)
                if trans_mag < Config.STOP_TRANS_THRES and rot_angle_deg < Config.STOP_ROT_THRES:
                    print(f"\n✅ [系统提示] 已达到最低迭代下限，且平移({trans_mag:.2f}mm)与旋转({rot_angle_deg:.2f}°)均低于安全阈值，完美收敛。")
                    break
        else:
            print(f"\n⚠️ [系统提示] 达到最大迭代次数 ({Config.MAX_ITER})。")

    # ==========================================
    # 💾 数据落盘与收敛排行
    # ==========================================
    print("\n" + "=" * 65)
    print("💾 正在保存最终结果与最优 Top-K 结果...")
    
    # 1. 保存最终一轮的结果 (Final)
    final_info = pose_history[-1]
    final_pose_path = os.path.join(Config.OUT_PRED_DIR, "final_pose.txt")
    np.savetxt(final_pose_path, final_info['pose'], fmt="%.6f")
    
    final_mesh = copy.deepcopy(mesh)
    final_mesh.transform(final_info['pose'])
    final_ply_path = os.path.join(Config.OUT_PRED_DIR, "pred_model_in_cam_final.ply")
    o3d.io.write_triangle_mesh(final_ply_path, final_mesh)
    print(f" [Final] 最终迭代结果 (Iter {final_info['iter']}) 已保存至: pred_model_in_cam_final.ply")

    # 2. 对历史记录按照 score (修正幅度) 进行升序排序，挑选最好的一批
    sorted_history = sorted(pose_history, key=lambda x: x['score'])
    
    for k in range(min(Config.SAVE_TOP_K, len(sorted_history))):
        best_info = sorted_history[k]
        rank = k + 1
        print(f" [Top {rank}] 来自 Iter {best_info['iter']:02d} | 修正步长 -> 平移 {best_info['trans_mag']:.3f}mm, 旋转 {best_info['rot_deg']:.3f}°")
        
        # 写入矩阵
        best_pose_path = os.path.join(Config.OUT_PRED_DIR, f"best_rank{rank}_iter{best_info['iter']:02d}_pose.txt")
        np.savetxt(best_pose_path, best_info['pose'], fmt="%.6f")
        
        # 写入点云
        best_mesh = copy.deepcopy(mesh)
        best_mesh.transform(best_info['pose'])
        best_ply_path = os.path.join(Config.OUT_PRED_DIR, f"best_rank{rank}_iter{best_info['iter']:02d}_model.ply")
        o3d.io.write_triangle_mesh(best_ply_path, best_mesh)

    print("\n🎉 推理管线执行完毕！")

if __name__ == "__main__":
    main()