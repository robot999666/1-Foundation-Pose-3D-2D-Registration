import os
import cv2
import open3d as o3d
import numpy as np
import copy
import torch
import torch.nn.functional as F

# 导入你的网络模型
from learning.models.refine_network import RefineNet

# ==========================================
# 1. 核心配置与路径
# ==========================================
class Config:
    TEST_DIR = r"data/test"
    MODEL_PLY = r"data/ori/Model_obj.ply"
    WEIGHT_PATH = r"result/best.pth"
    
    T_VIEW_PATH = r"data/ori/T_view.txt" 
    CAM_INTRINSICS_PATH = r"data/test/camera_new.txt"
    
    OUT_PRED_DIR = r"data/pred"
    OUT_DEBUG_DIR = r"data/debug"
    
    # 【新增】：迭代阻尼控制 (防止步子迈太大扯到蛋)
    # 0.3 意味着每次只执行网络预测修正量的 30%
    DAMPING_T = 1    # 平移阻尼
    DAMPING_R = 1    # 旋转阻尼
    
    MAX_ITER = 15       # 因为加了阻尼步子变小了，迭代次数可以适当放宽到 15 次
    STOP_TRANS_THRES = 1.0       # 停止阈值：1.0 mm 
    TRANS_SCALE = 50.0           # 必须与训练时保持绝对一致
    NETWORK_IN_SIZE = (480, 270) # 网络输入的低分辨率
    RENDER_SIZE = (960, 540)     # 物理渲染的高分辨率

os.makedirs(Config.OUT_PRED_DIR, exist_ok=True)
os.makedirs(Config.OUT_DEBUG_DIR, exist_ok=True)

# ==========================================
# 2. 严密数学与物理渲染工具函数
# ==========================================
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

    # 【修复 1：Depth 背景一致性】
    # step4/step5 生成的真实 B 与训练 A/B 都使用背景=1。
    # 旧版推理迭代渲染 A 使用背景=0，会导致第 0 轮到第 1 轮输入分布突变，卷积特征被“反相”。
    # 因此这里统一初始化为 1，只在肾脏有效区域写入 0~1 的 Z-depth 归一化值。
    depth_norm = np.ones_like(depth, dtype=np.float32)
    valid = depth > 0
    if valid.any():
        # 【修复 2：统一选择 Z-Depth】
        # 本函数光线方向保持未归一化，Open3D 返回的 t_hit 等价于相机坐标系 Z 方向深度。
        # 与 step5 的训练渲染保持一致，避免训练看 radial distance、推理看 z-depth 的空间扭曲。
        depth_norm[valid] = (depth[valid] - depth[valid].min()) / (depth[valid].max() - depth[valid].min() + 1e-5)

    return contour_img, mask, depth_norm

# ==========================================
# 3. 核心推理引擎
# ==========================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] 初始化带有阻尼系统的医疗级推理引擎，设备: {device}")

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

    # 【打通训练-推理回路】
    # 第一轮 A 不再读取可能过期的 data/base_pose，而是直接由当前 T_curr 在线渲染。
    # 这与 step6 中 30% base-pose 分支、70% local-refinement 分支的在线渲染机制保持一致。
    a_contour, a_mask, a_depth = render_kidney(
        mesh, K, T_curr,
        width=Config.RENDER_SIZE[0],
        height=Config.RENDER_SIZE[1]
    )

    print("\n" + "="*65)
    print("🚀 开始 3D-2D 术中位姿迭代细化 (Pose Refinement)...")
    print(f"📉 当前阻尼设置 -> 平移阻尼: {Config.DAMPING_T}, 旋转阻尼: {Config.DAMPING_R}")
    print("="*65)

    with torch.no_grad():
        for i in range(Config.MAX_ITER):
            tensor_A = pack_to_tensor(a_contour, a_depth, a_mask, Config.NETWORK_IN_SIZE).to(device)

            out = model(tensor_A, tensor_B)
            
            # 获取原始的预测值
            R_pred_raw = compute_rotation_matrix_from_6d(out['rot'])[0].cpu().numpy()
            t_pred_raw = (out['trans'][0].cpu().numpy() * Config.TRANS_SCALE).reshape(3, 1)
            
            # ----------------------------------------------------
            # 【核心功能】：阻尼缩放系统 (Damping)
            # ----------------------------------------------------
            # 1. 缩放平移向量
            t_pred_damped = t_pred_raw * Config.DAMPING_T
            
            # 2. 缩放旋转矩阵 (通过 Rodrigues 转换为轴角，缩放角度后再转回矩阵)
            rvec, _ = cv2.Rodrigues(R_pred_raw)         # rvec 的方向是轴，模长是旋转角度(弧度)
            rvec_damped = rvec * Config.DAMPING_R       # 缩小旋转角度
            R_pred_damped, _ = cv2.Rodrigues(rvec_damped)
            # ----------------------------------------------------

            # 组装阻尼后的 4x4 修正矩阵
            delta_T = np.eye(4, dtype=np.float32)
            delta_T[:3, :3] = R_pred_damped
            delta_T[:3, 3:4] = t_pred_damped

            # 计算回显指标 (展示实际执行的修正量)
            trans_mag = np.linalg.norm(t_pred_damped)
            rot_angle_deg = np.linalg.norm(rvec_damped) * 180.0 / np.pi
            
            print(f"[Iter {i:02d}] 实际执行修正 -> 平移: {trans_mag:6.3f} mm | 旋转: {rot_angle_deg:5.2f} 度")

            # 矩阵更新
            T_curr = T_curr @ delta_T

            # 渲染闭环
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

            # 终止条件
            if trans_mag < Config.STOP_TRANS_THRES:
                print(f"\n✅ [系统提示] 修正幅度低于安全阈值，模型完美收敛。")
                break
        else:
            print(f"\n⚠️ [系统提示] 达到最大迭代次数 ({Config.MAX_ITER})。")

    print("\n" + "="*65)
    print("💾 正在落盘最终医疗数据...")
    
    final_pose_path = os.path.join(Config.OUT_PRED_DIR, "final_pose.txt")
    np.savetxt(final_pose_path, T_curr, fmt="%.6f")
    print(f" [+] 最终位姿矩阵 (T_est) 已保存至: {final_pose_path}")

    mesh.transform(T_curr)
    final_ply_path = os.path.join(Config.OUT_PRED_DIR, "pred_model_in_cam.ply")
    o3d.io.write_triangle_mesh(final_ply_path, mesh)
    print(f" [+] 最终 3D 姿态点云已保存至: {final_ply_path}")
    print("🎉 推理管线执行完毕！")

if __name__ == "__main__":
    main()