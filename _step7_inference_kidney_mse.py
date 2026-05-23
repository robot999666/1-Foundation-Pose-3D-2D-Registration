import os
import cv2
import open3d as o3d
import numpy as np
import copy
import torch
import torch.nn.functional as F

from learning.models.refine_network import RefineNet


class Config:
    TEST_DIR = r"data/test"
    MODEL_PLY = r"data/ori/Model_obj.ply"
    WEIGHT_PATH = r"output_mse/best.pth"

    T_VIEW_PATH = r"data/ori/T_view.txt"
    CAM_INTRINSICS_PATH = r"data/test/camera_new.txt"

    OUT_ROOT = r"output_mse"
    OUT_PRED_DIR = r"output_mse/pred"
    OUT_DEBUG_DIR = r"output_mse/debug"

    DAMPING_T = 1
    DAMPING_R = 1

    MAX_ITER = 15
    STOP_TRANS_THRES = 1.0
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
    print("开始使用 MSE 训练权重执行 3D-2D 术中位姿迭代细化")
    print(f"输出目录: {Config.OUT_ROOT}")
    print("=" * 65)

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

            if trans_mag < Config.STOP_TRANS_THRES:
                print("\n[系统提示] 修正幅度低于安全阈值，停止迭代。")
                break
        else:
            print(f"\n[系统提示] 达到最大迭代次数 ({Config.MAX_ITER})。")

    final_pose_path = os.path.join(Config.OUT_PRED_DIR, "final_pose.txt")
    np.savetxt(final_pose_path, T_curr, fmt="%.6f")
    print(f"[+] 最终位姿矩阵已保存至: {final_pose_path}")

    mesh.transform(T_curr)
    final_ply_path = os.path.join(Config.OUT_PRED_DIR, "pred_model_in_cam.ply")
    o3d.io.write_triangle_mesh(final_ply_path, mesh)
    print(f"[+] 最终 3D 姿态模型已保存至: {final_ply_path}")


if __name__ == "__main__":
    main()
