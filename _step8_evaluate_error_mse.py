import os
import numpy as np
import open3d as o3d


class Config:
    PRED_PLY = r"output_mse/pred/pred_model_in_cam.ply"

    GT_PARTIAL_PLY = r"data/target/2d_real.ply"
    GT_PARTIAL_TXT = r"data/target/2d_real.txt"

    OUT_DIR = r"output_mse/pred"
    METRICS_PATH = r"output_mse/pred/eval_metrics.txt"

    THRESHOLD_STRICT = 2.0
    THRESHOLD_ACCEPTABLE = 5.0


def load_partial_point_cloud():
    if os.path.exists(Config.GT_PARTIAL_PLY):
        pcd = o3d.io.read_point_cloud(Config.GT_PARTIAL_PLY)
        if len(pcd.points) > 0:
            return pcd

    if os.path.exists(Config.GT_PARTIAL_TXT):
        print(f"[*] 从 TXT 文件加载表面坐标: {Config.GT_PARTIAL_TXT}")
        points = np.loadtxt(Config.GT_PARTIAL_TXT)
        if points.shape[1] >= 3:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points[:, :3])
            return pcd

    raise FileNotFoundError("无法加载真实的局部点云，请检查路径和格式！")


def main():
    print("=" * 65)
    print("启动 MSE 权重结果的 3D 表面几何评估")
    print("=" * 65)

    os.makedirs(Config.OUT_DIR, exist_ok=True)

    if not os.path.exists(Config.PRED_PLY):
        raise FileNotFoundError(f"找不到预测的完整模型: {Config.PRED_PLY}")

    pred_mesh = o3d.io.read_triangle_mesh(Config.PRED_PLY)
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=50000)

    gt_partial_pcd = load_partial_point_cloud()
    print(f"[+] 成功加载真实观测表面，包含 {len(gt_partial_pcd.points)} 个物理点。")

    dists = gt_partial_pcd.compute_point_cloud_distance(pred_pcd)
    dists = np.asarray(dists)

    mean_dist = np.mean(dists)
    median_dist = np.median(dists)
    rmse_dist = np.sqrt(np.mean(dists ** 2))
    hd95_dist = np.percentile(dists, 95)
    inlier_strict = np.sum(dists < Config.THRESHOLD_STRICT) / len(dists) * 100
    inlier_accept = np.sum(dists < Config.THRESHOLD_ACCEPTABLE) / len(dists) * 100

    report = (
        "[局部表面几何误差统计 - MSE 权重]\n"
        f"Mean Surface Dist : {mean_dist:.4f} mm\n"
        f"Median Dist       : {median_dist:.4f} mm\n"
        f"RMSE              : {rmse_dist:.4f} mm\n"
        f"HD-95             : {hd95_dist:.4f} mm\n"
        f"Inlier < {Config.THRESHOLD_STRICT}mm  : {inlier_strict:.2f} %\n"
        f"Inlier < {Config.THRESHOLD_ACCEPTABLE}mm  : {inlier_accept:.2f} %\n"
    )
    print("\n" + report)
    with open(Config.METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[+] 评估指标已保存至: {Config.METRICS_PATH}")

    max_color_error = 10.0
    dists_norm = np.clip(dists / max_color_error, 0, 1)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    colors = cmap(dists_norm)[:, :3]
    gt_partial_pcd.colors = o3d.utility.Vector3dVector(colors)

    print("    -> 正在注入法向量特征...")
    gt_partial_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    gt_partial_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    heatmap_path = os.path.join(Config.OUT_DIR, "error_heatmap.ply")
    o3d.io.write_point_cloud(heatmap_path, gt_partial_pcd, write_ascii=True)
    print(f"[+] 误差热力图已保存至: {heatmap_path}")

    print("    -> 正在启动交互式 3D 窗口 (关闭窗口以结束程序)...")
    pred_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    o3d.visualization.draw_geometries(
        [pred_mesh, gt_partial_pcd],
        window_name="MSE Result Surface Error Heatmap (Blue=Good, Red=Bad)",
        mesh_show_back_face=True
    )


if __name__ == "__main__":
    main()
