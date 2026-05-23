import os
import glob
import copy
import numpy as np
import open3d as o3d


class Config:
    # ==========================================
    # 📂 相对路径快速配置区 (统一修改根目录即可)
    # ==========================================
    SRC_DATA_ROOT = "data"            # 原始真值数据根目录
    DST_OUT_ROOT = "output_mse_adam"       # 推理结果输出根目录

    # 真值局部点云路径
    GT_PARTIAL_PLY = os.path.join(SRC_DATA_ROOT, "target/2d_real.ply")
    GT_PARTIAL_TXT = os.path.join(SRC_DATA_ROOT, "target/2d_real.txt")

    # 预测结果所在的文件夹 (自动扫描此目录下的所有 ply 模型)
    PRED_DIR = os.path.join(DST_OUT_ROOT, "pred")
    
    # 综合评估报告输出路径
    METRICS_PATH = os.path.join(DST_OUT_ROOT, "eval_metrics_all.txt")

    # ==========================================
    # ⚙️ 临床可接受的误差阈值 (毫米)
    # ==========================================
    THRESHOLD_STRICT = 2.0
    THRESHOLD_ACCEPTABLE = 5.0


def load_partial_point_cloud():
    """智能加载局部物理点云"""
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


def evaluate_single_mesh(mesh_path, gt_partial_pcd, model_name):
    """
    对单一模型进行误差评估，并生成对应的热力图
    """
    print(f"\n[-] 正在评估模型: {model_name}")
    
    pred_mesh = o3d.io.read_triangle_mesh(mesh_path)
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=50000)

    # 1. 计算距离
    dists = gt_partial_pcd.compute_point_cloud_distance(pred_pcd)
    dists = np.asarray(dists)

    # 2. 统计指标
    mean_dist = np.mean(dists)
    median_dist = np.median(dists)
    rmse_dist = np.sqrt(np.mean(dists ** 2))
    hd95_dist = np.percentile(dists, 95)
    inlier_strict = np.sum(dists < Config.THRESHOLD_STRICT) / len(dists) * 100
    inlier_accept = np.sum(dists < Config.THRESHOLD_ACCEPTABLE) / len(dists) * 100

    report_str = (
        f"--- 模型: {model_name} ---\n"
        f"  Mean Surface Dist : {mean_dist:.4f} mm\n"
        f"  Median Dist       : {median_dist:.4f} mm\n"
        f"  RMSE              : {rmse_dist:.4f} mm\n"
        f"  HD-95             : {hd95_dist:.4f} mm\n"
        f"  Inlier < {Config.THRESHOLD_STRICT}mm  : {inlier_strict:.2f} %\n"
        f"  Inlier < {Config.THRESHOLD_ACCEPTABLE}mm  : {inlier_accept:.2f} %\n"
    )
    print(report_str.strip())

    # 3. 生成专属误差热力图点云
    max_color_error = 10.0
    dists_norm = np.clip(dists / max_color_error, 0, 1)

    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    colors = cmap(dists_norm)[:, :3]
    
    # 为了不污染原始真值点云对象，深拷贝一份用于着色
    heatmap_pcd = copy.deepcopy(gt_partial_pcd)
    heatmap_pcd.colors = o3d.utility.Vector3dVector(colors)

    heatmap_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    heatmap_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))

    heatmap_name = f"heatmap_{model_name}"
    heatmap_path = os.path.join(Config.PRED_DIR, heatmap_name)
    o3d.io.write_point_cloud(heatmap_path, heatmap_pcd, write_ascii=True)
    
    return report_str


def main():
    import copy # 引入 copy 用于深拷贝

    print("=" * 65)
    print("🚀 启动 3D 表面几何批量自动化评估系统")
    print(f"📁 目标检测目录: {Config.PRED_DIR}")
    print("=" * 65)

    if not os.path.exists(Config.PRED_DIR):
        raise FileNotFoundError(f"预测目录不存在: {Config.PRED_DIR}")

    # 加载统一的真实基准表面
    gt_partial_pcd = load_partial_point_cloud()
    print(f"[+] 成功加载真实观测表面，包含 {len(gt_partial_pcd.points)} 个物理点。")

    # 扫描 PRED_DIR 下所有的 .ply 模型文件 (排除掉之前生成的热力图)
    all_ply_files = glob.glob(os.path.join(Config.PRED_DIR, "*.ply"))
    target_models = [f for f in all_ply_files if "heatmap" not in os.path.basename(f)]

    if not target_models:
        print("[!] 未在预测目录中找到任何可以评估的模型 (.ply)。")
        return

    print(f"[+] 发现 {len(target_models)} 个待评估的预测模型。\n")

    # 初始化汇总报告
    full_report = "=================================================\n"
    full_report += "        批量评估综合指标报告 (MSE 权重)        \n"
    full_report += "=================================================\n\n"

    # 开始批量评估
    for model_path in target_models:
        model_name = os.path.basename(model_path)
        
        # 执行单体评估
        report_str = evaluate_single_mesh(model_path, gt_partial_pcd, model_name)
        
        # 记录到总报告
        full_report += report_str + "\n"

    # 将汇总报告落盘
    with open(Config.METRICS_PATH, "w", encoding="utf-8") as f:
        f.write(full_report)
        
    print("=" * 65)
    print(f"✅ 批量评估完毕！所有指标已汇总至: {Config.METRICS_PATH}")
    print("   -> 请使用 CloudCompare 等软件单独查看各个 heatmap_xxx.ply 结果。")
    print("=" * 65)


if __name__ == "__main__":
    main()