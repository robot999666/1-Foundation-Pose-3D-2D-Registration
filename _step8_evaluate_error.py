import os
import numpy as np
import open3d as o3d

class Config:
    # 你的预测结果 (完整的 3D 模型)
    PRED_PLY = r"data/pred/pred_model_in_cam.ply"
    
    # 真实的术中局部点云
    GT_PARTIAL_PLY = r"data/target/2d_real.ply"
    GT_PARTIAL_TXT = r"data/target/2d_real.txt" # 备用路径
    
    # 临床可接受的误差阈值 (毫米)
    THRESHOLD_STRICT = 2.0
    THRESHOLD_ACCEPTABLE = 5.0

def load_partial_point_cloud():
    """智能加载局部点云，优先 ply，其次 txt"""
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
    print("="*65)
    print("🔬 启动医疗级 3D 表面几何评估系统 (高兼容版)")
    print("="*65)

    # ==========================================
    # 1. 加载数据
    # ==========================================
    if not os.path.exists(Config.PRED_PLY):
        raise FileNotFoundError(f"找不到预测的完整模型: {Config.PRED_PLY}")

    pred_mesh = o3d.io.read_triangle_mesh(Config.PRED_PLY)
    # 将网络预测的网格转换为密集点云（采样 5 万个点）
    pred_pcd = pred_mesh.sample_points_uniformly(number_of_points=50000)
    
    gt_partial_pcd = load_partial_point_cloud()
    print(f"[+] 成功加载真实观测表面，包含 {len(gt_partial_pcd.points)} 个物理点。")

    # ==========================================
    # 2. 计算单向表面距离 (Real Surface -> Predicted Model)
    # ==========================================
    dists = gt_partial_pcd.compute_point_cloud_distance(pred_pcd)
    dists = np.asarray(dists)

    # ==========================================
    # 3. 统计指标计算
    # ==========================================
    mean_dist = np.mean(dists)
    median_dist = np.median(dists)
    rmse_dist = np.sqrt(np.mean(dists**2))
    hd95_dist = np.percentile(dists, 95)
    inlier_strict = np.sum(dists < Config.THRESHOLD_STRICT) / len(dists) * 100
    inlier_accept = np.sum(dists < Config.THRESHOLD_ACCEPTABLE) / len(dists) * 100

    print("\n[📊 局部表面几何误差统计]")
    print(f" ├─ 平均距离 (Mean Surface Dist) : {mean_dist:.4f} mm")
    print(f" ├─ 中位数距离 (Median Dist)     : {median_dist:.4f} mm")
    print(f" ├─ 均方根误差 (RMSE)            : {rmse_dist:.4f} mm")
    print(f" ├─ 豪斯多夫 95% 距离 (HD-95)    : {hd95_dist:.4f} mm  <-- (最重要指标)")
    print(f" ├─ 完美匹配率 (误差 < {Config.THRESHOLD_STRICT}mm)   : {inlier_strict:.2f} %")
    print(f" └─ 临床可用率 (误差 < {Config.THRESHOLD_ACCEPTABLE}mm)   : {inlier_accept:.2f} %")
    print("="*65)

    # ==========================================
    # 4. 生成高兼容性误差热力图点云
    # ==========================================
    max_color_error = 10.0 # 10mm 以上全红
    dists_norm = np.clip(dists / max_color_error, 0, 1)
    
    import matplotlib.pyplot as plt
    cmap = plt.get_cmap("jet")
    colors = cmap(dists_norm)[:, :3] 
    
    # 赋予颜色
    gt_partial_pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # 【核心修复 1】：计算法向量，让光照系统能识别它
    print("    -> 正在注入法向量特征以突破渲染器限制...")
    gt_partial_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    # 将法向统一指向相机视角，防止背面剔除 (Backface Culling) 导致部分点隐形
    gt_partial_pcd.orient_normals_towards_camera_location(camera_location=np.array([0., 0., 0.]))
    
    heatmap_path = r"data/pred/error_heatmap.ply"
    
    # 【核心修复 2】：使用 write_ascii=True 写入明文格式，兼容所有 3D 软件
    o3d.io.write_point_cloud(heatmap_path, gt_partial_pcd, write_ascii=True)
    print(f"[+] 高兼容版误差热力图已保存至: {heatmap_path}")

    # ==========================================
    # 5. 代码内直接弹窗预览
    # ==========================================
    print("    -> 正在启动交互式 3D 窗口 (关闭窗口以结束程序)...")
    pred_mesh.paint_uniform_color([0.8, 0.8, 0.8])
    
    # 修改了渲染参数，使其更清晰
    o3d.visualization.draw_geometries(
        [pred_mesh, gt_partial_pcd], 
        window_name="Clinical Surface Error Heatmap (Blue=Good, Red=Bad)",
        mesh_show_back_face=True
    )

if __name__ == "__main__":
    main()