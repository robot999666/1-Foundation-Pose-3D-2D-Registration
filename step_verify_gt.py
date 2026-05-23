import open3d as o3d
import numpy as np
import cv2
import os

def load_camera_matrix(filepath):
    """从 txt 文件中读取 3x3 内参矩阵"""
    K = []
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"找不到内参文件: {filepath}")
        
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            K.append([float(x) for x in line.split()])
    return np.array(K, dtype=np.float64)

def project_and_draw(canvas, pcd_path, K_matrix, color_bgr, label):
    """通用的点云投影与绘制函数"""
    if not os.path.exists(pcd_path):
        print(f"  -> [警告] 找不到点云文件: {pcd_path}，已跳过。")
        return canvas
        
    pcd = o3d.io.read_point_cloud(pcd_path)
    if not pcd.has_points():
        print(f"  -> [错误] 点云 {label} 数据为空！")
        return canvas
        
    points_3d = np.asarray(pcd.points)
    print(f"  -> 成功加载 {label}，共 {len(points_3d)} 个点。")

    # 外参全设为0（点云已在相机坐标系下），畸变为0（已用无畸变内参）
    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    dist_coeffs = np.zeros(5)

    # 投影到 2D
    pts_2d, _ = cv2.projectPoints(points_3d, rvec, tvec, K_matrix, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)

    # 获取画布尺寸
    h_img, w_img = canvas.shape[:2]

    # 过滤掉跑到画面外的点
    x_pts = pts_2d[:, 0]
    y_pts = pts_2d[:, 1]
    valid_mask = (x_pts >= 0) & (x_pts < w_img - 1) & (y_pts >= 0) & (y_pts < h_img - 1)
    
    vx = x_pts[valid_mask]
    vy = y_pts[valid_mask]

    if len(vx) == 0:
        print(f"  -> [严重警告] {label} 的所有点都在有效画面之外！")
    else:
        # 画出醒目的 2x2 像素块
        canvas[vy, vx] = color_bgr
        canvas[vy+1, vx] = color_bgr
        canvas[vy, vx+1] = color_bgr
        canvas[vy+1, vx+1] = color_bgr
        print(f"  -> 成功将 {len(vx)} 个 {label} 点投影到有效画面内。")
        
    return canvas

def verify_combined_data():
    print("=== 联合检验：双重点云投影验证 ===")

    # --- 1. 路径配置 ---
    K_NEW_PATH = r"data/test/camera_new.txt"
    BG_IMG_PATH = r"data/test/test_new.png"  
    
    # 两组需要比对的点云
    PCD_3D_REAL = r"data/target/3d_real.ply"
    PCD_2D_REAL = r"data/target/2d_real.ply"
    
    RENDER_IMG_PATH = r"assets/combined_projection_vis.png"

    os.makedirs("assets", exist_ok=True)

    # --- 2. 加载基础数据 ---
    try:
        K_new = load_camera_matrix(K_NEW_PATH)
    except Exception as e:
        print(f"[错误] {e}")
        return

    bg_img = cv2.imread(BG_IMG_PATH)
    if bg_img is None:
        print(f"[警告] 找不到背景图 {BG_IMG_PATH}，将使用黑色背景进行渲染。")
        bg_img = np.zeros((540, 960, 3), dtype=np.uint8)
        
    canvas = bg_img.copy()

    # --- 3. 叠加投影 ---
    print("\n[开始投影 3d_real (红色)]...")
    # BGR 格式：红色为 [0, 0, 255]
    canvas = project_and_draw(canvas, PCD_3D_REAL, K_new, color_bgr=[0, 0, 255], label="3d_real")

    print("\n[开始投影 2d_real (蓝色)]...")
    # BGR 格式：蓝色为 [255, 0, 0]
    canvas = project_and_draw(canvas, PCD_2D_REAL, K_new, color_bgr=[255, 0, 0], label="2d_real")

    # --- 4. 添加图例与保存 ---
    # 在图片左上角画图例，方便观察
    cv2.putText(canvas, "Red: 3d_real.ply", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(canvas, "Blue: 2d_real.ply", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    cv2.imwrite(RENDER_IMG_PATH, canvas)
    print(f"\n[大功告成] 联合投影验证图已保存至: {RENDER_IMG_PATH}")
    print("快去打开图片，看看红蓝点阵的对齐情况吧！")

if __name__ == "__main__":
    verify_combined_data()