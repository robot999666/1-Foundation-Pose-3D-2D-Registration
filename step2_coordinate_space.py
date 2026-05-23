import open3d as o3d
import numpy as np
import cv2
import os
import copy
import math

def load_camera_matrix(filepath):
    """从 txt 文件中读取 3x3 内参矩阵"""
    K = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            K.append([float(x) for x in line.split()])
    return np.array(K, dtype=np.float64)

def euler_to_matrix(rx_deg, ry_deg, rz_deg):
    """将欧拉角(度)转换为 3x3 旋转矩阵"""
    rx, ry, rz = math.radians(rx_deg), math.radians(ry_deg), math.radians(rz_deg)
    R_x = np.array([[1, 0, 0], 
                    [0, math.cos(rx), -math.sin(rx)], 
                    [0, math.sin(rx), math.cos(rx)]])
    R_y = np.array([[math.cos(ry), 0, math.sin(ry)], 
                    [0, 1, 0], 
                    [-math.sin(ry), 0, math.cos(ry)]])
    R_z = np.array([[math.cos(rz), -math.sin(rz), 0], 
                    [math.sin(rz), math.cos(rz), 0], 
                    [0, 0, 1]])
    return np.dot(R_z, np.dot(R_y, R_x))

def setup_coordinate_spaces():
    print("=== 第二阶段：全自由度静态视角烘焙与验证 ===")

    # ==========================================
    # [核心微调区] 6-DoF 刚性变换调节
    # 每次看完 assets/test1.png，不满意就改这里的数字！
    # ==========================================
    
    # 1. 平移微调 (单位: 毫米)
    OFFSET_X = 0.0  # 负数向左移 (纠正偏右)
    OFFSET_Y =  -10.0  # 正数向下移 (纠正偏上)
    OFFSET_Z = 10.0  # 正数向远推 (纠正偏大/偏近)
    
    # 2. 旋转微调 (单位: 度 / 绕模型自身质心旋转)
    OFFSET_RX = -40.0   # 绕 X 轴旋转 (正数通常表现为模型向上"抬头")
    OFFSET_RY = 100.0   # 绕 Y 轴旋转 (正数通常表现为模型向右"侧身")
    OFFSET_RZ = -120.0   # 绕 Z 轴旋转 (平面内的顺/逆时针自转)
    
    print(f"[参数设定] 平移补偿: X={OFFSET_X}, Y={OFFSET_Y}, Z={OFFSET_Z}")
    print(f"[参数设定] 旋转补偿: RX={OFFSET_RX}°, RY={OFFSET_RY}°, RZ={OFFSET_RZ}°")

    # --- 路径配置 ---
    K_NEW_PATH = r"data/test/camera_new.txt"
    RAW_MESH_PATH = r"data/ori/Model_raw.ply"
    REAL_PCD_PATH = r"data/target/3d_real.ply"
    BG_IMG_PATH = r"data/test/test_new.png"  
    
    OBJ_MESH_PATH = r"data/ori/Model_obj.ply"
    T_VIEW_PATH = r"data/ori/T_view.txt"
    CAM_MESH_PATH = r"data/ori/Model_cam.ply"
    RENDER_IMG_PATH = r"assets/test1.png"

    os.makedirs("data/ori", exist_ok=True)
    os.makedirs("assets", exist_ok=True)

    # ==========================================
    # 1. 建立物体空间 (Model_obj) - 质心归零
    # ==========================================
    print("\n[1] 正在去中心化，生成 Model_obj...")
    mesh_raw = o3d.io.read_triangle_mesh(RAW_MESH_PATH)
    if not mesh_raw.has_vertices():
        print(f"[错误] 无法读取 {RAW_MESH_PATH}")
        return

    vertices_raw = np.asarray(mesh_raw.vertices)
    centroid_raw = np.mean(vertices_raw, axis=0)
    
    mesh_obj = copy.deepcopy(mesh_raw)
    mesh_obj.translate(-centroid_raw)
    o3d.io.write_triangle_mesh(OBJ_MESH_PATH, mesh_obj)

    # ==========================================
    # 2. 确立静态观测矩阵 (T_view)
    # ==========================================
    print("\n[2] 正在计算静态观测矩阵 (T_view)...")
    
    # 尝试读取真值点云作为基准位置
    init_tx, init_ty, init_tz = 0.0, 0.0, 150.0
    if os.path.exists(REAL_PCD_PATH):
        pcd_real = o3d.io.read_point_cloud(REAL_PCD_PATH)
        if pcd_real.has_points():
            real_points = np.asarray(pcd_real.points)
            centroid_real = np.mean(real_points, axis=0)
            init_tx, init_ty, init_tz = centroid_real[0], centroid_real[1], centroid_real[2]

    # 施加平移补偿
    final_tx = init_tx + OFFSET_X
    final_ty = init_ty + OFFSET_Y
    final_tz = init_tz + OFFSET_Z

    # 施加旋转补偿 (生成 3x3 旋转矩阵)
    R_mat = euler_to_matrix(OFFSET_RX, OFFSET_RY, OFFSET_RZ)

    # 构造最终的 4x4 T_view 矩阵
    T_view = np.eye(4)
    T_view[:3, :3] = R_mat
    T_view[:3, 3] = [final_tx, final_ty, final_tz]

    np.savetxt(T_VIEW_PATH, T_view, fmt='%.6f', header='T_view (Object Space to Camera Space)\n4x4 Matrix')
    print(f"  -> T_view 矩阵已保存至: {T_VIEW_PATH}")

    # ==========================================
    # 3. 映射到相机空间 (Model_cam)
    # ==========================================
    print("\n[3] 正在生成相机空间模型 (Model_cam)...")
    mesh_cam = copy.deepcopy(mesh_obj)
    mesh_cam.transform(T_view) 
    o3d.io.write_triangle_mesh(CAM_MESH_PATH, mesh_cam)
    print(f"  -> Model_cam 已保存")

    # ==========================================
    # 4. 投影验证 (透视渲染到原图背景)
    # ==========================================
    print("\n[4] 正在利用 K_new 执行 3D->2D 叠加渲染...")
    K_new = load_camera_matrix(K_NEW_PATH)
    
    bg_img = cv2.imread(BG_IMG_PATH)
    if bg_img is None:
        bg_img = np.zeros((540, 960, 3), dtype=np.uint8)
    
    img_render = bg_img.copy()

    vertices_cam = np.asarray(mesh_cam.vertices)
    triangles = np.asarray(mesh_cam.triangles)

    rvec = np.zeros((3, 1))
    tvec = np.zeros((3, 1))
    dist_coeffs = np.zeros(5)

    pts_2d, _ = cv2.projectPoints(vertices_cam, rvec, tvec, K_new, dist_coeffs)
    pts_2d = pts_2d.reshape(-1, 2).astype(np.int32)

    overlay = img_render.copy()
    
    # 降采样绘制加速
    step = max(1, len(triangles) // 20000)
    for tri in triangles[::step]:
        pt1, pt2, pt3 = pts_2d[tri[0]], pts_2d[tri[1]], pts_2d[tri[2]]
        if all(-500 < pt[0] < 1500 and -500 < pt[1] < 1500 for pt in (pt1, pt2, pt3)):
            cv2.fillConvexPoly(overlay, np.array([pt1, pt2, pt3]), color=(100, 50, 100))
            cv2.polylines(overlay, [np.array([pt1, pt2, pt3])], isClosed=True, color=(0, 255, 0), thickness=1)

    alpha = 0.5
    cv2.addWeighted(overlay, alpha, img_render, 1 - alpha, 0, img_render)

    cv2.imwrite(RENDER_IMG_PATH, img_render)
    print(f"  -> 叠加验证图已生成: {RENDER_IMG_PATH}")
    print("\n[调整建议]：")
    print("1. 先调 X, Y 让中心对齐。")
    print("2. 再调 Z 让大小（比例）匹配。")
    print("3. 最后调 RX, RY, RZ 让边缘姿态严丝合缝！")

if __name__ == "__main__":
    setup_coordinate_spaces()