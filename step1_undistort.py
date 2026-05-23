import cv2
import numpy as np
import os

def process_surgical_image():
    print("=== 第一阶段：相机参数重构与图像去畸变 ===")
    
    # 1. 填入你提供的真实的左目内参 (K_LEFT) 和 畸变系数 (DIST_LEFT)
    K_left = np.array([
        [799.704023, -0.976384, 403.470419],
        [0.000000,   803.260414, 275.213495],
        [0.000000,   0.000000,   1.000000]
    ], dtype=np.float64)

    dist_left = np.array([-0.402557, 0.176782, 0.000000, 0.000000, 0.000000], dtype=np.float64)

    # 图像分辨率
    w, h = 960, 540

    # 文件路径配置
    input_img_path = r"data/test/test.png"
    output_img_path = r"data/test/test_new.png"
    output_cam_path = r"data/test/camera_new.txt"

    # 2. 读取原始术中图片
    if not os.path.exists(input_img_path):
        print(f"[错误] 找不到输入图像: {input_img_path}")
        print("请检查路径或先放入一张测试图片。")
        return

    img_raw = cv2.imread(input_img_path)
    if img_raw is None:
        print("[错误] 图像读取失败，请检查文件格式。")
        return

    # 3. 计算最优的新内参矩阵 (New Camera Matrix)
    # 参数 alpha (0 到 1): 
    # alpha = 0: 裁剪掉所有去畸变后产生的黑边（推荐，避免黑边干扰深度估计网络）
    # alpha = 1: 保留所有原始像素（会有黑色弧形边框）
    alpha = 0 
    
    print("正在计算最优线性内参矩阵 (alpha=0)...")
    K_new, roi = cv2.getOptimalNewCameraMatrix(K_left, dist_left, (w, h), alpha, (w, h))

    # 4. 执行图像去畸变 (Undistortion)
    print("正在对图像进行物理畸变校正...")
    img_undistorted = cv2.undistort(img_raw, K_left, dist_left, None, K_new)

    # 5. 保存去畸变后的新图像
    os.makedirs(os.path.dirname(output_img_path), exist_ok=True)
    cv2.imwrite(output_img_path, img_undistorted)
    print(f"[成功] 无畸变图像已保存至: {output_img_path}")

    # 6. 将新的内参矩阵保存为 txt 文件
    # 按照标准 3x3 格式写入，方便后续渲染引擎(PyTorch3D/OpenGL)读取
    with open(output_cam_path, "w") as f:
        f.write("# NEW_LEFT_CAM_MATRIX (Undistorted)\n")
        f.write("# Resolution: 960x540\n")
        for row in K_new:
            f.write(f"{row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
            
    print(f"[成功] 全新相机内参已保存至: {output_cam_path}")
    print("\n=== 新内参矩阵 K_new ===")
    print(K_new)
    print("\n[重要提示]：原有的 K_LEFT 和 DIST_LEFT 已完成历史使命，请在后续的 3D 渲染和网络训练中，严格使用 camera_new.txt 中的矩阵！")

if __name__ == "__main__":
    process_surgical_image()