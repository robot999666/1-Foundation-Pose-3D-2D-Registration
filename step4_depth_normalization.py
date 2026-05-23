import numpy as np
import cv2
import os

def normalize_depth():
    print("=== 第四阶段：Depth Anything 深度翻转、掩码提取与归一化 ===")

    # --- 1. 路径配置 ---
    mask_path = r"data/test/mask.png"
    depth_path = r"data/test/depth.npy"
    
    out_npy_path = r"data/test/depth_normalized.npy"
    out_vis_path = r"assets/depth_vis.png" # 生成一张可视化的伪彩图放进 assets

    # 确保输出目录存在
    os.makedirs("assets", exist_ok=True)

    if not os.path.exists(mask_path) or not os.path.exists(depth_path):
        print(f"[错误] 找不到 {mask_path} 或 {depth_path}。请检查路径。")
        return

    # --- 2. 加载数据 ---
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    depth = np.load(depth_path)

    # 安全检查：确保掩码和深度图分辨率完全一致
    if mask.shape != depth.shape:
        print(f"[警告] 掩码尺寸 {mask.shape} 与深度图尺寸 {depth.shape} 不一致，正在强制对齐...")
        mask = cv2.resize(mask, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 将掩码严格二值化，确保 255 代表目标区域，0 代表背景
    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    valid_pixels = (mask_binary == 255) # 布尔型索引矩阵

    # --- 3. 提取与极值分析 ---
    target_depths = depth[valid_pixels]

    if len(target_depths) == 0:
        print("[错误] 掩码为空（全黑），无法提取目标深度！")
        return

    min_depth = target_depths.min()
    max_depth = target_depths.max()
    
    print(f"  -> Depth Anything 原始输出极值范围: [{min_depth:.4f}, {max_depth:.4f}]")
    print("  -> (注意：原模型中数值越大代表越近，我们将对其进行翻转)")

    # --- 4. 深度归一化与物理反转 ---
    depth_range = max_depth - min_depth
    if depth_range == 0:
        # 防止除零错误
        normalized_target = np.zeros_like(target_depths)
    else:
        # 第一步：将其压缩到 0.0 ~ 1.0 之间
        # 此时：1.0 是最近点，0.0 是最远点 (Depth Anything 的逻辑)
        raw_normalized = (target_depths - min_depth) / depth_range
        
        # 第二步：【核心反转】(1 - value) 恢复物理真实逻辑
        # 翻转后：0.0 变成最近点，1.0 变成最远点 (物理相机的逻辑)
        normalized_target = 1.0 - raw_normalized

    # --- 5. 构建最终矩阵 ---
    # 先创建一个全部为 1 的矩阵（物理极远背景处全为 1）
    depth_normalized = np.ones_like(depth, dtype=np.float32)
    # 把归一化并翻转后的目标深度填进去
    depth_normalized[valid_pixels] = normalized_target

    # 保存交给神经网络的数据
    np.save(out_npy_path, depth_normalized)
    print(f"  -> [数据落盘] 反转归一化深度矩阵已保存至: {out_npy_path}")

    # --- 6. 生成伪彩色可视化验证图 (Jet Colormap) ---
    # 将 0.0-1.0 映射到 0-255
    depth_uint8 = (depth_normalized * 255).astype(np.uint8)
    
    # 采用 JET 伪彩色：数值小(近，即接近0)呈蓝色，数值大(远，即接近1)呈红色
    depth_color = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_JET)
    
    # 为了让背景不干扰观察，把背景区域强制涂黑
    depth_color[~valid_pixels] = [0, 0, 0]

    cv2.imwrite(out_vis_path, depth_color)
    print(f"  -> [视觉落盘] 深度验证图已保存至: {out_vis_path}")
    print("\n[系统提示]：请打开 assets/depth_vis.png 查看。")
    print("【检验标准】：肾脏最凸起、离镜头最近的地方现在应该是深蓝色的；越往边缘、向后弯曲远离镜头的地方应该是红色的。")

if __name__ == "__main__":
    normalize_depth()