import cv2
import numpy as np
import os

# 全局状态变量
points = []
window_name = "Interactive Masking (L-Click: Add, R-Click: Undo, C: Clear, S: Save)"

def mouse_callback(event, x, y, flags, param):
    global points
    img_clean = param['img']
    
    # 左键：添加锚点
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        redraw(img_clean)
    # 右键：撤销上一个锚点
    elif event == cv2.EVENT_RBUTTONDOWN:
        if points:
            points.pop()
            redraw(img_clean)

def redraw(img_clean):
    global points
    img_show = img_clean.copy()
    
    if len(points) > 0:
        # 1. 绘制锚点
        for p in points:
            cv2.circle(img_show, p, 3, (0, 0, 255), -1) # 红色小圆点
            
        # 2. 绘制锚点之间的连线
        if len(points) > 1:
            cv2.polylines(img_show, [np.array(points)], isClosed=False, color=(0, 255, 0), thickness=2)
            
        # 3. 如果点数大于2，显示黄色的“闭合预测线”，让你知道最终轮廓长什么样
        if len(points) > 2:
            cv2.polylines(img_show, [np.array(points)], isClosed=True, color=(0, 255, 255), thickness=1, lineType=cv2.LINE_AA)
            
    cv2.imshow(window_name, img_show)

def extract_and_save():
    print("=== 第三阶段：真值 Mask 与 Contour 提取 ===")
    
    img_path = r"data/test/test_new.png"
    mask_path = r"data/test/mask.png"
    contour_path = r"data/test/contour.png"
    
    if not os.path.exists(img_path):
        print(f"[错误] 找不到图像: {img_path}，请确保第一步已生成。")
        return
        
    img_clean = cv2.imread(img_path)
    h, w = img_clean.shape[:2]
    
    # 初始化交互窗口
    cv2.namedWindow(window_name, cv2.WINDOW_GUI_EXPANDED)
    cv2.setMouseCallback(window_name, mouse_callback, param={'img': img_clean})
    
    print("\n[操作指南]：")
    print(" -> 鼠标【左键】：沿着肾脏边缘点击，放置描边锚点。")
    print(" -> 鼠标【右键】：撤销上一个点。")
    print(" -> 键盘【C键】：清空所有点，重新开始。")
    print(" -> 键盘【S键】：完成勾画，保存掩码并退出。")
    print(" -> 键盘【ESC】：取消并退出。")
    
    redraw(img_clean)
    
    while True:
        key = cv2.waitKey(20) & 0xFF
        
        if key == 27: # ESC 键
            print("操作已取消。")
            break
            
        elif key == ord('c') or key == ord('C'):
            global points
            points = []
            redraw(img_clean)
            
        elif key == ord('s') or key == ord('S'):
            if len(points) < 3:
                print("[警告] 至少需要 3 个点才能构成一个闭合的肾脏掩码！")
                continue
                
            print("\n[处理中] 正在生成数学掩码与轮廓...")
            
            # --- 1. 生成纯净掩码 (Solid Mask) ---
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [np.array(points)], 255)
            
            # --- 2. 生成边界轮廓线 (Contour Lines) ---
            contour_img = np.zeros((h, w), dtype=np.uint8)
            cv2.polylines(contour_img, [np.array(points)], isClosed=True, color=255, thickness=2, lineType=cv2.LINE_AA)
            
            # --- 3. [核心逻辑] 自动消除图片物理边缘的轮廓线 ---
            # 为什么要做这一步？因为如果你的肾脏被镜头边缘截断了，
            # 勾画的多边形会贴着图片边缘走，导致网络学到一个毫无意义的“直线几何特征”。
            # 解决方案：强行将图像最外圈（如 3 像素宽）的区域抹黑。
            EDGE_THICKNESS = 3 
            
            # 抹除上下边缘
            contour_img[0:EDGE_THICKNESS, :] = 0
            contour_img[-EDGE_THICKNESS:, :] = 0
            mask[0:EDGE_THICKNESS, :] = 0
            mask[-EDGE_THICKNESS:, :] = 0
            
            # 抹除左右边缘
            contour_img[:, 0:EDGE_THICKNESS] = 0
            contour_img[:, -EDGE_THICKNESS:] = 0
            mask[:, 0:EDGE_THICKNESS] = 0
            mask[:, -EDGE_THICKNESS:] = 0
            
            # --- 4. 图像落盘 ---
            cv2.imwrite(mask_path, mask)
            cv2.imwrite(contour_path, contour_img)
            
            print(f"[成功] 实体掩码已保存至: {mask_path}")
            print(f"[成功] 边缘过滤后的轮廓线已保存至: {contour_path}")
            print("（注：任何触碰到图像边框的直线已经被彻底剔除，不会干扰后续 3D 匹配）")
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    extract_and_save()