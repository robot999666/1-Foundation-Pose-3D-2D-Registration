import os
import json
import logging
import cv2
import open3d as o3d
import numpy as np

# ========================================================
# 【安全核心：解决 RTX 40 系显卡 sm_80 报错的官方补丁】
# 强制禁用有 Bug 的 FlashAttention 内核，回退到标准数学计算，防崩溃
import torch
if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
    torch.backends.cuda.enable_flash_sdp(False)
if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
    torch.backends.cuda.enable_mem_efficient_sdp(False)
if hasattr(torch.backends.cuda, 'enable_math_sdp'):
    torch.backends.cuda.enable_math_sdp(True)
# ========================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
# 移除 weight_decay 的干扰，使用纯正的 Adam
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from learning.models.refine_network import RefineNet


class Config:
    TRAIN_DIR = r"data/sample/train"
    VAL_DIR = r"data/sample/val"
    MODEL_PLY_PATH = r"data/ori/Model_obj.ply"
    # 输出目录调整，避免覆盖旧数据
    RESULT_DIR = r"output_mse_adam"

    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 8
    LR = 1e-4
    NUM_EPOCHS = 20
    NUM_WORKERS = 4

    ROT_REP = "6d"
    TRANS_SCALE = 50.0
    NUM_SAMPLED_PTS = 2000
    IMG_SIZE = (480, 270)


def setup_logger(log_dir):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger("KidneyTrain_AdamMSE")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    fh = logging.FileHandler(os.path.join(log_dir, 'train.log'), mode='a')
    fh.setFormatter(formatter)
    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def plot_loss_curves(train_losses, val_losses, result_dir):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Surface MSE (mm^2)', marker='o', alpha=0.7)
    plt.plot(val_losses, label='Val Surface MSE (mm^2)', marker='x', alpha=0.7)
    plt.title('Training and Validation Surface MSE')
    plt.xlabel('Epoch')
    plt.ylabel('Surface MSE (mm^2)')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.savefig(os.path.join(result_dir, 'loss_curve.png'))
    plt.close()


def load_3d_model_points(ply_path, num_points):
    mesh = o3d.io.read_triangle_mesh(ply_path)
    if not mesh.has_vertices():
        raise ValueError(f"无法读取 3D 模型: {ply_path}")
    pcd = mesh.sample_points_uniformly(number_of_points=num_points)
    points = np.asarray(pcd.points).astype(np.float32)
    return torch.from_numpy(points)


class KidneyTrackingDataset(Dataset):
    def __init__(self, data_dir):
        self.img_dir = os.path.join(data_dir, "images")
        with open(os.path.join(data_dir, "labels.json"), 'r', encoding='utf-8') as f:
            self.labels = json.load(f)
        self.keys = list(self.labels.keys())
        self.num_samples = len(self.keys)

    def _resize_modalities(self, mask, contour, depth):
        target_size = Config.IMG_SIZE
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        contour = cv2.resize(contour, target_size, interpolation=cv2.INTER_NEAREST)
        depth = cv2.resize(depth, target_size, interpolation=cv2.INTER_NEAREST)
        return mask, contour, depth

    def _modalities_to_tensor(self, mask, contour, depth):
        c = (contour / 255.0).astype(np.float32)
        m = (mask / 255.0).astype(np.float32)
        d = depth.astype(np.float32)
        return torch.from_numpy(np.stack([c, d, m], axis=0))

    def _pack_pair_side(self, prefix, role):
        mask = cv2.imread(os.path.join(self.img_dir, f"{prefix}_{role}_mask.png"), cv2.IMREAD_GRAYSCALE)
        contour = cv2.imread(os.path.join(self.img_dir, f"{prefix}_{role}_contour.png"), cv2.IMREAD_GRAYSCALE)
        depth = np.load(os.path.join(self.img_dir, f"{prefix}_{role}_depth.npy"))
        mask, contour, depth = self._resize_modalities(mask, contour, depth)
        return self._modalities_to_tensor(mask, contour, depth)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        prefix = self.keys[idx]
        label = self.labels[prefix]

        tensor_A = self._pack_pair_side(prefix, "A")
        tensor_B = self._pack_pair_side(prefix, "B")
        gt_T_A = np.array(label["Delta_A"], dtype=np.float32)
        gt_T_B = np.array(label["Delta_B"], dtype=np.float32)

        network_input = torch.cat([tensor_A, tensor_B], dim=0)
        return network_input, torch.from_numpy(gt_T_A), torch.from_numpy(gt_T_B)


def compute_rotation_matrix_from_6d(poses):
    x_raw = poses[:, 0:3]
    y_raw = poses[:, 3:6]
    x = F.normalize(x_raw, p=2, dim=-1)
    z_dot_x = (y_raw * x).sum(dim=-1, keepdim=True)
    y = F.normalize(y_raw - z_dot_x * x, p=2, dim=-1)
    z = torch.cross(x, y, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def compute_surface_mse_loss(pred_trans, pred_rot_6d, gt_T_A, gt_T_B, model_points, trans_scale=50.0):
    batch_size = pred_trans.shape[0]

    R_rel_pred = compute_rotation_matrix_from_6d(pred_rot_6d)
    t_rel_pred = (pred_trans * trans_scale).unsqueeze(2)

    R_A = gt_T_A[:, :3, :3]
    t_A = gt_T_A[:, :3, 3:4]
    R_B = gt_T_B[:, :3, :3]
    t_B = gt_T_B[:, :3, 3:4]

    R_est = torch.bmm(R_A, R_rel_pred)
    t_est = torch.bmm(R_A, t_rel_pred) + t_A

    pts_obj = model_points.unsqueeze(0).expand(batch_size, -1, -1).transpose(1, 2)
    pts_est = torch.bmm(R_est, pts_obj) + t_est
    pts_gt = torch.bmm(R_B, pts_obj) + t_B

    # 严格使用带平方的 L2 范数 (MSE)
    sq_dist = torch.sum((pts_est - pts_gt) ** 2, dim=1)
    return sq_dist.mean()


def main():
    cfg = Config()
    logger = setup_logger(cfg.RESULT_DIR)
    logger.info("=== 启动医疗级 FoundationPose Tracking MSE 训练 (Adam无衰减版) ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"物理 BatchSize: {cfg.BATCH_SIZE} | 等效 BatchSize: {cfg.BATCH_SIZE * cfg.ACCUMULATION_STEPS}")

    train_dataset = KidneyTrackingDataset(cfg.TRAIN_DIR)
    val_dataset = KidneyTrackingDataset(cfg.VAL_DIR)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=cfg.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS, pin_memory=True)

    model_points = load_3d_model_points(cfg.MODEL_PLY_PATH, cfg.NUM_SAMPLED_PTS).to(device)

    class DummyCfg:
        use_BN = True
        rot_rep = cfg.ROT_REP

    model = RefineNet(cfg=DummyCfg(), c_in=3).to(device)
    
    # 使用纯正 Adam
    optimizer = Adam(model.parameters(), lr=cfg.LR)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2, verbose=True)
    scaler = torch.cuda.amp.GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    train_losses, val_losses = [], []

    last_ckpt_path = os.path.join(cfg.RESULT_DIR, "last.pth")
    if os.path.exists(last_ckpt_path):
        logger.info(f"正在从 {last_ckpt_path} 恢复...")
        checkpoint = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state'])
        optimizer.load_state_dict(checkpoint['optimizer_state'])
        scheduler.load_state_dict(checkpoint['scheduler_state'])
        scaler.load_state_dict(checkpoint.get('scaler_state', scaler.state_dict()))
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_losses = checkpoint['train_losses']
        val_losses = checkpoint['val_losses']

    for epoch in range(start_epoch, cfg.NUM_EPOCHS):
        logger.info(f"--- Epoch [{epoch+1}/{cfg.NUM_EPOCHS}] (LR: {optimizer.param_groups[0]['lr']:.2e}) ---")

        model.train()
        epoch_train_loss = 0.0
        pbar = tqdm(train_loader, desc="Training MSE")
        optimizer.zero_grad()

        for batch_idx, (inputs, gt_T_A, gt_T_B) in enumerate(pbar):
            inputs, gt_T_A, gt_T_B = inputs.to(device), gt_T_A.to(device), gt_T_B.to(device)

            A_inputs = inputs[:, :3, :, :]
            B_inputs = inputs[:, 3:, :, :]

            with torch.cuda.amp.autocast():
                out = model(A_inputs, B_inputs)
                loss = compute_surface_mse_loss(out['trans'], out['rot'], gt_T_A, gt_T_B, model_points, trans_scale=cfg.TRANS_SCALE)
                scaled_loss = loss / cfg.ACCUMULATION_STEPS

            scaler.scale(scaled_loss).backward()

            if ((batch_idx + 1) % cfg.ACCUMULATION_STEPS == 0) or ((batch_idx + 1) == len(train_loader)):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            true_loss_val = loss.item()
            epoch_train_loss += true_loss_val
            pbar.set_postfix({'SurfaceMSE(mm^2)': f"{true_loss_val:.2f}"})

        avg_train_loss = epoch_train_loss / len(train_loader)

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for inputs, gt_T_A, gt_T_B in tqdm(val_loader, desc="Validation MSE"):
                inputs, gt_T_A, gt_T_B = inputs.to(device), gt_T_A.to(device), gt_T_B.to(device)

                A_inputs = inputs[:, :3, :, :]
                B_inputs = inputs[:, 3:, :, :]

                with torch.cuda.amp.autocast():
                    out = model(A_inputs, B_inputs)
                    loss = compute_surface_mse_loss(out['trans'], out['rot'], gt_T_A, gt_T_B, model_points, trans_scale=cfg.TRANS_SCALE)

                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        logger.info(f"[结果] Train Surface MSE: {avg_train_loss:.4f} mm^2 | Val Surface MSE: {avg_val_loss:.4f} mm^2")

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        plot_loss_curves(train_losses, val_losses, cfg.RESULT_DIR)

        scheduler.step(avg_val_loss)

        checkpoint_state = {
            'epoch': epoch,
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict(),
            'best_val_loss': best_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }
        torch.save(checkpoint_state, last_ckpt_path)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(cfg.RESULT_DIR, "best.pth")
            torch.save(model.state_dict(), best_path)
            logger.info(f"[*] 新纪录！最佳权重已保存至: {best_path} (Surface MSE 降至 {best_val_loss:.4f} mm^2)")


if __name__ == "__main__":
    main()