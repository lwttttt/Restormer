"""
在线 Rain Soft Label 生成器

将 rain_layer (lq - gt) 通过 HOG + PCA + Codebook 距离
转换为 7 类软标签概率分布 T:
  - 第 0 类: 无雨/背景 (灰度 std < threshold)
  - 第 1~6 类: 6 种雨纹聚类 (softmax(-distance²/τ))

用法:
    gen = RainSoftLabelGenerator(codebook_dir, temperature=1.0, energy_threshold=10.0)
    T = gen.compute_soft_labels(lq_batch, gt_batch)  # (B, 7)
"""
import numpy as np
import torch
import torch.nn.functional as F
import joblib
import cv2
from skimage.feature import hog
from pathlib import Path


# Class 0 label smoothing: 避免 KL 散度中 log(0) 导致梯度不稳定
CLEAN_LABEL_SMOOTH = 0.01  # 每个非主类分到 0.01
CLEAN_LABEL = None  # 延迟初始化


class RainSoftLabelGenerator:

    def __init__(self, codebook_dir, temperature=1.0, energy_threshold=10.0):
        """
        Args:
            codebook_dir: 包含 scaler.pkl, pca.pkl, codebook_k6.npy 的目录
            temperature: softmax 温度系数
            energy_threshold: 灰度 std 阈值，低于此值归为第 0 类(无雨)
                              注意: std 在 resize 前的原始分辨率上计算，与离线一致
        """
        codebook_dir = Path(codebook_dir)
        self.scaler = joblib.load(codebook_dir / 'scaler.pkl')
        self.pca = joblib.load(codebook_dir / 'pca.pkl')
        self.codebook = np.load(codebook_dir / 'codebook_k6.npy')  # (6, 50)
        self.temperature = temperature
        self.energy_threshold = energy_threshold
        self.K = self.codebook.shape[0]  # 6
        self.num_classes = self.K + 1     # 7 (含第 0 类)

        # Class 0 的 smoothed label: [0.94, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
        self._clean_label = np.full(self.num_classes, CLEAN_LABEL_SMOOTH, dtype=np.float32)
        self._clean_label[0] = 1.0 - CLEAN_LABEL_SMOOTH * self.K

    def _rain_to_mean_gray_uint8(self, rain_rgb_float):
        """
        简单 RGB 平均灰度化 (门卫逻辑)。
        严格复刻 extract_patches.py line 78:
            img_gray = np.mean(img_array, axis=2)
        用于 std 阈值判断，决定 patch 是否归为 class 0。

        Args:
            rain_rgb_float: (H, W, 3) float32, [0,1] 范围, 已 clamp>=0
        Returns:
            gray_uint8: (H, W) uint8, [0,255]
        """
        rain_255 = np.clip(rain_rgb_float * 255.0, 0, 255).astype(np.uint8)
        gray = np.mean(rain_255.astype(np.float32), axis=2)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _rain_to_weighted_gray_uint8(self, rain_rgb_float):
        """
        加权灰度化 (专家逻辑)。
        严格复刻 extract_rain_features_hog.py 的 PIL.convert('L'):
            0.299*R + 0.587*G + 0.114*B
        用于 HOG 特征提取，保证与 codebook 的 PCA 空间一致。

        Args:
            rain_rgb_float: (H, W, 3) float32, [0,1] 范围, 已 clamp>=0
        Returns:
            gray_uint8: (H, W) uint8, [0,255]
        """
        rain_255 = np.clip(rain_rgb_float * 255.0, 0, 255).astype(np.uint8)
        gray = (0.299 * rain_255[:, :, 0].astype(np.float32)
                + 0.587 * rain_255[:, :, 1].astype(np.float32)
                + 0.114 * rain_255[:, :, 2].astype(np.float32))
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _extract_hog(self, gray_uint8):
        """
        对灰度图 resize 到 128×128 后提取 HOG 特征。
        参数与离线完全一致: orientations=9, ppc=(8,8), cpb=(2,2)
        插值算法: cv2.INTER_AREA (与离线 extract_rain_features_hog.py 一致)

        Args:
            gray_uint8: (H, W) uint8
        Returns:
            hog_feat: (8100,) float64
        """
        resized = cv2.resize(gray_uint8, (128, 128), interpolation=cv2.INTER_AREA)
        return hog(resized, orientations=9, pixels_per_cell=(8, 8),
                   cells_per_block=(2, 2), feature_vector=True)

    def compute_soft_labels(self, lq, gt):
        """
        计算一个 batch 的 soft label。

        Args:
            lq: (B, 3, H, W) tensor, [0,1] RGB, 任意 device
            gt: (B, 3, H, W) tensor, [0,1] RGB, 任意 device
        Returns:
            T: (B, 7) tensor, 在 lq 的 device 上
        """
        device = lq.device
        B = lq.shape[0]

        # rain_layer = clamp(lq - gt, min=0), 转 CPU numpy
        rain = torch.clamp(lq - gt, min=0).detach().cpu().numpy()  # (B, 3, H, W)

        T = np.zeros((B, self.num_classes), dtype=np.float32)

        # 分离: 哪些样本需要走 HOG, 哪些直接归 class 0
        hog_indices = []
        hog_grays = []

        for i in range(B):
            rain_rgb = rain[i].transpose(1, 2, 0)  # (H, W, 3)

            # 门卫: 简单平均灰度, 严格复刻 extract_patches.py
            gray_mean = self._rain_to_mean_gray_uint8(rain_rgb)

            # 全 0 图像保护: 如果 rain layer 全黑，直接归 class 0
            if gray_mean.max() == 0:
                T[i] = self._clean_label
                continue

            # std 在原始分辨率上计算 (resize 前)，用简单平均灰度与离线一致
            std_val = float(np.std(gray_mean))

            if std_val < self.energy_threshold:
                T[i] = self._clean_label
            else:
                # 专家: 加权灰度, 严格复刻 extract_rain_features_hog.py
                gray_weighted = self._rain_to_weighted_gray_uint8(rain_rgb)
                hog_indices.append(i)
                hog_grays.append(gray_weighted)

        # batch 提取 HOG + 向量化 scaler/PCA/距离
        if hog_indices:
            n = len(hog_indices)
            hog_features = np.zeros((n, 8100), dtype=np.float64)
            for j, gray in enumerate(hog_grays):
                hog_features[j] = self._extract_hog(gray)

            # scaler + PCA (向量化)
            hog_scaled = self.scaler.transform(hog_features)  # (n, 8100)
            pca_feats = self.pca.transform(hog_scaled)         # (n, 50)

            # 距离² (向量化广播)
            diff = pca_feats[:, np.newaxis, :] - self.codebook[np.newaxis, :, :]
            dist_sq = np.sum(diff ** 2, axis=2)  # (n, K)

            # softmax(-dist²/τ) → 6 维概率
            logits = -dist_sq / self.temperature
            logits_t = torch.from_numpy(logits.astype(np.float32))
            probs = F.softmax(logits_t, dim=1).numpy()  # (n, 6)

            # 填入 T: [0, p1, p2, ..., p6]
            for j, idx in enumerate(hog_indices):
                T[idx, 0] = 0.0
                T[idx, 1:] = probs[j]

        return torch.from_numpy(T).to(device)
