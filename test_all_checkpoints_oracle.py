"""
Oracle Restormer 全量测试 - 在所有测试集上评估所有checkpoint的PSNR/SSIM
基于 Restormer/test_all_checkpoints.py 修改，增加 FiLM condition 支持
用法: python test_all_checkpoints_oracle.py [--ckpt_dir ...] [--data_dir ...] [--output ...]
"""

import numpy as np
import os
import argparse
import csv
import sys
import warnings
from tqdm import tqdm
from natsort import natsorted
from glob import glob

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Deraining'))

from basicsr.models.archs.restormer_arch import Restormer
from skimage import img_as_ubyte
import utils  # Deraining/utils.py: load_img, save_img, calculate_psnr, calculate_ssim

# ==================== Y通道转换 (与MATLAB rgb2ycbcr一致) ====================

def rgb2ycbcr_pt(img_rgb):
    """RGB uint8 [0,255] -> Y channel float64 [0,255], 与MATLAB rgb2ycbcr一致"""
    img = img_rgb.astype(np.float64)
    y = 16.0 + (65.481 * img[:,:,0] + 128.553 * img[:,:,1] + 24.966 * img[:,:,2]) / 255.0
    return y


# ==================== 模型加载与推理 ====================

def load_model(yaml_file, weights_path, device='cuda:0'):
    """加载模型到指定GPU"""
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    x['network_g'].pop('type', None)
    model = Restormer(**x['network_g'])

    checkpoint = torch.load(weights_path, map_location='cpu')
    if 'params_ema' in checkpoint:
        state_dict = checkpoint['params_ema']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # strict=False: 允许 checkpoint 缺少 rain_predictor/film_mlp 权重
    # (零初始化，缺失时 FiLM 无效果，等价于 scale=0)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        warnings.warn(f"Missing keys in checkpoint (will use init values): {missing}")
    if unexpected:
        warnings.warn(f"Unexpected keys in checkpoint: {unexpected}")

    model.to(device)
    model.eval()
    return model


def tile_inference(model, input_, tile_size=720, tile_overlap=32):
    """Tile-based推理 (来自demo.py)，用于大图避免OOM。E/W放CPU防超大图OOM"""
    b, c, h, w = input_.shape
    tile = min(tile_size, h, w)
    tile = (tile // 8) * 8  # 确保是8的倍数
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h - tile, stride)) + [max(0, h - tile)]
    w_idx_list = list(range(0, w - tile, stride)) + [max(0, w - tile)]
    E = torch.zeros(b, c, h, w)  # CPU上累积
    W = torch.zeros_like(E)
    for h_idx in h_idx_list:
        for w_idx in w_idx_list:
            in_patch = input_[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
            out_patch = model(in_patch)
            if isinstance(out_patch, tuple):
                out_patch = out_patch[0]
            out_patch = out_patch.cpu()
            E[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(out_patch)
            W[..., h_idx:h_idx+tile, w_idx:w_idx+tile].add_(torch.ones_like(out_patch))
    return E.div_(W)


def process_single_image(file_, gt_dir, model, device, factor, tile_size, tile_overlap):
    """单张图片推理+计算指标，返回 (psnr, ssim) 或 None"""
    torch.cuda.empty_cache()
    img = np.float32(utils.load_img(file_)) / 255.
    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).to(device)

    h, w = input_.shape[2], input_.shape[3]
    H = ((h + factor) // factor) * factor
    W = ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect')

    fname = os.path.splitext(os.path.basename(file_))[0]

    if tile_size and (input_.shape[2] > tile_size or input_.shape[3] > tile_size):
        restored = tile_inference(model, input_,
                                  tile_size=tile_size, tile_overlap=tile_overlap)
        restored = restored[:, :, :h, :w]  # tile_inference返回CPU tensor
    else:
        restored = model(input_)
        if isinstance(restored, tuple):
            restored = restored[0]
        restored = restored[:, :, :h, :w].cpu()

    restored = torch.clamp(restored, 0, 1).detach().permute(0, 2, 3, 1).squeeze(0).numpy()
    restored = img_as_ubyte(restored)

    gt_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp']:
        candidate = os.path.join(gt_dir, fname + ext)
        if os.path.exists(candidate):
            gt_path = candidate
            break
    if gt_path is None:
        warnings.warn(f"GT not found for {os.path.basename(file_)}, skipping")
        return None

    gt_img = utils.load_img(gt_path)
    if restored.shape != gt_img.shape:
        mh = min(restored.shape[0], gt_img.shape[0])
        mw = min(restored.shape[1], gt_img.shape[1])
        restored = restored[:mh, :mw, :]
        gt_img = gt_img[:mh, :mw, :]

    restored_y = rgb2ycbcr_pt(restored)
    gt_y = rgb2ycbcr_pt(gt_img)
    p = utils.calculate_psnr(restored_y, gt_y, border=0)
    s = utils.calculate_ssim(restored_y, gt_y, border=0)
    return (p, s)


def _gpu_worker(gpu_id, file_list, gt_dir, yaml_file, weights_path, factor,
                tile_size, tile_overlap, result_queue):
    """多进程worker: 在指定GPU上加载模型并推理一组图片"""
    device = f'cuda:{gpu_id}'
    model = load_model(yaml_file, weights_path, device=device)
    local_results = []
    with torch.no_grad():
        for f in file_list:
            r = process_single_image(f, gt_dir, model, device, factor,
                                     tile_size, tile_overlap)
            local_results.append(r)
    model.cpu()
    del model
    torch.cuda.empty_cache()
    result_queue.put((gpu_id, local_results))


def evaluate_checkpoint(yaml_file, weights_path, num_gpus, datasets, data_dir,
                        tile_size, tile_overlap):
    """多GPU多进程并行评估一个checkpoint在所有数据集上的PSNR/SSIM"""
    factor = 8
    results = {}
    all_psnr, all_ssim = [], []

    for dataset in datasets:
        inp_dir = os.path.join(data_dir, 'test', dataset, 'input')
        gt_dir = os.path.join(data_dir, 'test', dataset, 'target')

        if not os.path.exists(inp_dir) or not os.path.exists(gt_dir):
            print(f"  [跳过] {dataset}: 目录不存在")
            continue

        files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
        if len(files) == 0:
            print(f"  [跳过] {dataset}: 无图片")
            continue

        psnr_list, ssim_list = [], []

        # 将文件均匀分配到各GPU
        chunks = [[] for _ in range(num_gpus)]
        for i, f in enumerate(files):
            chunks[i % num_gpus].append(f)

        result_queue = mp.Queue()
        processes = []
        for gpu_id in range(num_gpus):
            if len(chunks[gpu_id]) == 0:
                continue
            p = mp.Process(
                target=_gpu_worker,
                args=(gpu_id, chunks[gpu_id], gt_dir, yaml_file, weights_path,
                      factor, tile_size, tile_overlap, result_queue)
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        # 收集结果
        while not result_queue.empty():
            gpu_id, local_res = result_queue.get()
            for r in local_res:
                if r is not None:
                    psnr_list.append(r[0])
                    ssim_list.append(r[1])

        if len(psnr_list) > 0:
            avg_psnr = np.mean(psnr_list)
            avg_ssim = np.mean(ssim_list)
            results[dataset] = {'psnr': avg_psnr, 'ssim': avg_ssim, 'count': len(psnr_list)}
            all_psnr.extend(psnr_list)
            all_ssim.extend(ssim_list)
            print(f"  {dataset}: PSNR={avg_psnr:.4f} SSIM={avg_ssim:.6f} ({len(psnr_list)} images)")

    if len(all_psnr) > 0:
        results['Average'] = {'psnr': np.mean(all_psnr), 'ssim': np.mean(all_ssim), 'count': len(all_psnr)}
    return results


def main():
    parser = argparse.ArgumentParser(description='RainPredictor Restormer - 在所有测试集上评估所有checkpoint')
    parser.add_argument('--ckpt_dir', type=str,
                        default='../Restormer/experiments/Deraining_Oracle_Restormer/models/',
                        help='checkpoint目录')
    parser.add_argument('--data_dir', type=str,
                        default='./Deraining/Datasets/',
                        help='数据集根目录')
    parser.add_argument('--yaml_file', type=str,
                        default='./Deraining/Options/Deraining_RainPredictor_8xA100.yml',
                        help='模型配置文件')
    parser.add_argument('--output', type=str,
                        default='./oracle_eval_results.csv',
                        help='输出CSV文件路径')
    parser.add_argument('--step', type=int, default=1,
                        help='每隔多少个checkpoint测试一次 (1=全部测试)')
    parser.add_argument('--tile', type=int, default=720,
                        help='Tile大小，大图自动分块推理')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Tile重叠像素数')
    parser.add_argument('--num_gpus', type=int, default=None,
                        help='使用GPU数量 (默认自动检测)')
    args = parser.parse_args()

    num_gpus = args.num_gpus or torch.cuda.device_count()
    num_gpus = max(1, min(num_gpus, torch.cuda.device_count()))
    print(f"使用 {num_gpus} 张GPU, tile_size={args.tile}, tile_overlap={args.tile_overlap}")

    datasets = ['LHP-RAIN', 'RainDrop', 'RainDS-Real', 'RainDS-Syn',
                'RealRain-1k', 'SynRain-13k', 'WeatherBench']

    # 收集所有checkpoint (排除latest)
    ckpt_files = natsorted(glob(os.path.join(args.ckpt_dir, 'net_g_*.pth')))
    ckpt_files = [f for f in ckpt_files if 'latest' not in os.path.basename(f)]

    if args.step > 1:
        ckpt_files = ckpt_files[::args.step]

    # CSV header
    header = ['Checkpoint', 'Iteration']
    for d in datasets:
        header.extend([f'{d}_PSNR', f'{d}_SSIM'])
    header.extend(['Avg_PSNR', 'Avg_SSIM'])

    # Resume: 读取已有CSV中已完成的checkpoint，跳过它们
    all_results = []
    done_ckpts = set()
    if os.path.exists(args.output):
        with open(args.output, 'r', newline='') as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
            for row in reader:
                if len(row) >= 2 and row[0].strip():
                    all_results.append(row)
                    done_ckpts.add(row[0])
        print(f"从已有CSV恢复: 已完成 {len(done_ckpts)} 个checkpoint")

    ckpt_files = [f for f in ckpt_files if os.path.basename(f) not in done_ckpts]

    print(f"剩余 {len(ckpt_files)} 个checkpoint待测试")
    print(f"测试数据集: {datasets}")
    print(f"结果将保存到: {args.output}")
    print("=" * 80)

    for ckpt_path in ckpt_files:
        ckpt_name = os.path.basename(ckpt_path)
        iter_num = ckpt_name.replace('net_g_', '').replace('.pth', '')
        print(f"\n{'='*80}")
        print(f"测试 {ckpt_name} (iter {iter_num})")
        print(f"{'='*80}")

        results = evaluate_checkpoint(args.yaml_file, ckpt_path, num_gpus, datasets,
                                      args.data_dir, args.tile, args.tile_overlap)

        row = [ckpt_name, iter_num]
        for d in datasets:
            if d in results:
                row.extend([f"{results[d]['psnr']:.4f}", f"{results[d]['ssim']:.6f}"])
            else:
                row.extend(['N/A', 'N/A'])
        if 'Average' in results:
            row.extend([f"{results['Average']['psnr']:.4f}", f"{results['Average']['ssim']:.6f}"])
        else:
            row.extend(['N/A', 'N/A'])
        all_results.append(row)

        # 每测完一个checkpoint就写入CSV (防止中断丢失)
        with open(args.output, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(all_results)

    # 打印汇总
    print("\n\n" + "=" * 120)
    print("汇总结果表格")
    print("=" * 120)
    fmt_header = f"{'Checkpoint':<20} {'Iter':>10}"
    for d in datasets:
        fmt_header += f" {d+'_P':>10} {d+'_S':>10}"
    fmt_header += f" {'Avg_PSNR':>10} {'Avg_SSIM':>10}"
    print(fmt_header)
    print("-" * len(fmt_header))
    for row in all_results:
        line = f"{row[0]:<20} {row[1]:>10}"
        for v in row[2:]:
            line += f" {v:>10}"
        print(line)

    # 找最佳checkpoint
    best_idx, best_val = -1, -1
    for i, row in enumerate(all_results):
        try:
            val = float(row[-2])
            if val > best_val:
                best_val = val
                best_idx = i
        except (ValueError, IndexError):
            pass
    if best_idx >= 0:
        print(f"\n最佳checkpoint (按平均PSNR): {all_results[best_idx][0]} "
              f"(iter {all_results[best_idx][1]}) "
              f"Avg_PSNR={all_results[best_idx][-2]} "
              f"Avg_SSIM={all_results[best_idx][-1]}")

    print(f"\n结果已保存到: {args.output}")


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
