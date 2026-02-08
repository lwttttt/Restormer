#!/usr/bin/env python3
"""
准备训练数据脚本：将 sampled_dataset_balanced 中的多个子数据集
通过符号链接合并为 Restormer 训练所需的单一目录结构。

注意：
- 不会修改 sampled_dataset_balanced 原始数据
- 仅创建符号链接，不复制文件
- 合并后的目录结构：
    Deraining/Datasets/train/AllRain/input/   (所有训练输入图像的符号链接)
    Deraining/Datasets/train/AllRain/target/  (所有训练GT图像的符号链接)
    Deraining/Datasets/test/<DatasetName>/input/   (各测试集)
    Deraining/Datasets/test/<DatasetName>/target/
"""

import os
import sys
import json
import shutil
from pathlib import Path

# ============ 配置 ============
BALANCED_DATASET_ROOT = "/home/zyb/Downloads/rain/sampled_dataset_balanced"
RESTORMER_ROOT = "/home/zyb/Downloads/rain/Restormer"
OUTPUT_TRAIN_DIR = os.path.join(RESTORMER_ROOT, "Deraining/Datasets/train/AllRain")
OUTPUT_TEST_DIR = os.path.join(RESTORMER_ROOT, "Deraining/Datasets/test")

# 所有子数据集
DATASETS = [
    "LHP-RAIN",
    "RainDrop",
    "RainDS-Real",
    "RainDS-Syn",
    "RealRain-1k",
    "SynRain-13k",
    "WeatherBench",
]


def main():
    print("=" * 60)
    print("Restormer 训练数据准备脚本")
    print("=" * 60)
    print(f"数据源: {BALANCED_DATASET_ROOT}")
    print(f"训练输出: {OUTPUT_TRAIN_DIR}")
    print(f"测试输出: {OUTPUT_TEST_DIR}")
    print()

    # 检查源数据是否存在
    if not os.path.isdir(BALANCED_DATASET_ROOT):
        print(f"[错误] 找不到数据源目录: {BALANCED_DATASET_ROOT}")
        sys.exit(1)

    # 读取元数据
    metadata_path = os.path.join(BALANCED_DATASET_ROOT, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
        print(f"数据集总采样数: {metadata['total_sampled']}")
        print(f"训练集: {metadata['subset_distribution']['train']}")
        print(f"测试集: {metadata['subset_distribution']['test']}")
        print()

    # ---- 创建训练集合并目录 ----
    train_input_dir = os.path.join(OUTPUT_TRAIN_DIR, "input")
    train_target_dir = os.path.join(OUTPUT_TRAIN_DIR, "target")

    # 如果已存在，先清理（只删除符号链接目录，不影响原始数据）
    for d in [train_input_dir, train_target_dir]:
        if os.path.exists(d):
            print(f"[清理] 删除已有目录: {d}")
            shutil.rmtree(d)

    os.makedirs(train_input_dir, exist_ok=True)
    os.makedirs(train_target_dir, exist_ok=True)

    train_count = 0
    skipped = 0
    dataset_stats = {}

    for ds_name in DATASETS:
        ds_train_input = os.path.join(BALANCED_DATASET_ROOT, ds_name, "train", "input")
        ds_train_target = os.path.join(BALANCED_DATASET_ROOT, ds_name, "train", "target")

        if not os.path.isdir(ds_train_input):
            print(f"[跳过] {ds_name}: 找不到 train/input 目录")
            continue

        input_files = sorted(os.listdir(ds_train_input))
        target_files = sorted(os.listdir(ds_train_target))

        if len(input_files) != len(target_files):
            print(f"[警告] {ds_name}: input({len(input_files)}) 和 target({len(target_files)}) 数量不匹配!")

        ds_count = 0
        for fname in input_files:
            src_input = os.path.join(ds_train_input, fname)
            src_target = os.path.join(ds_train_target, fname)
            dst_input = os.path.join(train_input_dir, fname)
            dst_target = os.path.join(train_target_dir, fname)

            # 检查target是否存在
            if not os.path.exists(src_target):
                print(f"  [跳过] {fname}: target不存在")
                skipped += 1
                continue

            # 检查文件名冲突
            if os.path.exists(dst_input):
                print(f"  [冲突] {fname}: 文件名已存在，跳过")
                skipped += 1
                continue

            os.symlink(os.path.abspath(src_input), dst_input)
            os.symlink(os.path.abspath(src_target), dst_target)
            ds_count += 1

        train_count += ds_count
        dataset_stats[ds_name] = ds_count
        print(f"[训练] {ds_name}: 链接了 {ds_count} 对图像")

    print(f"\n训练集总计: {train_count} 对图像 (跳过 {skipped})")
    print(f"各数据集分布: {json.dumps(dataset_stats, indent=2)}")

    # ---- 创建测试集目录 ----
    print(f"\n{'=' * 60}")
    print("创建测试集目录...")

    test_total = 0
    for ds_name in DATASETS:
        ds_test_input = os.path.join(BALANCED_DATASET_ROOT, ds_name, "test", "input")
        ds_test_target = os.path.join(BALANCED_DATASET_ROOT, ds_name, "test", "target")

        if not os.path.isdir(ds_test_input):
            print(f"[跳过] {ds_name}: 找不到 test 目录")
            continue

        out_test_input = os.path.join(OUTPUT_TEST_DIR, ds_name, "input")
        out_test_target = os.path.join(OUTPUT_TEST_DIR, ds_name, "target")

        # 清理已有
        for d in [out_test_input, out_test_target]:
            if os.path.exists(d):
                shutil.rmtree(d)
            os.makedirs(d, exist_ok=True)

        test_files = sorted(os.listdir(ds_test_input))
        ds_test_count = 0
        for fname in test_files:
            src_input = os.path.join(ds_test_input, fname)
            src_target = os.path.join(ds_test_target, fname)
            if not os.path.exists(src_target):
                continue
            os.symlink(os.path.abspath(src_input), os.path.join(out_test_input, fname))
            os.symlink(os.path.abspath(src_target), os.path.join(out_test_target, fname))
            ds_test_count += 1

        test_total += ds_test_count
        print(f"[测试] {ds_name}: 链接了 {ds_test_count} 对图像")

    print(f"\n测试集总计: {test_total} 对图像")

    # ---- 验证 ----
    print(f"\n{'=' * 60}")
    print("验证...")
    actual_train_input = len(os.listdir(train_input_dir))
    actual_train_target = len(os.listdir(train_target_dir))
    print(f"训练 input: {actual_train_input} 文件")
    print(f"训练 target: {actual_train_target} 文件")
    assert actual_train_input == actual_train_target, "input和target数量不匹配!"
    print("验证通过!")

    # ---- 输出摘要 ----
    print(f"\n{'=' * 60}")
    print("准备完成! 目录结构:")
    print(f"  训练集: {OUTPUT_TRAIN_DIR}/")
    print(f"    input/  ({actual_train_input} 图像)")
    print(f"    target/ ({actual_train_target} 图像)")
    print(f"  测试集: {OUTPUT_TEST_DIR}/")
    for ds_name in DATASETS:
        test_dir = os.path.join(OUTPUT_TEST_DIR, ds_name, "input")
        if os.path.isdir(test_dir):
            count = len(os.listdir(test_dir))
            print(f"    {ds_name}/ ({count} 图像)")

    print(f"\n下一步: 使用以下命令开始训练:")
    print(f"  ./train.sh Deraining/Options/Deraining_AllRain_8xA100.yml")


if __name__ == "__main__":
    main()
