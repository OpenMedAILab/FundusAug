# 环境依赖：pip install opencv-python albumentations tqdm numpy

# 模式选择：
# 1. 真实图片处理：python ultimate_augment_tool_v9.py --input test.txt --mode real --ratio 0.3
# 2. 随机生成图像：python ultimate_augment_tool_v9.py --input test.txt --mode random --ratio 0.3

# 参数说明：
# --input: 修改 XX.txt 来指定需要读取的标签文件。
# --ratio: 修改 x 的值来调整扩增比例（以样本数最多的类别为基准，自动扩充其他所有少数类别）。
# --output_txt: (可选) 加上这个参数可以自定义生成的新标签文件名，方便区分不同任务。
import os
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import random
import argparse
from collections import defaultdict

"""
功能：
1. 多类别支持：自动识别 TXT 中的所有标签 (0, 1, 2, 3...)。
2. 自动平衡：以样本数最多的类别为基准，自动扩充其他所有少数类别。
3. 模式切换：支持读取真实图片 (--mode real) 或 随机生成图片 (--mode random)。
4. 官方方案：采用 Albumentations 官方推荐的医疗影像增强策略。
"""

def parse_args():
    parser = argparse.ArgumentParser(description="Universal Multi-Class Data Augmentation Tool")
    parser.add_argument("--input", type=str, required=True, help="输入的标签文件 (txt)")
    parser.add_argument("--mode", type=str, choices=['real', 'random'], default='real', 
                        help="运行模式：'real' 读取真实图片，'random' 随机生成图片")
    parser.add_argument("--ratio", type=float, default=0.4, help="目标比例：少数类数量 / 多数类数量 (例如 0.4 代表 10:4)")
    parser.add_argument("--output_txt", type=str, default="multi_class_augmented_labels.txt", help="生成的增强标签文件")
    parser.add_argument("--img_size", type=int, default=224, help="随机模式下的图像尺寸")
    return parser.parse_args()

def get_albumentations_transform():
    """Albumentations 官方推荐增强方案"""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=30, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=10, p=0.3),
        A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.3),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5), p=1.0),
            A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
        ], p=0.3),
    ])

def generate_random_image(size):
    """生成随机占位图像"""
    return np.random.randint(0, 256, (size, size, 3), dtype=np.uint8)

def main():
    args = parse_args()
    
    if not os.path.exists(args.input):
        print(f"错误：找不到输入文件 {args.input}")
        return

    # 设置输出目录
    aug_dir = "images/augmented_images"
    os.makedirs(aug_dir, exist_ok=True)

    # 1. 读取并自动分类数据
    class_data = defaultdict(list)
    with open(args.input, 'r') as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                path, label = line.rsplit(',', 1)
                class_data[label.strip()].append(line)
            except ValueError:
                print(f"跳过格式错误的行: {line}")

    # 2. 统计分布
    all_labels = sorted(class_data.keys())
    counts = {label: len(items) for label, items in class_data.items()}
    max_label = max(counts, key=counts.get)
    max_count = counts[max_label]

    print(f"--- 原始数据统计 ---")
    for label in all_labels:
        print(f"标签 {label}: {counts[label]} 个样本")
    print(f"多数类为标签 {max_label}，样本数: {max_count}")
    print(f"运行模式: {args.mode.upper()}")

    new_labels = []
    transform = get_albumentations_transform()

    # 3. 记录所有原始样本
    print(f"\n步骤 1: 记录原始样本...")
    for label in all_labels:
        new_labels.extend(class_data[label])

    # 4. 自动平衡所有少数类
    print(f"\n步骤 2: 正在执行多类别自动平衡 (目标比例: {args.ratio})...")
    for label in all_labels:
        if label == max_label:
            continue
        
        current_count = counts[label]
        target_count = int(max_count * args.ratio)
        needed_aug = target_count - current_count
        
        if needed_aug > 0:
            print(f" -> 正在增强标签 {label}: 需新增 {needed_aug} 个样本...")
            for i in tqdm(range(needed_aug), desc=f"Label {label}"):
                original_line = random.choice(class_data[label])
                img_path, _ = original_line.rsplit(',', 1)
                
                # 获取图像
                if args.mode == 'real':
                    image = cv2.imread(img_path)
                    if image is None: continue
                else:
                    image = generate_random_image(args.img_size)

                # 增强
                augmented = transform(image=image)["image"]
                
                # 命名与保存
                eye_side = "left" if "left" in img_path.lower() else "right"
                filename = os.path.basename(img_path)
                name_part, ext = os.path.splitext(filename)
                aug_name = f"label{label}_{eye_side}_{name_part}_aug_{i}{ext if ext else '.jpg'}"
                save_path = os.path.join(aug_dir, aug_name).replace("\\", "/")
                
                cv2.imwrite(save_path, augmented)
                new_labels.append(f"{save_path},{label}")
        else:
            print(f" -> 标签 {label} 数量已达标，无需增强。")

    # 5. 保存结果
    with open(args.output_txt, 'w') as f:
        for item in new_labels:
            f.write(item + '\n')

    print(f"\n--- 处理完成 ---")
    print(f"最终标签文件: {args.output_txt}")
    print(f"最终总样本数: {len(new_labels)}")
    print(f"增强图像目录: {aug_dir}")

if __name__ == "__main__":
    main()
