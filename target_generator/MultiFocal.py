import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm
import argparse
import json
import time

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="生成多焦点目标图像")
    
    # 图像参数
    parser.add_argument('--image_size', type=int, default=1024, help='图像分辨率')
    parser.add_argument('--start_depth', type=int, default=180000, help='起始深度')
    parser.add_argument('--end_depth', type=int, default=220000, help='终点深度')
    parser.add_argument('--step', type=int, default=400, help='深度步长')
    parser.add_argument('--unit', type=str, default='um', help='深度单位单位')
    
    # 焦点参数
    parser.add_argument('--focus_start_depth', type=int, default=190000, help='聚焦起始深度')
    parser.add_argument('--focus_end_depth', type=int, default=210000, help='聚焦终点深度')
    parser.add_argument('--focus_radius', type=int, default=3, help='焦点半径(像素)')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default=None, help='输出目录名称')
    
    return parser.parse_args()

def generate_target_images():
    """
    生成一系列目标图像，在指定的聚焦深度区间内生成中心有圆形焦点的图像，
    在其他深度区间生成全黑图像。
    """
    # 解析参数
    args = parse_arguments()

    if args.unit == 'um':
        unit = 1e-6
    elif args.unit == 'nm':
        unit = 1e-9
    elif args.unit == 'mm':
        unit = 1e-3
    elif args.unit == 'cm':
        unit = 1e-2
    elif args.unit == 'dm':
        unit = 1e-1
    elif args.unit == 'm':
        unit = 1
    else:
        raise ValueError(f"不支持的深度单位: {args.unit}")
    
    args.start_depth = args.start_depth * unit
    args.end_depth = args.end_depth * unit
    args.step = args.step * unit
    args.focus_start_depth = args.focus_start_depth * unit
    args.focus_end_depth = args.focus_end_depth * unit
    
    # 创建保存目录
    if args.output_dir is None:
        save_dir = f'target/MultiFocal/{time.strftime("%Y%m%d%H%M%S", time.localtime())}'
    else:
        save_dir = f'target/MultiFocal/{args.output_dir}'
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 图像尺寸
    image_size = args.image_size
    
    # 创建基础图像（全黑）
    black_image = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 计算中心点
    center_y, center_x = image_size // 2, image_size // 2
    
    # 创建圆形掩码
    radius = args.focus_radius
    y, x = np.ogrid[:image_size, :image_size]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # 创建带焦点的图像
    focus_image = black_image.copy()
    focus_image[mask] = 255  # 设置圆内像素为白色（值为255）
    
    # 生成图像，从start_depth到end_depth（包含），步长为step
    distances = np.arange(args.start_depth, args.end_depth, args.step)
    
    print("正在生成目标图像...")
    for distance in tqdm(distances):
        # 文件名
        filename = f"{distance/unit:.0f}{args.unit}.png"
        filepath = os.path.join(save_dir, filename)
        
        # 判断是否在聚焦深度区间内
        if args.focus_start_depth <= distance <= args.focus_end_depth:
            # 在聚焦区间内，使用带焦点的图像
            img = Image.fromarray(focus_image)
        else:
            # 在聚焦区间外，使用全黑图像
            img = Image.fromarray(black_image)
        
        # 保存图像
        img.save(filepath)
    
    # 保存参数到json文件
    with open(os.path.join(save_dir, 'parameters.json'), 'w') as f:
        json.dump(args.__dict__, f)
    
    
    print(f"已生成{len(distances)}张目标图像，保存在{save_dir}目录中")
    print(f"监督区间: {args.start_depth/unit:.0f}um - {args.end_depth/unit:.0f}um")
    print(f"聚焦区间: {args.focus_start_depth/unit:.0f}um - {args.focus_end_depth/unit:.0f}um")
    print(f"焦点半径: {args.focus_radius}像素")

if __name__ == "__main__":
    generate_target_images()
