import torch
import argparse
import os
import PIL.Image as Image
import numpy as np
from utils.propagation.ASM import ASM, generate_complex_field
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="扫描重建结果")
    parser.add_argument('--phase_profile', type=str, default='multifocal_1000by1000_5000to7000_step20/result/phase_profile.png', help='相位分布图像路径')
    parser.add_argument('--start_distance', type=float, default=000e-6, help='起始距离(m)')
    parser.add_argument('--end_distance', type=float, default=10000e-6, help='结束距离(m)') 
    parser.add_argument('--step', type=float, default=100e-3, help='步长(m)')
    parser.add_argument('--dx', type=float, default=6.4e-6, help='采样间隔')
    parser.add_argument('--wavelength', type=list, default=[520e-9], help='波长')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--save_dir', type=str, default='scan_result', help='保存目录')
    return parser.parse_args()

def main():
    # 解析参数
    args = parse_args()
    
    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 读取相位分布图像
    phase_image = Image.open(args.phase_profile)
    phase_array = np.array(phase_image)
    
    # 将0-255的灰度值转换为0-2pi的相位
    phase_profile = torch.from_numpy(phase_array).float() * 2 * torch.pi / 255
    phase_profile = phase_profile.to(args.device)
    
    # 扫描距离区间
    distances = torch.arange(args.start_distance, args.end_distance, args.step)
    
    print("正在扫描重建结果...")
    for distance in tqdm(distances):
        args.propagation_distance = distance
        
        # 计算重建结果
        intensity = torch.abs(ASM(phase_profile, args))**2
        intensity = intensity.detach().cpu().numpy()
        
        # 归一化并转换为8位灰度图
        intensity_normalized = (intensity / intensity.max() * 255).astype(np.uint8)
        
        # 保存图像
        filename = f"intensity_{distance*1e6:.0f}um.png"
        filepath = os.path.join(args.save_dir, filename)
        intensity_image = Image.fromarray(intensity_normalized)
        intensity_image.save(filepath)
    
    print(f"扫描完成，结果保存在{args.save_dir}目录中")

if __name__ == "__main__":
    main()
