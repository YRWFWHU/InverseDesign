import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import PIL.Image as Image
from torchvision import transforms
from tqdm import tqdm
from utils.factories import propagation_function_factory, phase_profile_mask_factory
import json

def load_target(args: argparse.Namespace) -> dict[float, torch.Tensor]:
    # 读取target_dir中所有图像文件，并将其名字作为key，图像作为value，返回一个字典
    target_images = {}
    for file in os.listdir(args.target_dir):
        if file.endswith('.png'):
            file_name = file.split('.')[0]
            if 'nm' in file_name:
                propagation_distance = float(file_name.split('nm')[0]) * 1e-9
            elif 'um' in file_name:
                propagation_distance = float(file_name.split('um')[0]) * 1e-6
            elif 'mm' in file_name:
                propagation_distance = float(file_name.split('mm')[0]) * 1e-3
            else:
                raise ValueError(f"Invalid propagation distance unit: {file_name}")
            
            image = Image.open(os.path.join(args.target_dir, file))
            transform = transforms.ToTensor()
            image_tensor = transform(image)
            image_tensor = image_tensor.to(args.device)
            target_images[propagation_distance] = image_tensor
            
    return target_images

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inverse design")

    # task parameters
    parser.add_argument('--target_dir', type=str, default='multifocal_1024by1024_190000to210000_step20', help='target image')
    parser.add_argument('--propagation_funtion', type=str, default='default', help='propagation function')
    parser.add_argument('--phase_profile_mask', type=str, default='rect', choices=['circle', 'rect'])

    # meta parameters
    parser.add_argument('--dx', type=float, default=6.4e-6, help='meta period')
    parser.add_argument('--wavelength', type=float, default=520e-9, help='wavelength')

    # optimization parameters
    parser.add_argument('--device', type=str, default='cuda:1', help='device')
    parser.add_argument('--iterations', type=int, default=300, help='Number of iterations for optimization')
    parser.add_argument('--learning_rate', type=float, default=1e-1, help='Learning rate for the optimizer')
    return parser.parse_args()

def main() -> None:
    args = parse_arguments()

    # 根据args.propagation_funtion选择传播函数
    propagation_function = propagation_function_factory(args.propagation_funtion)

    target_images = load_target(args)

    # 构建phase profile，分辨率与target_images相同，初始化为随机值，并乘以mask
    target = list(target_images.values())[0]
    phase_profile = torch.randn((target.shape[-2], target.shape[-1]), requires_grad=True, device=args.device)
    phase_profile_mask = phase_profile_mask_factory(phase_profile, 
                                                    args.phase_profile_mask).to(args.device)
    
    
    # 定义优化器
    optimizer = optim.Adam([phase_profile], lr=args.learning_rate)

    # 添加进度条，在进度条上显示loss
    pbar = tqdm(range(args.iterations), desc="优化进度")
    for i in pbar:
        
        optimizer.zero_grad()
        total_loss = 0

        phase_profile_masked = phase_profile * phase_profile_mask
        
        # 对所有传播距离计算损失
        for propagation_distance, target_image in target_images.items():
            enhanced_target = target_image * 10
            args.propagation_distance = propagation_distance
            intensity = propagation_function(phase_profile_masked, args)
            loss = F.mse_loss(intensity.unsqueeze(0), enhanced_target)
            total_loss += loss
        
        # 只进行一次反向传播
        total_loss.backward(retain_graph=True if i < args.iterations - 1 else False)
        optimizer.step()
        
        # 更新进度条显示当前loss
        pbar.set_postfix({"loss": f"{total_loss.item():.4f}"})

        # 如果是最后一次迭代，以图片形式保存phase profile，将其范围限制在0到2pi，和在左右深度的重建结果
        if i == args.iterations - 1:
            # 确保结果目录存在
            result_dir = os.path.join(args.target_dir, "result")
            os.makedirs(result_dir, exist_ok=True)
            
            # 处理相位分布
            phase_profile_final = phase_profile_masked.detach().clone()
            phase_profile_final = phase_profile_final % (2 * torch.pi) / (2 * torch.pi)
            phase_profile_final = phase_profile_final.cpu().numpy()
            
            # 将相位分布转换为0-255的灰度图像
            phase_image = (phase_profile_final * 255).astype('uint8')
            phase_image = Image.fromarray(phase_image)
            phase_image.save(os.path.join(result_dir, "phase_profile.png"))

            # 保存每个传播距离的强度图像
            min_distance = 180000e-6
            max_distance = 220000e-6

            # 首先收集所有深度的强度图像
            all_intensities = []
            propagation_distances = []
            
            for propagation_distance in torch.arange(min_distance, max_distance, 100e-6):
                args.propagation_distance = propagation_distance
                intensity = propagation_function(phase_profile_masked, args)
                intensity = intensity.detach().cpu().numpy()
                all_intensities.append(intensity)
                propagation_distances.append(propagation_distance)
            
            # 找到所有强度图像中的全局最大值，用于统一归一化
            global_max = max([intensity.max() for intensity in all_intensities])
            
            # 使用全局最大值对所有图像进行相同的归一化
            for i, (intensity, propagation_distance) in enumerate(zip(all_intensities, propagation_distances)):
                intensity_normalized = (intensity / global_max * 255).astype('uint8')
                intensity_image = Image.fromarray(intensity_normalized)
                intensity_image.save(os.path.join(result_dir, f"intensity_{propagation_distance*1e6:.0f}um" + ".png"))

            # 保存 args (需要将args转换为可序列化的字典)
            args_dict = vars(args)
            # 将不可序列化的对象转换为字符串
            for key, value in args_dict.items():
                if isinstance(value, torch.Tensor):
                    args_dict[key] = value.detach().cpu().tolist()
                elif not isinstance(value, (str, int, float, bool, list, dict, type(None))):
                    args_dict[key] = str(value)
                    
            with open(os.path.join(result_dir, "args.json"), "w") as f:
                json.dump(args_dict, f, indent=4)

    
if __name__ == "__main__":
    main()
