import torch
import torch.optim as optim
import torch.nn.functional as F
import argparse
import os
import PIL.Image as Image
from torchvision import transforms
from tqdm import tqdm
from utils.factories import propagation_function_factory, phase_profile_mask_factory
from utils.utils import format_distance
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
    parser.add_argument('--target_dir', type=str, default='multifocal', help='target image')
    parser.add_argument('--propagation_funtion', type=str, default='default', help='propagation function')
    parser.add_argument('--phase_profile_mask', type=str, default='circle', help='phase profile mask')

    # meta parameters
    parser.add_argument('--dx', type=float, default=300e-9, help='meta period')
    parser.add_argument('--wavelength', type=list, default=[532e-9], help='wavelength list')

    # optimization parameters
    parser.add_argument('--device', type=str, default='cuda', help='device')
    parser.add_argument('--iterations', type=int, default=20, help='Number of iterations for optimization')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate for the optimizer')
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
    phase_profile_masked = phase_profile * phase_profile_mask
    
    # 定义优化器
    optimizer = optim.SGD([phase_profile], lr=args.learning_rate)

    # 添加进度条，在进度条上显示loss
    for i in tqdm(range(args.iterations), desc="优化进度"):
        optimizer.zero_grad()
        total_loss = 0
        
        # 对所有传播距离计算损失
        for propagation_distance, target_image in target_images.items():
            args.propagation_distance = propagation_distance
            intensity = propagation_function(phase_profile_masked, args)
            loss = F.mse_loss(intensity.unsqueeze(0), target_image)
            total_loss += loss
        
        # 只进行一次反向传播
        total_loss.backward()
        optimizer.step()
        tqdm.write(f"总损失: {total_loss.item()}")

        # 如果是最后一次迭代，以图片形式保存phase profile，将其范围限制在0到2pi，和在左右深度的重建结果
        if i == args.iterations - 1:
            phase_profile = phase_profile % (2 * torch.pi) / (2 * torch.pi)
            phase_profile = phase_profile.detach().cpu().numpy()
            phase_profile = Image.fromarray(phase_profile)
            phase_profile.save(os.path.join(args.target_dir, "result/phase_profile.png"))

            for propagation_distance, target_image in target_images.items():
                args.propagation_distance = propagation_distance
                intensity = propagation_function(phase_profile, args)
                intensity = intensity.detach().cpu().numpy()
                intensity = Image.fromarray(intensity)
                intensity.save(os.path.join(args.target_dir, "result/intensity_" + format_distance(propagation_distance) + ".png"))

            # 保存 args
            with open(os.path.join(args.target_dir, "result/args.json"), "w") as f:
                json.dump(args, f)      

    
if __name__ == "__main__":
    main()
