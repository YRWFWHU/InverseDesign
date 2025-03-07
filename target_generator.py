import os
import numpy as np
import PIL.Image as Image
from tqdm import tqdm

def generate_target_images():
    """
    生成一系列目标图像，每个图像中心有一个6像素直径的圆，其余部分为0。
    图像分辨率为3000×3000，保存为灰度图。
    """
    # 创建保存目录
    save_dir = 'multifocal_1000by1000_2500to4500_step100'
    os.makedirs(save_dir, exist_ok=True)
    
    # 图像尺寸
    image_size = 1000
    
    # 创建基础图像（全黑）
    base_image = np.zeros((image_size, image_size), dtype=np.uint8)
    
    # 计算中心点
    center_y, center_x = image_size // 2, image_size // 2
    
    # 创建圆形掩码（直径6像素）
    radius = 3  # 半径为3，直径为6
    y, x = np.ogrid[:image_size, :image_size]
    mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
    
    # 将掩码应用到基础图像
    base_image[mask] = 255  # 设置圆内像素为白色（值为255）
    
    # 生成60张图像，从3000um到3600um，步长为12um
    distances = np.arange(5000, 5500, 100)
    
    print("正在生成目标图像...")
    for distance in tqdm(distances):
        # 文件名
        filename = f"{distance}um.png"
        filepath = os.path.join(save_dir, filename)
        
        # 保存图像
        img = Image.fromarray(base_image)
        img.save(filepath)
    
    print(f"已生成{len(distances)}张目标图像，保存在{save_dir}目录中")

if __name__ == "__main__":
    generate_target_images()
