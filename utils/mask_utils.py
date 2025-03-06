import torch


def create_circle_mask(phase_profile: torch.Tensor) -> torch.Tensor:
    """
    生成一个方形张量的最大内接圆掩码。
    
    参数:
        phase_profile: 张量
        
    返回:
        torch.Tensor: 二值掩码，圆内为1，圆外为0
    """
    # 计算中心点
    center_y = (phase_profile.shape[-2] - 1) / 2
    center_x = (phase_profile.shape[-1] - 1) / 2
    
    # 计算半径 (取较小的边长的一半)
    radius = min(center_y, center_x)
    
    # 创建坐标网格
    y_grid, x_grid = torch.meshgrid(
        torch.arange(phase_profile.shape[-2], dtype=torch.float32),
        torch.arange(phase_profile.shape[-1], dtype=torch.float32),
        indexing='ij'
    )
    
    # 计算每个点到中心的距离
    distance = torch.sqrt((y_grid - center_y)**2 + (x_grid - center_x)**2)
    
    # 创建掩码：距离小于等于半径的点为1，其他为0
    mask = (distance <= radius).float()
    
    return mask