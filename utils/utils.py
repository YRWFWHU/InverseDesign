def format_distance(distance: float) -> str:
    """将以米为单位的距离值转换为带有适当单位的字符串表示。
    
    例如：
    - 5e-6 -> '5um'
    - 500e-9 -> '500nm'
    - 1e-3 -> '1mm'
    """
    if distance >= 1e-3:  # 毫米或更大
        return f"{distance * 1e3:.0f}mm"
    elif distance >= 1e-6:  # 微米
        return f"{distance * 1e6:.0f}um"
    else:  # 纳米
        return f"{distance * 1e9:.0f}nm"
