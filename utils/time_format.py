def format_seconds(seconds: float) -> str:
    """
    将秒数转换为格式化字符串，例如:
    3661.22 -> "1h1m1.22s"
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60  # 保留小数部分

    parts = []
    if hours > 0:
        parts.append(f"{hours}h ")
    if minutes > 0 or hours > 0:  # 有小时就要显示分钟，即使是0
        parts.append(f"{minutes}m ")
    parts.append(f"{secs:.2f}s")  # 保留2位小数

    return "".join(parts)