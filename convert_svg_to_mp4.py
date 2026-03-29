#!/usr/bin/env python3
"""
将 SVG 动画批量转换为 MP4 视频格式
用于在 PowerPoint 等应用中展示 MAPF 动画

使用 resvg 渲染 SVG 帧，ffmpeg 创建视频
"""

import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

from lxml import etree as ET


def parse_svg_duration(svg_path: str) -> int:
    """
    解析 SVG 动画时长 (毫秒)
    """
    with open(svg_path, 'rb') as f:
        content = f.read()
    
    # 查找所有 dur 属性
    dur_matches = re.findall(rb'dur="([\d.]+)(s|ms)"', content)
    
    max_duration_ms = 1000  # 默认 1 秒
    for value, unit in dur_matches:
        value = float(value.decode())
        unit = unit.decode()
        if unit == 's':
            value *= 1000
        if value > max_duration_ms:
            max_duration_ms = value
    
    return int(max_duration_ms)


def interpolate_value(values: List[str], key_times: List[float], t: float) -> str:
    """
    在关键帧之间插值
    
    Args:
        values: 关键帧值列表
        key_times: 关键帧时间 (0-1)
        t: 当前时间 (0-1)
    
    Returns:
        插值后的值
    """
    if not values or not key_times:
        return ""
    
    if len(values) == 1:
        return values[0]
    
    # 找到当前时间所在的区间
    for i in range(len(key_times) - 1):
        if key_times[i] <= t <= key_times[i + 1]:
            # 尝试数值插值
            try:
                v1 = float(values[i])
                v2 = float(values[i + 1])
                ratio = (t - key_times[i]) / (key_times[i + 1] - key_times[i])
                return str(int(v1 + (v2 - v1) * ratio))
            except ValueError:
                # 非数值，使用起始值
                return values[i]
    
    # 超出范围，返回最后一个值
    return values[-1]


def render_svg_frame(svg_path: str, time_ms: int, output_png: str, output_size: Tuple[int, int] = (512, 377)):
    """
    使用 resvg 渲染 SVG 在指定时间点的帧
    """
    # 读取 SVG 内容 (字节模式)
    with open(svg_path, 'rb') as f:
        svg_content = f.read()
    
    # 解析 SVG
    root = ET.fromstring(svg_content)
    ns = {'svg': 'http://www.w3.org/2000/svg'}
    
    # 对于每个 animate 元素，计算当前时间点的值并应用
    animate_elements = root.xpath('//svg:animate', namespaces=ns)
    
    for anim in animate_elements:
        parent = anim.getparent()
        if parent is None:
            continue
        
        attr_name = anim.get('attributeName', '')
        dur_str = anim.get('dur', '1s')
        
        # 解析时长
        match = re.match(r'([\d.]+)(s|ms)?', dur_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or 's'
            if unit == 's':
                duration_ms_anim = value * 1000
            else:
                duration_ms_anim = value
        else:
            duration_ms_anim = 1000
        
        # 获取关键帧时间和值
        keyTimes_str = anim.get('keyTimes', '')
        values_str = anim.get('values', '')
        
        if not keyTimes_str or not values_str:
            continue
        
        key_times_anim = [float(t) for t in keyTimes_str.split(';')]
        values = values_str.split(';')
        
        # 计算当前时间点在动画中的位置
        t = (time_ms % int(duration_ms_anim)) / int(duration_ms_anim)
        
        # 插值得到当前值
        current_value = interpolate_value(values, key_times_anim, t)
        
        # 应用值到父元素 (支持更多属性)
        if attr_name in ['cx', 'cy', 'r', 'fill', 'stroke', 'stroke-width', 'x', 'y', 'width', 'height', 'opacity', 'visibility']:
            parent.set(attr_name, current_value)
        
        # 移除 animate 元素
        parent.remove(anim)
    
    # 序列化修改后的 SVG (字节)
    modified_svg = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    
    # 写入临时文件
    with tempfile.NamedTemporaryFile(mode='wb', suffix='.svg', delete=False) as f:
        f.write(modified_svg)
        temp_svg = f.name
    
    try:
        # 使用 resvg 渲染，设置白色背景
        cmd = [
            'resvg',
            '--width', str(output_size[0]),
            '--height', str(output_size[1]),
            '--background', 'white',
            temp_svg,
            output_png
        ]
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        os.unlink(temp_svg)


def convert_svg_to_mp4(svg_path: str, output_path: str, fps: int = 30, scale: int = 2):
    """
    将 SVG 动画转换为 MP4 视频
    
    Args:
        svg_path: SVG 文件路径
        output_path: 输出 MP4 文件路径
        fps: 帧率
        scale: 缩放倍数（1=原始尺寸，2=2 倍尺寸，以此类推）
    """
    # 解析 SVG 获取动画时长
    duration_ms = parse_svg_duration(svg_path)
    print(f"  动画时长：{duration_ms}ms")
    
    # 计算总帧数
    total_frames = max(int(duration_ms / 1000 * fps), 1)
    
    # 输出尺寸 (原始 512x377，可根据 scale 放大)
    base_width = 512
    base_height = 377
    output_size = (base_width * scale, base_height * scale)
    print(f"  输出尺寸：{output_size[0]}x{output_size[1]}")
    
    # 创建临时目录存储帧
    temp_dir = Path(tempfile.mkdtemp())
    
    # 渲染所有帧
    for frame_idx in range(total_frames):
        time_ms = int(frame_idx * 1000 / fps)
        frame_path = temp_dir / f"frame_{frame_idx:04d}.png"
        render_svg_frame(svg_path, time_ms, str(frame_path), output_size)
        
        if (frame_idx + 1) % 100 == 0:
            print(f"    已渲染 {frame_idx + 1}/{total_frames} 帧")
    
    # 使用 ffmpeg 创建视频 (使用 scale 滤镜确保宽高是 2 的倍数)
    cmd = [
        'ffmpeg',
        '-y',
        '-framerate', str(fps),
        '-i', str(temp_dir / 'frame_%04d.png'),
        '-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',  # 确保宽高是 2 的倍数
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-movflags', '+faststart',
        output_path
    ]
    
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    
    # 清理临时文件
    for i in range(total_frames):
        frame_path = temp_dir / f"frame_{i:04d}.png"
        if frame_path.exists():
            os.unlink(frame_path)
    os.rmdir(temp_dir)
    
    print(f"  生成 {total_frames} 帧 -> {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='将 SVG 动画转换为 MP4 视频')
    parser.add_argument('--scale', type=int, default=3, help='缩放倍数（1=512x376, 2=1024x752, 3=1536x1128, 4=2048x1504）')
    parser.add_argument('--fps', type=int, default=30, help='帧率')
    args = parser.parse_args()
    
    # 配置路径
    script_dir = Path(__file__).parent
    renders_dir = script_dir / "renders"
    output_dir = renders_dir / "videos"
    
    # 创建输出目录
    output_dir.mkdir(exist_ok=True)
    print(f"输出目录：{output_dir}")
    print(f"缩放倍数：{args.scale}x, 帧率：{args.fps}fps")
    
    # 查找所有 SVG 文件
    svg_files = list(renders_dir.glob("*.svg"))
    
    if not svg_files:
        print(f"在 {renders_dir} 中未找到 SVG 文件")
        return
    
    print(f"找到 {len(svg_files)} 个 SVG 文件")
    
    # 批量转换
    success_count = 0
    for svg_file in svg_files:
        output_file = output_dir / f"{svg_file.stem}.mp4"
        try:
            convert_svg_to_mp4(str(svg_file), str(output_file), fps=args.fps, scale=args.scale)
            success_count += 1
        except Exception as e:
            print(f"转换失败：{svg_file.name}")
            print(f"错误：{str(e)}")
            import traceback
            traceback.print_exc()
    
    print(f"\n转换完成：{success_count}/{len(svg_files)} 个文件成功")


if __name__ == "__main__":
    main()
