#!/usr/bin/env python3
"""
事件文本处理器实例运行文件
使用 EventTextProcessor 类处理事件文本
"""

import os
import sys
import json
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.event_text_processor import EventTextProcessor


def main():
    """
    主函数 - 运行事件文本处理流水线
    """
    print("=" * 80)
    print("事件文本处理器 v2.0")
    print("=" * 80)
    print()
    
    # 检查配置文件（使用项目根目录的绝对路径）
    config_file = project_root / "config" / "config_process.json"
    if not config_file.exists():
        print(f"警告: 配置文件 {config_file} 不存在，将使用默认配置")
        config_file = None
    else:
        print(f"使用配置文件: {config_file}")
        config_file = str(config_file)
    
    # 创建处理器实例
    try:
        processor = EventTextProcessor(config_file)
        print("✓ 处理器初始化成功")
    except Exception as e:
        print(f"✗ 处理器初始化失败: {e}")
        return 1
    
    # 显示当前配置
    print("\n当前配置:")
    config = processor.get_config()
    print(f"  - 输入目录: {config['input']['xml_dir']}")
    print(f"  - 最大处理文件数: {config['input']['max_files']}")
    print(f"  - 输出基础目录: {config['output']['base_dir']}")
    print(f"  - 使用多词短语识别: {config['tokenization']['use_mwe']}")
    print(f"  - 移除停用词: {config['normalization']['remove_stopwords']}")
    print(f"  - 词形还原: {config['normalization']['use_lemmatization']}")
    
    # 显示多进程配置
    processing_config = config.get('processing', {})
    print(f"  - 使用多进程并行处理: {processing_config.get('use_multithread', True)}")
    print(f"  - 最大工作进程数: {processing_config.get('max_workers', 'auto')}")
    
    # 确认运行
    print()
    confirm = input("确认开始处理? (y/n，直接回车确认): ").strip().lower()
    if confirm and confirm != 'y':
        print("已取消")
        return 0
    
    # 运行流水线
    try:
        result_stats = processor.run_pipeline()
        
        if result_stats:
            print("\n处理结果:")
            print(f"  - 成功处理事件数: {result_stats['success_count']}")
            print(f"  - 总文件数: {result_stats['total_files']}")
            print(f"  - 执行时间: {result_stats['elapsed_time']:.2f} 秒")
            print(f"  - Token 统计: {result_stats['token_stats']['original_tokens']} → {result_stats['token_stats']['final_tokens']}")
            
            print("\n输出目录:")
            for name, path in result_stats['output_dirs'].items():
                print(f"  - {name}: {path}")
            
            return 0
        else:
            print("✗ 处理失败")
            return 1
            
    except Exception as e:
        print(f"✗ 处理过程中发生错误: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
