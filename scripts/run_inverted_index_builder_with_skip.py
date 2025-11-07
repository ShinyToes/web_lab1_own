"""
带跳表指针的倒排索引构建器运行脚本
生成带跳表指针的二进制格式和压缩格式倒排表
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inverted_index_builder_with_skip import InvertedIndexBuilderWithSkip
from src.inverted_index_builder import InvertedIndexBuilder


def main():
    """主函数"""
    print("=" * 60)
    print("带跳表指针的倒排索引构建器")
    print("=" * 60)
    print()
    
    # 配置文件路径（优先使用带跳表指针的配置）
    config_path_skip = os.path.join(project_root, "config", "inverted_index_config_with_skip.json")
    config_path_base = os.path.join(project_root, "config", "inverted_index_config.json")
    
    # 优先使用带跳表指针的配置，否则使用基础配置
    if os.path.exists(config_path_skip):
        config_path = config_path_skip
        print(f"使用配置文件: {config_path_skip}")
    else:
        config_path = config_path_base
        print(f"使用基础配置文件: {config_path_base}")
    
    # 检查配置文件是否存在
    if not os.path.exists(config_path):
        print(f"错误：配置文件不存在: {config_path}")
        print("请确保 inverted_index_config.json 文件存在")
        return
    
    input_dir = os.path.join(project_root, "output", "normalized_events")
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误：输入目录不存在: {input_dir}")
        print("请先运行 part_1 的文本处理流程")
        return
    
    try:
        # 创建带跳表指针的构建器（会继承所有原始构建器的功能）
        print("开始构建带跳表指针的倒排索引...")
        builder_skip = InvertedIndexBuilderWithSkip(config_path)
        
        # 显示配置信息
        print("\n【配置信息】")
        print(f"  - 输入目录: {builder_skip.config['input_dir']}")
        print(f"  - 输出目录: {builder_skip.config.get('output_dir', '../output_inverted_index')}")
        print(f"  - 跳表指针标记（二进制）: 0x{builder_skip.SKIP_MARKER_BINARY:08X}")
        print(f"  - 跳表指针标记（压缩）: 0x{builder_skip.SKIP_MARKER_COMPRESSED:02X}")
        print(f"  - 跳表指针格式（二进制）: [标记:0xFFFFFFFF(4字节)][目标doc_id:4字节][目标file_pos:4字节]")
        print(f"  - 跳表指针策略: 间隔√L均匀放置")
        print()
        
        # 执行完整构建流程（会自动生成带跳表指针的版本）
        builder_skip.build()
        
        print("\n" + "=" * 60)
        print("带跳表指针的倒排索引构建完成！")
        print("=" * 60)
        
        # 显示输出文件信息
        output_dir = builder_skip.config["output_dir"]
        print(f"\n输出文件（均在 {output_dir} 目录）：")
        print(f"  - 倒排索引（文本）: {builder_skip.config['files']['inverted_index']}")
        print(f"  - 倒排索引（二进制，带跳表指针）: {builder_skip.config['files']['inverted_index_bin_with_skip']}")
        print(f"  - 倒排索引（压缩，带跳表指针）: {builder_skip.config['files']['inverted_index_compressed_with_skip']}")
        print(f"  - 词典（文本）: {builder_skip.config['files']['dictionary']}")
        print(f"  - 词典（二进制）: {builder_skip.config['files']['dictionary_bin']}")
        print()
        
    except Exception as e:
        print(f"错误：构建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

