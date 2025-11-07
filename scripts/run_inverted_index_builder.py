"""
倒排索引构建器运行实例
使用配置文件运行倒排索引构建流程
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.inverted_index_builder import InvertedIndexBuilder


def main():
    """主函数"""
    print("=" * 60)
    print("倒排索引构建器运行实例")
    print("=" * 60)
    print()
    
    # 配置文件路径
    config_path = os.path.join(project_root, "config", "inverted_index_config.json")
    
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
        # 创建倒排索引构建器实例
        print("初始化倒排索引构建器...")
        builder = InvertedIndexBuilder(config_path)
        
        # 显示配置信息
        print("\n【配置信息】")
        print(f"  - 输入目录: {builder.config['input_dir']}")
        print(f"  - 输出目录: {builder.config['output_dir']}")
        print(f"  - 词项最大长度: {builder.config['storage']['term_max_length']} 字节")
        print(f"  - 文档ID位数: {builder.config['storage']['doc_id_bits']} bit")
        print(f"  - 压缩K间隔: {builder.config['compression']['k_interval']}")
        print()
        
        # 执行构建流程
        print("开始构建倒排索引...")
        builder.build()
        
        print("\n✓ 倒排索引构建完成！")
        
    except Exception as e:
        print(f"错误：倒排索引构建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
