"""
向量空间模型文档检索系统运行脚本
支持测试模式和交互式查询模式
"""

import os
import sys
import argparse
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.vector_space_retrieval import VectorSpaceRetrieval


def run_test_mode(retrieval_system: VectorSpaceRetrieval):
    """
    运行测试模式：执行硬编码的测试查询
    
    参数:
        retrieval_system: 检索系统实例
    """
    print("=" * 80)
    print("测试模式：执行预定义查询")
    print("=" * 80)
    
    test_queries = retrieval_system.config['test_queries']
    print(f"将执行 {len(test_queries)} 个测试查询")
    
    # 批量执行查询
    all_results = retrieval_system.batch_search(test_queries)
    
    # 显示结果
    for query in test_queries:
        results = all_results[query]
        retrieval_system.display_results(query, results)
    
    # 保存结果和生成报告
    output_dir = os.path.join(project_root, retrieval_system.config['output_dir'])
    query_results_dir = os.path.join(output_dir, "query_results")
    
    print("\n" + "=" * 80)
    print("保存查询结果")
    print("=" * 80)
    
    for i, query in enumerate(test_queries, 1):
        results = all_results[query]
        retrieval_system.save_results_to_file(query, results, query_results_dir, i)
    
    # 生成详细报告
    print("\n" + "=" * 80)
    print("生成检索报告")
    print("=" * 80)
    
    retrieval_system.generate_retrieval_report(all_results, output_dir)
    retrieval_system.save_statistics(all_results, output_dir)
    
    print(f"\n测试模式完成！所有结果已保存到: {output_dir}")


def run_interactive_mode(retrieval_system: VectorSpaceRetrieval):
    """
    运行交互式模式：用户输入查询进行检索
    
    参数:
        retrieval_system: 检索系统实例
    """
    print("=" * 80)
    print("交互式查询模式")
    print("=" * 80)
    print("输入查询文本进行文档检索，输入 'quit' 或 'exit' 退出")
    print("输入 'help' 查看帮助信息")
    print("输入 'config' 查看当前配置")
    print("输入 'methods' 查看可用的 TF-IDF 方法")
    print("输入 'stats' 查看系统统计信息")
    print()
    
    query_count = 0
    
    while True:
        try:
            query = input("请输入查询: ").strip()
            
            if not query:
                continue
            
            # 处理特殊命令
            if query.lower() in ['quit', 'exit', 'q']:
                print("退出交互式模式")
                break
            elif query.lower() == 'help':
                print_help()
                continue
            elif query.lower() == 'config':
                print_config(retrieval_system)
                continue
            elif query.lower() == 'methods':
                print_tfidf_methods(retrieval_system)
                continue
            elif query.lower() == 'stats':
                print_stats(retrieval_system)
                continue
            elif query.lower().startswith('method '):
                method = query[7:].strip()
                retrieval_system.set_tfidf_method(method)
                continue
            
            # 执行查询
            query_count += 1
            print(f"\n[查询 {query_count}] 执行查询: '{query}'")
            
            results = retrieval_system.search(query)
            retrieval_system.display_results(query, results)
            
            # 询问是否保存结果
            save_choice = input("\n是否保存此查询结果？(y/n): ").strip().lower()
            if save_choice in ['y', 'yes', '是']:
                output_dir = os.path.join(project_root, retrieval_system.config['output_dir'])
                query_results_dir = os.path.join(output_dir, "query_results")
                retrieval_system.save_results_to_file(query, results, query_results_dir, query_count)
            
            print("\n" + "-" * 80)
            
        except KeyboardInterrupt:
            print("\n\n检测到 Ctrl+C，退出交互式模式")
            break
        except Exception as e:
            print(f"查询执行出错: {e}")
            continue


def print_help():
    """打印帮助信息"""
    print("\n" + "=" * 60)
    print("帮助信息")
    print("=" * 60)
    print("可用命令:")
    print("  help     - 显示此帮助信息")
    print("  config   - 显示当前配置")
    print("  methods  - 显示可用的 TF-IDF 计算方法")
    print("  stats    - 显示系统统计信息")
    print("  method <name> - 切换 TF-IDF 计算方法")
    print("  quit/exit/q - 退出程序")
    print()
    print("查询示例:")
    print("  music concert")
    print("  food recipe cooking")
    print("  art exhibition")
    print("  technology innovation")
    print()


def print_config(retrieval_system: VectorSpaceRetrieval):
    """打印当前配置"""
    print("\n" + "=" * 60)
    print("当前配置")
    print("=" * 60)
    config = retrieval_system.get_config()
    
    print(f"TF-IDF 方法: {config['tfidf_method']}")
    print(f"Top-K 设置: {config['top_k']}")
    print(f"最小相似度阈值: {config['min_similarity']}")
    print(f"倒排索引目录: {config['inverted_index_dir']}")
    print(f"文档目录: {config['documents_dir']}")
    print(f"输出目录: {config['output_dir']}")
    print(f"测试查询数量: {len(config['test_queries'])}")
    print()


def print_tfidf_methods(retrieval_system: VectorSpaceRetrieval):
    """打印可用的 TF-IDF 方法"""
    print("\n" + "=" * 60)
    print("可用的 TF-IDF 计算方法")
    print("=" * 60)
    
    methods = retrieval_system.get_available_tfidf_methods()
    current_method = retrieval_system.config['tfidf_method']
    
    for method in methods:
        description = retrieval_system.get_tfidf_method_description(method)
        marker = " (当前)" if method == current_method else ""
        print(f"  {method:12s}: {description}{marker}")
    
    print()
    print("使用方法: method <方法名>")
    print("例如: method log")


def print_stats(retrieval_system: VectorSpaceRetrieval):
    """打印系统统计信息"""
    print("\n" + "=" * 60)
    print("系统统计信息")
    print("=" * 60)
    
    stats = retrieval_system.get_stats()
    
    print(f"总文档数: {stats['total_documents']:,}")
    print(f"词汇表大小: {stats['vocabulary_size']:,}")
    print(f"平均文档长度: {stats['avg_doc_length']:.1f} 词项")
    print(f"总词项数: {stats['total_terms']:,}")
    print()


def check_prerequisites():
    """
    检查运行前提条件
    
    返回:
        bool: 是否满足运行条件
    """
    print("检查运行前提条件...")
    
    # 检查倒排索引文件
    inverted_index_path = os.path.join(project_root, "output_inverted_index", "inverted_index.txt")
    if not os.path.exists(inverted_index_path):
        print(f"倒排索引文件不存在: {inverted_index_path}")
        print("请先运行倒排索引构建脚本: python scripts/run_inverted_index_builder.py")
        return False
    
    # 检查文档目录
    documents_dir = os.path.join(project_root, "output", "normalized_events")
    if not os.path.exists(documents_dir):
        print(f"文档目录不存在: {documents_dir}")
        print("请先运行文本处理脚本: python scripts/run_processor.py")
        return False
    
    # 检查文档文件
    doc_files = [f for f in os.listdir(documents_dir) if f.endswith('.txt')]
    if not doc_files:
        print(f"文档目录中没有找到 .txt 文件: {documents_dir}")
        return False
    
    print(f"找到 {len(doc_files)} 个文档文件")
    print("前提条件检查通过")
    return True


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='向量空间模型文档检索系统')
    parser.add_argument('--mode', choices=['test', 'interactive'], default='test',
                       help='运行模式: test (测试模式) 或 interactive (交互式模式)')
    parser.add_argument('--config', type=str, 
                       help='配置文件路径 (默认: config/retrieval_config.json)')
    parser.add_argument('--method', type=str, choices=['standard', 'log', 'augmented', 'binary'],
                       help='TF-IDF 计算方法')
    parser.add_argument('--top-k', type=int, default=10,
                       help='返回的文档数量 (默认: 10)')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("向量空间模型文档检索系统")
    print("=" * 80)
    
    # 检查前提条件
    if not check_prerequisites():
        return
    
    try:
        # 配置文件路径
        config_path = args.config
        if not config_path:
            config_path = os.path.join(project_root, "config", "retrieval_config.json")
        
        # 创建检索系统实例
        print("\n初始化检索系统...")
        retrieval_system = VectorSpaceRetrieval(config_path)
        
        # 应用命令行参数
        if args.method:
            retrieval_system.set_tfidf_method(args.method)
        
        if args.top_k != 10:
            retrieval_system.config['top_k'] = args.top_k
            print(f"Top-K 设置为: {args.top_k}")
        
        # 加载数据
        print("\n加载数据...")
        inverted_index_path = os.path.join(project_root, "output_inverted_index", "inverted_index.txt")
        documents_dir = os.path.join(project_root, "output", "normalized_events")
        
        if not retrieval_system.load_inverted_index(inverted_index_path):
            print("错误: 倒排索引加载失败")
            return
        
        if not retrieval_system.load_documents(documents_dir):
            print("错误: 文档加载失败")
            return
        
        if not retrieval_system.build_document_vectors():
            print("错误: 文档向量构建失败")
            return
        
        print("\n数据加载完成，系统准备就绪")
        
        # 根据模式运行
        if args.mode == 'test':
            run_test_mode(retrieval_system)
        else:
            run_interactive_mode(retrieval_system)
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
