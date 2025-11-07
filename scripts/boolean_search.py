"""
布尔检索效率实验
实现 AND/OR/NOT 混合查询，并比较不同处理顺序下的查询时间差异
"""

import time
import re
import struct
from typing import List, Dict, Set, Tuple


# 二进制文件格式常量
TERM_MAX_LENGTH = 20  # 词项最大长度（字节）
DOC_FREQ_SIZE = 4  # 文档频率字段大小（字节）
POINTER_SIZE = 4  # 倒排表指针字段大小（字节）
ENTRY_SIZE = TERM_MAX_LENGTH + DOC_FREQ_SIZE + POINTER_SIZE  # 每个词项总开销（28字节）
DOC_ID_SIZE = 4  # 文档ID存储大小（字节）


# ==================== 模块 1：数据准备 ====================

def load_dictionary_from_binary(dict_bin_path: str) -> Dict[str, Tuple[int, int]]:
    """
    从二进制文件加载词典
    
    参数:
        dict_bin_path: 词典二进制文件路径
    返回:
        字典，键为词项，值为(文档频率, 倒排表指针)元组
    """
    dictionary = {}
    
    with open(dict_bin_path, 'rb') as f:
        while True:
            # 读取一个词典条目（28字节）
            entry_data = f.read(ENTRY_SIZE)
            if not entry_data or len(entry_data) < ENTRY_SIZE:
                break
            
            # 解析词项（20字节）
            term_bytes = entry_data[:TERM_MAX_LENGTH]
            # 移除填充的null字节并解码
            term = term_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
            
            # 解析文档频率（4字节）
            doc_freq = struct.unpack('I', entry_data[TERM_MAX_LENGTH:TERM_MAX_LENGTH+DOC_FREQ_SIZE])[0]
            
            # 解析倒排表指针（4字节）
            pointer = struct.unpack('I', entry_data[TERM_MAX_LENGTH+DOC_FREQ_SIZE:ENTRY_SIZE])[0]
            
            dictionary[term] = (doc_freq, pointer)
    
    return dictionary


def load_posting_list_from_binary(bin_index_path: str, pointer: int, doc_freq: int) -> List[int]:
    """
    从二进制倒排索引文件中读取指定位置的posting list
    
    参数:
        bin_index_path: 二进制倒排索引文件路径
        pointer: 倒排表指针（字节偏移）
        doc_freq: 文档频率（posting list长度）
    返回:
        文档ID列表
    """
    doc_ids = []
    
    with open(bin_index_path, 'rb') as f:
        # 定位到倒排表起始位置
        f.seek(pointer)
        
        # 读取doc_freq个文档ID
        for _ in range(doc_freq):
            doc_id_bytes = f.read(DOC_ID_SIZE)
            if len(doc_id_bytes) < DOC_ID_SIZE:
                break
            doc_id = struct.unpack('I', doc_id_bytes)[0]
            doc_ids.append(doc_id)
    
    return doc_ids


def load_inverted_index_from_binary(dict_bin_path: str, bin_index_path: str) -> Dict[str, List[int]]:
    """
    从二进制文件加载完整的倒排索引
    
    参数:
        dict_bin_path: 词典二进制文件路径
        bin_index_path: 倒排索引二进制文件路径
    返回:
        字典，键为词项，值为文档ID列表
    """
    # 加载词典
    dictionary = load_dictionary_from_binary(dict_bin_path)
    
    # 根据词典中的指针加载所有倒排表
    inverted_index = {}
    for term, (doc_freq, pointer) in dictionary.items():
        doc_ids = load_posting_list_from_binary(bin_index_path, pointer, doc_freq)
        inverted_index[term] = sorted(doc_ids)  # 确保升序
    
    return inverted_index


def get_all_document_ids(inverted_index: Dict[str, List[int]]) -> Set[int]:
    """
    获取所有文档ID的集合（用于NOT操作）
    
    参数:
        inverted_index: 倒排索引字典
    返回:
        所有文档ID的集合
    """
    all_docs = set()
    for doc_list in inverted_index.values():
        all_docs.update(doc_list)
    return all_docs


# ==================== 模块 2：布尔查询操作 ====================

def intersect(list1: List[int], list2: List[int]) -> List[int]:
    """
    计算两个有序列表的交集（AND操作）
    
    参数:
        list1: 第一个文档ID列表
        list2: 第二个文档ID列表
    返回:
        交集结果列表
    """
    result = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    
    return result


def union(list1: List[int], list2: List[int]) -> List[int]:
    """
    计算两个有序列表的并集（OR操作）
    
    参数:
        list1: 第一个文档ID列表
        list2: 第二个文档ID列表
    返回:
        并集结果列表
    """
    result = []
    i, j = 0, 0
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            result.append(list1[i])
            i += 1
        else:
            result.append(list2[j])
            j += 1
    
    # 添加剩余元素
    result.extend(list1[i:])
    result.extend(list2[j:])
    
    return result


def complement(doc_list: List[int], all_docs: Set[int]) -> List[int]:
    """
    计算补集（NOT操作）
    
    参数:
        doc_list: 要取反的文档ID列表
        all_docs: 全部文档ID集合
    返回:
        补集结果列表
    """
    doc_set = set(doc_list)
    result = sorted(list(all_docs - doc_set))
    return result


# ==================== 模块 3：查询执行 ====================

def parse_and_execute_query(query: str, inverted_index: Dict[str, List[int]], 
                            all_docs: Set[int], optimize: bool = False) -> Tuple[List[int], float]:
    """
    解析并执行布尔查询
    
    参数:
        query: 查询字符串，如 "(apple AND banana) OR cherry"
        inverted_index: 倒排索引
        all_docs: 全部文档ID集合
        optimize: 是否使用优化策略（优先处理短列表）
    返回:
        (结果文档ID列表, 执行时间)
    """
    start_time = time.perf_counter()
    
    # 简化的查询解析和执行
    # 这里实现一个简单的递归下降解析器
    result = execute_query_expression(query, inverted_index, all_docs, optimize)
    
    end_time = time.perf_counter()
    execution_time = end_time - start_time
    
    return result, execution_time


def execute_query_expression(query: str, inverted_index: Dict[str, List[int]], 
                             all_docs: Set[int], optimize: bool) -> List[int]:
    """
    执行查询表达式（递归解析）
    
    参数:
        query: 查询字符串
        inverted_index: 倒排索引
        all_docs: 全部文档ID集合
        optimize: 是否优化
    返回:
        结果文档ID列表
    """
    query = query.strip()
    
    # 处理括号
    if query.startswith('(') and query.endswith(')'):
        # 找到匹配的括号
        query = query[1:-1].strip()
    
    # 处理 OR 操作（最低优先级）
    or_parts = split_by_operator(query, 'OR')
    if len(or_parts) > 1:
        results = [execute_query_expression(part, inverted_index, all_docs, optimize) 
                  for part in or_parts]
        if optimize:
            # 优化：按列表长度排序
            results.sort(key=len)
        result = results[0]
        for r in results[1:]:
            result = union(result, r)
        return result
    
    # 处理 AND 操作（中等优先级）
    and_parts = split_by_operator(query, 'AND')
    if len(and_parts) > 1:
        results = [execute_query_expression(part, inverted_index, all_docs, optimize) 
                  for part in and_parts]
        if optimize:
            # 优化：按列表长度排序，从短到长
            results.sort(key=len)
        result = results[0]
        for r in results[1:]:
            result = intersect(result, r)
        return result
    
    # 处理 NOT 操作（最高优先级）
    if query.startswith('NOT '):
        term = query[4:].strip()
        term_docs = get_term_postings(term, inverted_index, all_docs)
        return complement(term_docs, all_docs)
    
    # 基本词项
    return get_term_postings(query, inverted_index, all_docs)


def split_by_operator(query: str, operator: str) -> List[str]:
    """
    按操作符分割查询字符串（考虑括号嵌套）
    
    参数:
        query: 查询字符串
        operator: 操作符（AND 或 OR）
    返回:
        分割后的部分列表
    """
    parts = []
    current = ""
    paren_depth = 0
    i = 0
    
    while i < len(query):
        if query[i] == '(':
            paren_depth += 1
            current += query[i]
            i += 1
        elif query[i] == ')':
            paren_depth -= 1
            current += query[i]
            i += 1
        elif paren_depth == 0 and query[i:i+len(operator)+2] == f' {operator} ':
            parts.append(current.strip())
            current = ""
            i += len(operator) + 2
        else:
            current += query[i]
            i += 1
    
    if current.strip():
        parts.append(current.strip())
    
    return parts if len(parts) > 1 else [query]


def get_term_postings(term: str, inverted_index: Dict[str, List[int]], 
                      all_docs: Set[int]) -> List[int]:
    """
    获取词项的posting list，处理括号
    
    参数:
        term: 词项
        inverted_index: 倒排索引
        all_docs: 全部文档ID集合
    返回:
        文档ID列表
    """
    term = term.strip()
    if term.startswith('(') and term.endswith(')'):
        term = term[1:-1].strip()
    
    return inverted_index.get(term, [])


# ==================== 模块 4：实验设计与结果输出 ====================

def design_queries() -> List[str]:
    """
    设计测试查询（不少于3条）
    
    返回:
        查询字符串列表
    """
    queries = [
        "(ability AND able) OR accept",
        "adventure AND (age OR alert)",
        "(alone OR amazing) AND NOT activity",
        "(allow AND also) OR (alternative AND although)",
        "((area AND arrive) OR art) AND NOT already"
    ]
    return queries


def run_experiments(inverted_index: Dict[str, List[int]], all_docs: Set[int]):
    """
    运行布尔查询实验，比较不同策略的性能
    
    参数:
        inverted_index: 倒排索引
        all_docs: 全部文档ID集合
    """
    queries = design_queries()
    
    print("=" * 80)
    print("布尔检索效率实验")
    print("=" * 80)
    print()
    
    for idx, query in enumerate(queries, 1):
        print(f"查询 {idx}: {query}")
        print("-" * 80)
        
        # 从左到右执行（不优化）
        result_normal, time_normal = parse_and_execute_query(query, inverted_index, all_docs, optimize=False)
        
        # 优化策略执行（按posting list长度排序）
        result_optimized, time_optimized = parse_and_execute_query(query, inverted_index, all_docs, optimize=True)
        
        # 计算加速比
        if time_optimized > 0:
            speedup = ((time_normal - time_optimized) / time_normal) * 100
        else:
            speedup = 0
        
        # 输出结果
        print(f"从左到右执行时间:     {time_normal:.6f}s")
        print(f"优化策略执行时间:     {time_optimized:.6f}s")
        print(f"性能提升:             {speedup:+.2f}%")
        print(f"结果文档数量:         {len(result_normal)} 个")
        print(f"结果匹配:             {'✓ 一致' if result_normal == result_optimized else '✗ 不一致'}")
        print()
    
    print("=" * 80)


def save_results_to_file(inverted_index: Dict[str, List[int]], all_docs: Set[int], 
                         output_file: str):
    """
    将实验结果保存到文件
    
    参数:
        inverted_index: 倒排索引
        all_docs: 全部文档ID集合
        output_file: 输出文件路径
    """
    queries = design_queries()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("布尔检索效率实验结果\n")
        f.write("=" * 80 + "\n\n")
        
        for idx, query in enumerate(queries, 1):
            f.write(f"查询 {idx}: {query}\n")
            f.write("-" * 80 + "\n")
            
            # 执行查询
            result_normal, time_normal = parse_and_execute_query(query, inverted_index, all_docs, optimize=False)
            result_optimized, time_optimized = parse_and_execute_query(query, inverted_index, all_docs, optimize=True)
            
            # 计算加速比
            if time_optimized > 0:
                speedup = ((time_normal - time_optimized) / time_normal) * 100
            else:
                speedup = 0
            
            # 写入结果
            f.write(f"从左到右执行时间:     {time_normal:.6f}s\n")
            f.write(f"优化策略执行时间:     {time_optimized:.6f}s\n")
            f.write(f"性能提升:             {speedup:+.2f}%\n")
            f.write(f"结果文档数量:         {len(result_normal)} 个\n")
            f.write(f"结果匹配:             {'✓ 一致' if result_normal == result_optimized else '✗ 不一致'}\n")
            
            # 显示部分结果文档ID
            if result_normal:
                f.write(f"结果文档ID示例:       {result_normal[:10]}{'...' if len(result_normal) > 10 else ''}\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")


# ==================== 主程序 ====================

def main():
    """
    主函数：执行布尔检索效率实验
    """
    # 配置路径（使用绝对路径）
    import os
    base_dir = "D:/2025_2/web/Lab/web_lab1"
    dict_bin_path = f"{base_dir}/output_inverted_index/dictionary.bin"
    bin_index_path = f"{base_dir}/output_inverted_index/inverted_index.bin"
    output_dir = f"{base_dir}/output"
    output_file = f"{output_dir}/boolean_search_results.txt"
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("正在从二进制文件加载倒排索引...")
    print(f"  - 词典文件: {dict_bin_path}")
    print(f"  - 倒排索引文件: {bin_index_path}")
    
    inverted_index = load_inverted_index_from_binary(dict_bin_path, bin_index_path)
    print(f"加载完成！共 {len(inverted_index)} 个词项\n")
    
    # 获取所有文档ID
    all_docs = get_all_document_ids(inverted_index)
    print(f"文档总数: {len(all_docs)}\n")
    
    # 运行实验
    run_experiments(inverted_index, all_docs)
    
    # 保存结果到文件
    save_results_to_file(inverted_index, all_docs, output_file)
    print(f"实验结果已保存到: {output_file}")


if __name__ == "__main__":
    main()

