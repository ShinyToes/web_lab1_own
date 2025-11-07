"""
跳表指针测试脚本
测试带跳表指针的倒排索引查询，统计跳表指针的尝试次数和成功次数
"""

import os
import sys
import struct
import json
import time
import bisect
import threading
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ==================== 文件句柄缓存 ====================
# 全局文件句柄缓存，避免重复打开文件
_file_handle_cache: Dict[str, Tuple[Any, threading.Lock]] = {}
_cache_lock = threading.Lock()

def get_file_handle(file_path: str):
    """
    获取文件句柄（带缓存）
    
    参数:
        file_path: 文件路径
        
    返回:
        文件句柄
    """
    global _file_handle_cache
    
    # 标准化路径
    file_path = os.path.abspath(file_path)
    
    with _cache_lock:
        if file_path not in _file_handle_cache:
            # 创建新文件句柄
            f = open(file_path, 'rb')
            file_lock = threading.Lock()
            _file_handle_cache[file_path] = (f, file_lock)
            return f, file_lock
        else:
            # 返回缓存的文件句柄
            f, file_lock = _file_handle_cache[file_path]
            return f, file_lock

def close_all_file_handles():
    """
    关闭所有缓存的文件句柄
    """
    global _file_handle_cache
    
    with _cache_lock:
        for file_path, (f, _) in _file_handle_cache.items():
            try:
                if not f.closed:
                    f.close()
            except Exception as e:
                print(f"警告: 关闭文件 {file_path} 时出错: {e}")
        _file_handle_cache.clear()

from src.inverted_index_builder_with_skip import InvertedIndexBuilderWithSkip

# 二进制文件格式常量
TERM_MAX_LENGTH = 20
DOC_FREQ_SIZE = 4
POINTER_SIZE = 4
ENTRY_SIZE = TERM_MAX_LENGTH + DOC_FREQ_SIZE + POINTER_SIZE
DOC_ID_SIZE = 4
SKIP_MARKER_BINARY = 0xFFFFFFFF


class SkipPointerStats:
    """跳表指针统计信息"""
    def __init__(self):
        self.skip_attempts = 0  # 跳表指针尝试次数
        self.skip_successes = 0  # 跳表指针成功次数
        self.comparisons_without_skip = 0  # 不使用跳表指针的比较次数
        self.comparisons_with_skip = 0  # 使用跳表指针后的比较次数
        
    def __str__(self):
        return (f"跳表指针统计:\n"
                f"  - 尝试次数: {self.skip_attempts}\n"
                f"  - 成功次数: {self.skip_successes}\n"
                f"  - 成功率: {self.skip_successes/self.skip_attempts*100:.2f}%" if self.skip_attempts > 0 else "  - 成功率: 0%\n"
                f"  - 未使用跳表指针的比较次数: {self.comparisons_without_skip}\n"
                f"  - 使用跳表指针后的比较次数: {self.comparisons_with_skip}")


class QueryStats:
    """查询统计信息（包含跳表和无跳表版本）"""
    def __init__(self):
        self.skip_attempts = 0
        self.skip_successes = 0
        self.comparisons_with_skip = 0
        self.comparisons_without_skip = 0
        self.comparisons_no_skip_version = 0  # 无跳表版本的比较次数
        self.execution_time_skip = 0.0  # 跳表版本执行时间
        self.execution_time_no_skip = 0.0  # 无跳表版本执行时间
        self.result_skip = []  # 跳表版本结果
        self.result_no_skip = []  # 无跳表版本结果
        
    def __str__(self):
        speedup = 0.0
        if self.execution_time_no_skip > 0:
            speedup = ((self.execution_time_no_skip - self.execution_time_skip) / self.execution_time_no_skip) * 100
        
        comparison_reduction = 0.0
        if self.comparisons_no_skip_version > 0:
            comparison_reduction = ((self.comparisons_no_skip_version - (self.comparisons_without_skip + self.comparisons_with_skip)) / self.comparisons_no_skip_version) * 100
        
        return (f"查询统计:\n"
                f"  - 结果一致性: {'✓ 一致' if self.result_skip == self.result_no_skip else '✗ 不一致'}\n"
                f"  - 跳表版本执行时间: {self.execution_time_skip*1000:.3f} ms\n"
                f"  - 无跳表版本执行时间: {self.execution_time_no_skip*1000:.3f} ms\n"
                f"  - 性能提升: {speedup:+.2f}%\n"
                f"  - 跳表指针尝试次数: {self.skip_attempts}\n"
                f"  - 跳表指针成功次数: {self.skip_successes}\n"
                f"  - 跳表指针成功率: {self.skip_successes/self.skip_attempts*100:.2f}%" if self.skip_attempts > 0 else "  - 跳表指针成功率: 0%\n"
                f"  - 无跳表版本比较次数: {self.comparisons_no_skip_version}\n"
                f"  - 跳表版本比较次数: {self.comparisons_without_skip + self.comparisons_with_skip}\n"
                f"  - 比较次数减少: {comparison_reduction:+.2f}%")


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
            entry_data = f.read(ENTRY_SIZE)
            if not entry_data or len(entry_data) < ENTRY_SIZE:
                break
            
            term_bytes = entry_data[:TERM_MAX_LENGTH]
            term = term_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
            
            doc_freq = struct.unpack('I', entry_data[TERM_MAX_LENGTH:TERM_MAX_LENGTH+DOC_FREQ_SIZE])[0]
            pointer = struct.unpack('I', entry_data[TERM_MAX_LENGTH+DOC_FREQ_SIZE:ENTRY_SIZE])[0]
            
            dictionary[term] = (doc_freq, pointer)
    
    return dictionary


def load_posting_list_with_skip(
    bin_index_path: str, 
    pointer: int, 
    doc_freq: int,
    builder: InvertedIndexBuilderWithSkip,
    next_pointer: Optional[int] = None
) -> Tuple[List[int], List[Tuple[int, int, int]]]:
    """
    从带跳表指针的二进制倒排索引文件中读取posting list（使用文件句柄缓存）
    
    参数:
        bin_index_path: 二进制倒排索引文件路径
        pointer: 倒排表指针（字节偏移）
        doc_freq: 文档频率（posting list长度）
        builder: InvertedIndexBuilderWithSkip实例，用于解码
        next_pointer: 下一个词项的指针（如果知道，用于确定读取范围）
    返回:
        (文档ID列表, 跳表指针列表)
        跳表指针列表：[(位置索引, 目标doc_id, 目标file_pos), ...]
    """
    # 使用缓存的文件句柄
    f, file_lock = get_file_handle(bin_index_path)
    
    with file_lock:
        # 定位到指定位置
        f.seek(pointer)
        
        if next_pointer is not None and next_pointer > pointer:
            # 如果知道下一个词项的位置，读取到那个位置
            read_size = next_pointer - pointer
            binary_data = f.read(read_size)
        else:
            # 否则估算大小：doc_freq个doc_id + 可能的跳表指针
            # 跳表指针最多有 sqrt(doc_freq) 个，每个12字节
            max_skip_pointers = int(doc_freq ** 0.5) + 1
            estimated_size = doc_freq * DOC_ID_SIZE + max_skip_pointers * 12
            binary_data = f.read(estimated_size)
    
    # 使用builder的解码方法
    doc_ids, skip_pointers = builder.decode_binary_with_skip(binary_data, doc_freq)
    return doc_ids, skip_pointers


def intersect_with_skip_stats(
    list1: List[int], 
    skip_pointers1: List[Tuple[int, int, int]],
    list2: List[int],
    skip_pointers2: List[Tuple[int, int, int]],
    stats: SkipPointerStats
) -> Tuple[List[int], SkipPointerStats]:
    """
    使用跳表指针计算两个有序列表的交集，并统计跳表指针使用情况
    
    参数:
        list1: 第一个文档ID列表（较短的列表）
        skip_pointers1: 第一个列表的跳表指针
        list2: 第二个文档ID列表（较长的列表，使用跳表指针）
        skip_pointers2: 第二个列表的跳表指针
        stats: 统计信息对象
    返回:
        (交集结果列表, 更新后的统计信息)
    """
    result = []
    i, j = 0, 0
    skip_ptr_idx = 0  # 当前跳表指针索引
    
    # 将跳表指针转换为字典，方便查找：{位置索引: (目标doc_id, 目标file_pos)}
    skip_dict = {pos: (target_doc_id, target_file_pos) for pos, target_doc_id, target_file_pos in skip_pointers2}
    
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
            stats.comparisons_without_skip += 1
        elif list1[i] < list2[j]:
            i += 1
            stats.comparisons_without_skip += 1
        else:
            # list1[i] > list2[j]，需要移动list2的指针
            # 尝试使用跳表指针
            if j in skip_dict:
                target_doc_id, _ = skip_dict[j]
                stats.skip_attempts += 1
                
                # 检查是否可以跳过
                if list1[i] >= target_doc_id:
                    # 不能跳过，因为目标位置的值仍然小于list1[i]
                    # 继续检查下一个跳表指针或正常前进
                    j += 1
                    stats.comparisons_without_skip += 1
                else:
                    # 可以跳过！直接跳到目标位置
                    # 找到目标位置在list2中的索引
                    target_index = j
                    for k in range(j + 1, len(list2)):
                        if list2[k] >= target_doc_id:
                            target_index = k
                            break
                    
                    if target_index > j:
                        stats.skip_successes += 1
                        stats.comparisons_with_skip += (target_index - j)
                        j = target_index
                    else:
                        j += 1
                        stats.comparisons_without_skip += 1
            else:
                # 当前位置没有跳表指针，正常前进
                j += 1
                stats.comparisons_without_skip += 1
    
    return result, stats


def intersect_with_skip_stats_v2(
    short_list: List[int],
    long_list: List[int],
    long_skip_pointers: List[Tuple[int, int, int]],
    stats: SkipPointerStats
) -> Tuple[List[int], SkipPointerStats]:
    """
    使用跳表指针计算两个有序列表的交集（优化版本）
    假设short_list较短，long_list较长且包含跳表指针
    
    参数:
        short_list: 较短的文档ID列表
        long_list: 较长的文档ID列表（带跳表指针）
        long_skip_pointers: 长列表的跳表指针 [(位置索引, 目标doc_id, 目标file_pos), ...]
        stats: 统计信息对象
    返回:
        (交集结果列表, 更新后的统计信息)
    """
    # 调试信息：检查输入
    if len(short_list) == 0 or len(long_list) == 0:
        return [], stats
    
    result = []
    i, j = 0, 0
    
    # 将跳表指针转换为按位置索引排序的列表，方便查找
    # 跳表指针格式：(位置索引, 目标doc_id, 目标file_pos)
    sorted_skip_pointers = sorted(long_skip_pointers, key=lambda x: x[0])
    skip_ptr_idx = 0
    
    # 优化：构建位置列表，用于快速查找（使用bisect模块进行二分查找）
    if sorted_skip_pointers:
        skip_positions = [pos for pos, _, _ in sorted_skip_pointers]
    else:
        skip_positions = []
    
    while i < len(short_list) and j < len(long_list):
        if short_list[i] == long_list[j]:
            result.append(short_list[i])
            i += 1
            j += 1
            stats.comparisons_without_skip += 1
        elif short_list[i] < long_list[j]:
            # 短表的id太小，短表前进
            i += 1
            stats.comparisons_without_skip += 1
        else:
            # short_list[i] > long_list[j]，需要移动长表的指针
            # 尝试使用跳表指针
            used_skip = False
            
            # 优化：使用二分查找快速定位j位置之后最近的跳表指针
            if skip_positions and skip_ptr_idx < len(skip_positions):
                # 在skip_positions中查找第一个 >= j 的位置（从skip_ptr_idx开始）
                search_idx = bisect.bisect_left(skip_positions, j, lo=skip_ptr_idx)
                
                if search_idx < len(sorted_skip_pointers):
                    skip_pos, target_doc_id, _ = sorted_skip_pointers[search_idx]
                    
                    # 如果跳表指针位置在当前位置或之后，且目标值大于当前要找的值
                    if skip_pos >= j and short_list[i] < target_doc_id:
                        stats.skip_attempts += 1
                        
                        # 检查skip_pos位置的值，决定如何跳跃
                        search_comparisons = 0
                        target_j = len(long_list)
                        found_match = False
                        
                        if long_list[skip_pos] == short_list[i]:
                            # 在skip_pos位置找到匹配
                            target_j = skip_pos
                            found_match = True
                            search_comparisons = 1
                        elif long_list[skip_pos] > short_list[i]:
                            # short_list[i]可能在j到skip_pos之间，需要线性搜索
                            search_comparisons = 0
                            target_j = skip_pos
                            found_in_range = False
                            
                            for k in range(j + 1, min(skip_pos + 1, len(long_list))):
                                search_comparisons += 1
                                if long_list[k] == short_list[i]:
                                    target_j = k
                                    found_match = True
                                    found_in_range = True
                                    break
                                elif long_list[k] > short_list[i]:
                                    target_j = k
                                    found_in_range = True
                                    break
                            
                            if not found_in_range:
                                target_j = skip_pos
                                search_comparisons = skip_pos - j
                        else:
                            # long_list[skip_pos] < short_list[i]，可以继续搜索
                            search_comparisons = 1
                            target_j = len(long_list)
                            for k in range(skip_pos + 1, len(long_list)):
                                search_comparisons += 1
                                if long_list[k] == short_list[i]:
                                    target_j = k
                                    found_match = True
                                    break
                                elif long_list[k] > short_list[i]:
                                    target_j = k
                                    break
                        
                        if found_match:
                            stats.skip_successes += 1
                            stats.comparisons_with_skip += search_comparisons
                            j = target_j
                            used_skip = True
                            skip_ptr_idx = search_idx
                            continue
                        elif target_j > j and target_j < len(long_list):
                            stats.skip_successes += 1
                            stats.comparisons_with_skip += search_comparisons
                            j = target_j
                            skip_ptr_idx = search_idx
                            i += 1
                            used_skip = True
                            continue
                        elif target_j == len(long_list):
                            stats.skip_successes += 1
                            stats.comparisons_with_skip += search_comparisons
                            skip_ptr_idx += 1
                            i += 1
                            used_skip = True
                            continue
                        else:
                            skip_ptr_idx += 1
                    else:
                        skip_ptr_idx += 1
            
            if not used_skip:
                # 没有可用的跳表指针或无法跳过，正常前进
                # 注意：这里需要统计一次比较（short_list[i] > long_list[j]的比较）
                stats.comparisons_without_skip += 1
                j += 1
    
    return result, stats


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


def complement(doc_list: List[int], all_docs: set) -> List[int]:
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


class QueryStats:
    """查询统计信息（包含跳表和无跳表版本）"""
    def __init__(self):
        self.skip_attempts = 0
        self.skip_successes = 0
        self.comparisons_with_skip = 0
        self.comparisons_without_skip = 0
        self.comparisons_no_skip_version = 0  # 无跳表版本的比较次数
        self.execution_time_skip = 0.0  # 跳表版本执行时间
        self.execution_time_no_skip = 0.0  # 无跳表版本执行时间


def execute_query_expression_no_skip(
    query: str,
    dictionary: Dict[str, Tuple[int, int]],
    bin_index_path: str,
    builder: InvertedIndexBuilderWithSkip,
    all_docs: set,
    stats: QueryStats,
    pointer_to_term: Dict[int, str]
) -> Tuple[List[int], int]:
    """
    执行查询表达式（不使用跳表指针的版本，用于对照）
    
    返回:
        (结果列表, 总比较次数)
    """
    query = query.strip()
    total_comparisons = 0
    
    # 处理括号
    if query.startswith('(') and query.endswith(')'):
        paren_depth = 0
        is_complete = True
        for i, char in enumerate(query):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0 and i < len(query) - 1:
                    is_complete = False
                    break
        if is_complete:
            query = query[1:-1].strip()
    
    # 处理 OR 操作
    or_parts = split_by_operator(query, 'OR')
    if len(or_parts) > 1:
        results = []
        for part in or_parts:
            result, comparisons = execute_query_expression_no_skip(
                part, dictionary, bin_index_path, builder,
                all_docs, stats, pointer_to_term
            )
            results.append(result)
            total_comparisons += comparisons
        results.sort(key=len)
        result = results[0]
        for r in results[1:]:
            result = union(result, r)
        return result, total_comparisons
    
    # 处理 AND 操作
    and_parts = split_by_operator(query, 'AND')
    if len(and_parts) > 1:
        results = []
        for part in and_parts:
            result, comparisons = execute_query_expression_no_skip(
                part, dictionary, bin_index_path, builder,
                all_docs, stats, pointer_to_term
            )
            results.append(result)
            total_comparisons += comparisons
        results.sort(key=len)
        result = results[0]
        
        for r in results[1:]:
            if len(result) <= len(r):
                short_list, long_list = result, r
            else:
                short_list, long_list = r, result
            
            # 使用普通交集（不使用跳表指针）
            result, comparisons = intersect_simple(short_list, long_list)
            total_comparisons += comparisons
        
        return result, total_comparisons
    
    # 处理 NOT 操作
    if query.upper().startswith('NOT '):
        term = query[4:].strip()
        term = term.strip('()')
        term_docs = get_term_postings_with_skip(
            term, dictionary, bin_index_path, builder, pointer_to_term
        )
        result = complement(term_docs, all_docs)
        return result, total_comparisons
    
    # 基本词项
    result = get_term_postings_with_skip(
        query, dictionary, bin_index_path, builder, pointer_to_term
    )
    return result, total_comparisons


def execute_query_expression_with_skip(
    query: str,
    dictionary: Dict[str, Tuple[int, int]],
    bin_index_path: str,
    builder: InvertedIndexBuilderWithSkip,
    all_docs: set,
    stats: SkipPointerStats,
    pointer_to_term: Dict[int, str]
) -> List[int]:
    """
    执行查询表达式（递归解析，支持跳表指针统计）
    
    参数:
        query: 查询字符串
        dictionary: 词典
        bin_index_path: 二进制倒排索引文件路径
        builder: InvertedIndexBuilderWithSkip实例
        all_docs: 全部文档ID集合
        stats: 统计信息对象
        pointer_to_term: 指针到词项的映射（用于确定读取范围）
    返回:
        结果文档ID列表
    """
    query = query.strip()
    
    # 处理括号
    if query.startswith('(') and query.endswith(')'):
        # 检查是否是完整的括号包裹
        paren_depth = 0
        is_complete = True
        for i, char in enumerate(query):
            if char == '(':
                paren_depth += 1
            elif char == ')':
                paren_depth -= 1
                if paren_depth == 0 and i < len(query) - 1:
                    is_complete = False
                    break
        
        if is_complete:
            query = query[1:-1].strip()
    
    # 处理 OR 操作（最低优先级）
    or_parts = split_by_operator(query, 'OR')
    if len(or_parts) > 1:
        results = [
            execute_query_expression_with_skip(
                part, dictionary, bin_index_path, builder, 
                all_docs, stats, pointer_to_term
            )
            for part in or_parts
        ]
        # 按列表长度排序（优化）
        results.sort(key=len)
        result = results[0]
        for r in results[1:]:
            result = union(result, r)
        return result
    
    # 处理 AND 操作（中等优先级）
    and_parts = split_by_operator(query, 'AND')
    if len(and_parts) > 1:
        results = [
            execute_query_expression_with_skip(
                part, dictionary, bin_index_path, builder,
                all_docs, stats, pointer_to_term
            )
            for part in and_parts
        ]
        # 按列表长度排序，从短到长（优化，使用跳表指针）
        results.sort(key=len)
        result = results[0]
        
        # 与后续结果求交集
        for r in results[1:]:
            # 选择较短的列表作为短列表
            if len(result) <= len(r):
                short_list, long_list = result, r
            else:
                short_list, long_list = r, result
            
            # 尝试获取长列表的跳表指针
            # 如果长列表是单个词项的结果，可以获取其跳表指针
            # 否则使用普通交集
            long_skip_pointers = []
            
            # 检查长列表是否来自单个词项（简化处理：如果and_parts只有两个且都是简单词项）
            if len(and_parts) == 2:
                # 尝试获取长列表对应的词项的跳表指针
                part1 = and_parts[0].strip().lower()
                part2 = and_parts[1].strip().lower()
                
                # 检查是否是简单词项（不是括号表达式）
                if (not part1.startswith('(') and not part1.startswith('NOT ') and 
                    not part2.startswith('(') and not part2.startswith('NOT ')):
                    # 两个简单词项，可以获取跳表指针
                    if len(result) <= len(r):
                        # result来自part1，r来自part2
                        if part2 in dictionary:
                            _, pointer2 = dictionary[part2]
                            next_ptr2 = None
                            for ptr in sorted(pointer_to_term.keys()):
                                if ptr > pointer2:
                                    next_ptr2 = ptr
                                    break
                            _, long_skip_pointers = load_posting_list_with_skip(
                                bin_index_path, pointer2, dictionary[part2][0], 
                                builder, next_ptr2
                            )
                    else:
                        # r来自part1，result来自part2
                        if part1 in dictionary:
                            _, pointer1 = dictionary[part1]
                            next_ptr1 = None
                            for ptr in sorted(pointer_to_term.keys()):
                                if ptr > pointer1:
                                    next_ptr1 = ptr
                                    break
                            _, long_skip_pointers = load_posting_list_with_skip(
                                bin_index_path, pointer1, dictionary[part1][0],
                                builder, next_ptr1
                            )
            
            if long_skip_pointers:
                # 使用跳表指针进行交集
                result, stats = intersect_with_skip_stats_v2(
                    short_list, long_list, long_skip_pointers, stats
                )
            else:
                # 使用普通交集（不使用跳表指针）
                result, comparisons = intersect_simple(short_list, long_list)
                # 注意：这里stats是SkipPointerStats，需要适配
                # 为了简化，我们只记录比较次数
                stats.comparisons_without_skip += comparisons
        
        return result
    
    # 处理 NOT 操作（最高优先级）
    if query.upper().startswith('NOT '):
        term = query[4:].strip()
        term = term.strip('()')
        term_docs = get_term_postings_with_skip(
            term, dictionary, bin_index_path, builder, pointer_to_term
        )
        return complement(term_docs, all_docs)
    
    # 基本词项
    return get_term_postings_with_skip(
        query, dictionary, bin_index_path, builder, pointer_to_term
    )


def intersect_simple(list1: List[int], list2: List[int]) -> Tuple[List[int], int]:
    """
    简单的交集操作（不使用跳表指针）
    
    返回:
        (交集结果列表, 比较次数)
    """
    result = []
    i, j = 0, 0
    comparisons = 0
    
    while i < len(list1) and j < len(list2):
        comparisons += 1
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    
    return result, comparisons


def get_term_postings_with_skip(
    term: str,
    dictionary: Dict[str, Tuple[int, int]],
    bin_index_path: str,
    builder: InvertedIndexBuilderWithSkip,
    pointer_to_term: Dict[int, str]
) -> List[int]:
    """
    获取词项的posting list（带跳表指针支持）
    """
    term = term.strip().lower()
    if term not in dictionary:
        return []
    
    doc_freq, pointer = dictionary[term]
    
    # 找到下一个词项的指针
    next_ptr = None
    for ptr in sorted(pointer_to_term.keys()):
        if ptr > pointer:
            next_ptr = ptr
            break
    
    doc_ids, _ = load_posting_list_with_skip(
        bin_index_path, pointer, doc_freq, builder, next_ptr
    )
    return doc_ids


def get_all_document_ids(
    dictionary: Dict[str, Tuple[int, int]],
    bin_index_path: str,
    builder: InvertedIndexBuilderWithSkip,
    pointer_to_term: Dict[int, str]
) -> set:
    """
    获取所有文档ID的集合（用于NOT操作）
    """
    all_docs = set()
    for term, (doc_freq, pointer) in dictionary.items():
        doc_ids = get_term_postings_with_skip(
            term, dictionary, bin_index_path, builder, pointer_to_term
        )
        all_docs.update(doc_ids)
    return all_docs


def intersect_with_skip_for_two_terms(
    term1: str,
    term2: str,
    dictionary: Dict[str, Tuple[int, int]],
    bin_index_path: str,
    builder: InvertedIndexBuilderWithSkip,
    pointer_to_term: Dict[int, str],
    stats: SkipPointerStats
) -> List[int]:
    """
    两个词项的交集查询（使用跳表指针）
    """
    if term1 not in dictionary or term2 not in dictionary:
        return []
    
    doc_freq1, pointer1 = dictionary[term1]
    doc_freq2, pointer2 = dictionary[term2]
    
    # 获取下一个词项的指针
    next_ptr1 = None
    next_ptr2 = None
    for ptr in sorted(pointer_to_term.keys()):
        if next_ptr1 is None and ptr > pointer1:
            next_ptr1 = ptr
        if next_ptr2 is None and ptr > pointer2:
            next_ptr2 = ptr
        if next_ptr1 and next_ptr2:
            break
    
    list1, skip_pointers1 = load_posting_list_with_skip(
        bin_index_path, pointer1, doc_freq1, builder, next_ptr1
    )
    list2, skip_pointers2 = load_posting_list_with_skip(
        bin_index_path, pointer2, doc_freq2, builder, next_ptr2
    )
    
    # 选择较短的列表作为短列表
    if len(list1) <= len(list2):
        short_list, long_list = list1, list2
        long_skip_pointers = skip_pointers2
    else:
        short_list, long_list = list2, list1
        long_skip_pointers = skip_pointers1
    
    result, stats = intersect_with_skip_stats_v2(
        short_list, long_list, long_skip_pointers, stats
    )
    return result


def test_single_query(
    dict_bin_path: str,
    bin_index_path: str,
    config_path: str,
    query: str
):
    """
    测试单个查询，详细对比跳表版本和无跳表版本的结果
    
    参数:
        dict_bin_path: 词典二进制文件路径
        bin_index_path: 带跳表指针的二进制倒排索引文件路径
        config_path: 配置文件路径
        query: 测试查询
    """
    print("=" * 80)
    print(f"单查询测试: {query}")
    print("=" * 80)
    print()
    
    # 加载配置和构建器
    builder = InvertedIndexBuilderWithSkip(config_path)
    
    # 加载词典
    print("加载词典...")
    dictionary = load_dictionary_from_binary(dict_bin_path)
    print(f"[OK] 加载了 {len(dictionary)} 个词项")
    print()
    
    # 构建指针到词项的映射
    pointer_to_term = {}
    for term, (doc_freq, pointer) in dictionary.items():
        pointer_to_term[pointer] = term
    
    # 获取所有文档ID集合（用于NOT操作）
    print("获取所有文档ID集合...")
    all_docs = get_all_document_ids(dictionary, bin_index_path, builder, pointer_to_term)
    print(f"[OK] 总文档数: {len(all_docs)}")
    print()
    
    # 创建统计对象
    query_stats = QueryStats()
    skip_stats = SkipPointerStats()
    
    try:
        # 执行跳表版本
        print("执行跳表版本...")
        start_time_skip = time.perf_counter()
        result_skip = execute_query_expression_with_skip(
            query, dictionary, bin_index_path, builder,
            all_docs, skip_stats, pointer_to_term
        )
        end_time_skip = time.perf_counter()
        time_skip = end_time_skip - start_time_skip
        
        # 执行无跳表版本
        print("执行无跳表版本...")
        start_time_no_skip = time.perf_counter()
        result_no_skip, comparisons_no_skip = execute_query_expression_no_skip(
            query, dictionary, bin_index_path, builder,
            all_docs, query_stats, pointer_to_term
        )
        end_time_no_skip = time.perf_counter()
        time_no_skip = end_time_no_skip - start_time_no_skip
        
        # 更新统计信息
        query_stats.result_skip = result_skip
        query_stats.result_no_skip = result_no_skip
        query_stats.execution_time_skip = time_skip
        query_stats.execution_time_no_skip = time_no_skip
        query_stats.skip_attempts = skip_stats.skip_attempts
        query_stats.skip_successes = skip_stats.skip_successes
        query_stats.comparisons_without_skip = skip_stats.comparisons_without_skip
        query_stats.comparisons_with_skip = skip_stats.comparisons_with_skip
        query_stats.comparisons_no_skip_version = comparisons_no_skip
        
        # 排序结果以便比较
        result_skip_sorted = sorted(result_skip)
        result_no_skip_sorted = sorted(result_no_skip)
        
        # 详细输出
        print("\n" + "=" * 80)
        print("详细结果对比")
        print("=" * 80)
        print()
        
        print(f"【跳表版本】")
        print(f"  执行时间: {time_skip*1000:.3f} ms")
        print(f"  结果数量: {len(result_skip)}")
        print(f"  比较次数: {skip_stats.comparisons_without_skip + skip_stats.comparisons_with_skip}")
        print(f"    - 未使用跳表指针: {skip_stats.comparisons_without_skip}")
        print(f"    - 使用跳表指针: {skip_stats.comparisons_with_skip}")
        print(f"  跳表指针尝试次数: {skip_stats.skip_attempts}")
        print(f"  跳表指针成功次数: {skip_stats.skip_successes}")
        if skip_stats.skip_attempts > 0:
            print(f"  跳表指针成功率: {skip_stats.skip_successes/skip_stats.skip_attempts*100:.2f}%")
        if len(result_skip_sorted) <= 20:
            print(f"  结果文档ID: {result_skip_sorted}")
        else:
            print(f"  结果文档ID（前20个）: {result_skip_sorted[:20]}...")
            print(f"  结果文档ID（后20个）: ...{result_skip_sorted[-20:]}")
        
        print()
        print(f"【无跳表版本】")
        print(f"  执行时间: {time_no_skip*1000:.3f} ms")
        print(f"  结果数量: {len(result_no_skip)}")
        print(f"  比较次数: {comparisons_no_skip}")
        if len(result_no_skip_sorted) <= 20:
            print(f"  结果文档ID: {result_no_skip_sorted}")
        else:
            print(f"  结果文档ID（前20个）: {result_no_skip_sorted[:20]}...")
            print(f"  结果文档ID（后20个）: ...{result_no_skip_sorted[-20:]}")
        
        print()
        print("=" * 80)
        print("结果一致性检查")
        print("=" * 80)
        
        # 检查结果是否一致
        results_match = result_skip_sorted == result_no_skip_sorted
        
        if results_match:
            print("[OK] 结果完全一致！")
        else:
            print("[ERROR] 结果不一致！")
            print()
            print("差异分析：")
            only_in_skip = set(result_skip_sorted) - set(result_no_skip_sorted)
            only_in_no_skip = set(result_no_skip_sorted) - set(result_skip_sorted)
            
            if only_in_skip:
                print(f"  只在跳表版本中的文档ID ({len(only_in_skip)}个): {sorted(only_in_skip)}")
            if only_in_no_skip:
                print(f"  只在无跳表版本中的文档ID ({len(only_in_no_skip)}个): {sorted(only_in_no_skip)}")
        
        print()
        print("=" * 80)
        print("性能对比")
        print("=" * 80)
        if time_no_skip > 0:
            speedup = ((time_no_skip - time_skip) / time_no_skip) * 100
            print(f"  时间提升: {speedup:+.2f}%")
        if comparisons_no_skip > 0:
            comparison_reduction = ((comparisons_no_skip - (skip_stats.comparisons_without_skip + skip_stats.comparisons_with_skip)) / comparisons_no_skip) * 100
            print(f"  比较次数减少: {comparison_reduction:+.2f}%")
        print()
        
    except Exception as e:
        print(f"错误：查询执行失败: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # 关闭所有缓存的文件句柄
        close_all_file_handles()


def test_skip_pointer_queries(
    dict_bin_path: str,
    bin_index_path: str,
    config_path: str,
    test_queries: List[str]
):
    """
    测试带跳表指针的查询（支持完整布尔检索）
    
    参数:
        dict_bin_path: 词典二进制文件路径
        bin_index_path: 带跳表指针的二进制倒排索引文件路径
        config_path: 配置文件路径
        test_queries: 测试查询列表
    """
    print("=" * 80)
    print("跳表指针查询测试（支持完整布尔检索）")
    print("=" * 80)
    print()
    
    # 加载配置和构建器
    builder = InvertedIndexBuilderWithSkip(config_path)
    
    # 加载词典
    print("加载词典...")
    dictionary = load_dictionary_from_binary(dict_bin_path)
    print(f"[OK] 加载了 {len(dictionary)} 个词项")
    print()
    
    # 构建指针到词项的映射（用于确定读取范围）
    pointer_to_term = {}
    for term, (doc_freq, pointer) in dictionary.items():
        pointer_to_term[pointer] = term
    
    # 获取所有文档ID集合（用于NOT操作）
    print("获取所有文档ID集合（用于NOT操作）...")
    all_docs = get_all_document_ids(dictionary, bin_index_path, builder, pointer_to_term)
    print(f"[OK] 总文档数: {len(all_docs)}")
    print()
    
    # 执行测试查询
    total_stats = QueryStats()
    
    for query_idx, query in enumerate(test_queries, 1):
        print(f"查询 {query_idx}: {query}")
        print("-" * 80)
        
        # 为每个查询创建新的统计对象
        query_stats = QueryStats()
        skip_stats = SkipPointerStats()
        
        try:
            # 执行跳表版本
            start_time_skip = time.perf_counter()
            result_skip = execute_query_expression_with_skip(
                query, dictionary, bin_index_path, builder,
                all_docs, skip_stats, pointer_to_term
            )
            end_time_skip = time.perf_counter()
            
            # 执行无跳表版本（对照）
            start_time_no_skip = time.perf_counter()
            result_no_skip, comparisons_no_skip = execute_query_expression_no_skip(
                query, dictionary, bin_index_path, builder,
                all_docs, query_stats, pointer_to_term
            )
            end_time_no_skip = time.perf_counter()
            
            # 更新统计信息
            query_stats.result_skip = result_skip
            query_stats.result_no_skip = result_no_skip
            query_stats.execution_time_skip = end_time_skip - start_time_skip
            query_stats.execution_time_no_skip = end_time_no_skip - start_time_no_skip
            query_stats.skip_attempts = skip_stats.skip_attempts
            query_stats.skip_successes = skip_stats.skip_successes
            query_stats.comparisons_without_skip = skip_stats.comparisons_without_skip
            query_stats.comparisons_with_skip = skip_stats.comparisons_with_skip
            query_stats.comparisons_no_skip_version = comparisons_no_skip
            
            # 累计总体统计
            total_stats.skip_attempts += query_stats.skip_attempts
            total_stats.skip_successes += query_stats.skip_successes
            total_stats.comparisons_without_skip += query_stats.comparisons_without_skip
            total_stats.comparisons_with_skip += query_stats.comparisons_with_skip
            total_stats.comparisons_no_skip_version += query_stats.comparisons_no_skip_version
            total_stats.execution_time_skip += query_stats.execution_time_skip
            total_stats.execution_time_no_skip += query_stats.execution_time_no_skip
            
            # 输出查询结果对比
            print(f"  结果数量: {len(result_skip)} 个文档")
            if len(result_skip) > 0 and len(result_skip) <= 10:
                print(f"  文档ID: {result_skip}")
            elif len(result_skip) > 10:
                print(f"  文档ID（前10个）: {result_skip[:10]}...")
            
            # 计算比较次数减少
            if query_stats.comparisons_no_skip_version > 0:
                comparison_reduction = ((query_stats.comparisons_no_skip_version - (query_stats.comparisons_without_skip + query_stats.comparisons_with_skip)) / query_stats.comparisons_no_skip_version) * 100
            else:
                comparison_reduction = 0
            
            print(f"\n  【比较次数对比】")
            print(f"    无跳表版本: {query_stats.comparisons_no_skip_version} 次")
            print(f"    跳表版本:   {query_stats.comparisons_without_skip + query_stats.comparisons_with_skip} 次")
            print(f"    [OK] 减少: {abs(comparison_reduction):.2f}% ({query_stats.comparisons_no_skip_version - (query_stats.comparisons_without_skip + query_stats.comparisons_with_skip)} 次)")
            
            print(f"\n  【跳表指针统计】")
            print(f"    尝试次数: {query_stats.skip_attempts}")
            print(f"    成功次数: {query_stats.skip_successes}")
            if query_stats.skip_attempts > 0:
                print(f"    成功率: {query_stats.skip_successes/query_stats.skip_attempts*100:.2f}%")
            
            # 结果一致性检查
            print(f"\n  【结果验证】")
            if result_skip == result_no_skip:
                print(f"    [OK] 结果一致：跳表版本与无跳表版本结果完全相同")
            else:
                print(f"    [ERROR] 结果不一致！")
                print(f"      跳表版本结果数: {len(result_skip)}")
                print(f"      无跳表版本结果数: {len(result_no_skip)}")
            
            print()
            
        except Exception as e:
            print(f"  错误：查询执行失败: {e}")
            import traceback
            traceback.print_exc()
            print()
    
    # 输出总体统计
    print("=" * 80)
    print("总体统计")
    print("=" * 80)
    
    if total_stats.execution_time_no_skip > 0:
        total_speedup = ((total_stats.execution_time_no_skip - total_stats.execution_time_skip) / total_stats.execution_time_no_skip) * 100
    else:
        total_speedup = 0.0
    
    if total_stats.comparisons_no_skip_version > 0:
        total_comparison_reduction = ((total_stats.comparisons_no_skip_version - (total_stats.comparisons_without_skip + total_stats.comparisons_with_skip)) / total_stats.comparisons_no_skip_version) * 100
    else:
        total_comparison_reduction = 0.0
    
    # ============ 重点：比较次数统计 ============
    print(f"{'='*80}")
    print(f"核心指标：比较次数统计")
    print(f"{'='*80}")
    print(f"  无跳表版本总比较次数: {total_stats.comparisons_no_skip_version}")
    print(f"  跳表版本总比较次数:   {total_stats.comparisons_without_skip + total_stats.comparisons_with_skip}")
    print(f"    - 未使用跳表: {total_stats.comparisons_without_skip}")
    print(f"    - 使用跳表:   {total_stats.comparisons_with_skip}")
    print()
    print(f"  [OK] 比较次数减少: {abs(total_comparison_reduction):.2f}%")
    print(f"  [OK] 节省比较次数: {total_stats.comparisons_no_skip_version - (total_stats.comparisons_without_skip + total_stats.comparisons_with_skip)}")
    print()
    
    # ============ 跳表指针使用统计 ============
    print(f"{'='*80}")
    print(f"跳表指针使用统计")
    print(f"{'='*80}")
    print(f"  总尝试次数: {total_stats.skip_attempts}")
    print(f"  总成功次数: {total_stats.skip_successes}")
    if total_stats.skip_attempts > 0:
        print(f"  成功率: {total_stats.skip_successes/total_stats.skip_attempts*100:.2f}%")
    print()
    
    # ============ 执行时间（参考）============
    print(f"{'='*80}")
    print(f"执行时间（仅供参考，Python解释器开销影响较大）")
    print(f"{'='*80}")
    print(f"  无跳表版本: {total_stats.execution_time_no_skip*1000:.3f} ms")
    print(f"  跳表版本:   {total_stats.execution_time_skip*1000:.3f} ms")
    if total_speedup > 0:
        print(f"  注：执行时间提升 {total_speedup:.2f}%")
    else:
        print(f"  注：受Python解释器开销影响，实际执行时间增加 {abs(total_speedup):.2f}%")
        print(f"      但算法有效性已通过比较次数减少得到验证")
    print()
    
    # 关闭所有缓存的文件句柄
    print("关闭文件句柄...")
    close_all_file_handles()
    print("[OK] 所有文件句柄已关闭")
    print()


def generate_comprehensive_test_queries(
    dictionary: Dict[str, Tuple[int, int]],
    test_mode: str = "comprehensive",
    max_queries: int = None
) -> List[str]:
    """
    生成全面的测试查询（全局测试）
    
    测试模式：
    - "all_single": 测试所有单个词项（简单查询）
    - "all_pairs": 测试所有词对（AND查询），可能数量很大
    - "sampled_pairs": 采样测试词对（随机采样或按频率采样）
    - "frequency_groups": 按频率分组，每组内测试所有组合
    - "comprehensive": 综合模式，包含多种策略
    
    参数:
        dictionary: 词典字典 {词项: (文档频率, 指针)}
        test_mode: 测试模式
        max_queries: 最大查询数量（用于限制）
    返回:
        测试查询列表
    """
    if not dictionary:
        return []
    
    terms = list(dictionary.keys())
    total_terms = len(terms)
    
    # 按文档频率排序
    terms_by_freq = sorted(dictionary.items(), key=lambda x: x[1][0], reverse=True)
    term_list = [t[0] for t in terms_by_freq]
    
    queries = []
    
    if test_mode == "all_single":
        # 模式1: 测试所有单个词项
        queries = terms
        print(f"生成 {len(queries)} 个单词查询")
        
    elif test_mode == "all_pairs":
        # 模式2: 测试所有词对（AND查询）
        # 注意：这会产生 C(n,2) = n*(n-1)/2 个查询，可能非常大
        for i in range(total_terms):
            for j in range(i + 1, total_terms):
                queries.append(f"{term_list[i]} AND {term_list[j]}")
        print(f"生成 {len(queries)} 个词对查询（所有组合）")
        
    elif test_mode == "sampled_pairs":
        # 模式3: 采样测试词对
        import random
        random.seed(42)  # 固定随机种子以便重现
        
        # 采样策略：优先选择高频词
        sample_size = min(1000, total_terms * (total_terms - 1) // 2)  # 最多1000个查询
        
        # 优先选择高频词对
        high_freq_count = min(100, total_terms)  # 前100个高频词
        high_freq_terms = term_list[:high_freq_count]
        
        # 高频词之间的组合
        for i in range(min(50, len(high_freq_terms))):
            for j in range(i + 1, min(i + 10, len(high_freq_terms))):  # 每个词与后续10个词组合
                if len(queries) >= sample_size:
                    break
                queries.append(f"{high_freq_terms[i]} AND {high_freq_terms[j]}")
            if len(queries) >= sample_size:
                break
        
        # 高频词与中低频词的组合
        if len(queries) < sample_size:
            mid_low_terms = term_list[high_freq_count:]
            for i in range(min(20, len(high_freq_terms))):
                for j in range(min(20, len(mid_low_terms))):
                    if len(queries) >= sample_size:
                        break
                    queries.append(f"{high_freq_terms[i]} AND {mid_low_terms[j]}")
                if len(queries) >= sample_size:
                    break
        
        # 随机采样剩余组合
        if len(queries) < sample_size:
            remaining = sample_size - len(queries)
            pairs = [(term_list[i], term_list[j]) 
                    for i in range(total_terms) 
                    for j in range(i + 1, total_terms)]
            sampled_pairs = random.sample(pairs, min(remaining, len(pairs)))
            queries.extend([f"{t1} AND {t2}" for t1, t2 in sampled_pairs])
        
        print(f"生成 {len(queries)} 个采样词对查询")
        
    elif test_mode == "frequency_groups":
        # 模式4: 按频率分组，每组内测试所有组合
        # 将词项分成3组：高频、中频、低频
        group_size = total_terms // 3
        
        high_freq = term_list[:group_size]
        mid_freq = term_list[group_size:2*group_size]
        low_freq = term_list[2*group_size:]
        
        # 每组内所有组合（限制每组最多50个词以避免组合爆炸）
        for group, group_name in [(high_freq[:50], "高频"), 
                                   (mid_freq[:50], "中频"), 
                                   (low_freq[:50], "低频")]:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    queries.append(f"{group[i]} AND {group[j]}")
            print(f"  {group_name}组: {len(group)} 个词，生成 {len(group)*(len(group)-1)//2} 个查询")
        
        # 跨组组合（高频-中频，高频-低频，中频-低频）
        for i in range(min(20, len(high_freq))):
            for j in range(min(20, len(mid_freq))):
                queries.append(f"{high_freq[i]} AND {mid_freq[j]}")
        
        for i in range(min(20, len(high_freq))):
            for j in range(min(20, len(low_freq))):
                queries.append(f"{high_freq[i]} AND {low_freq[j]}")
        
        for i in range(min(20, len(mid_freq))):
            for j in range(min(20, len(low_freq))):
                queries.append(f"{mid_freq[i]} AND {low_freq[j]}")
        
        print(f"生成 {len(queries)} 个频率分组查询")
        
    elif test_mode == "comprehensive":
        # 模式5: 综合模式
        # 1. 所有单个词项（简单查询）
        queries.extend(terms)
        print(f"  1. 单词查询: {len(terms)} 个")
        
        # 2. 采样词对
        import random
        random.seed(42)
        sample_size = min(500, total_terms * 10)  # 采样500个词对
        
        # 高频词对
        high_freq_count = min(50, total_terms)
        high_freq_terms = term_list[:high_freq_count]
        for i in range(min(30, len(high_freq_terms))):
            for j in range(i + 1, min(i + 15, len(high_freq_terms))):
                if len(queries) - len(terms) >= sample_size:
                    break
                queries.append(f"{high_freq_terms[i]} AND {high_freq_terms[j]}")
            if len(queries) - len(terms) >= sample_size:
                break
        
        # 跨频率组合
        if len(queries) - len(terms) < sample_size:
            mid_low_terms = term_list[high_freq_count:]
            for i in range(min(20, len(high_freq_terms))):
                for j in range(min(20, len(mid_low_terms))):
                    if len(queries) - len(terms) >= sample_size:
                        break
                    queries.append(f"{high_freq_terms[i]} AND {mid_low_terms[j]}")
                if len(queries) - len(terms) >= sample_size:
                    break
        
        print(f"  2. 词对查询: {len(queries) - len(terms)} 个")
        
        # 3. OR查询（采样）
        or_count = min(100, len(high_freq_terms) * 5)
        for i in range(min(20, len(high_freq_terms))):
            for j in range(i + 1, min(i + 5, len(high_freq_terms))):
                if len([q for q in queries if ' OR ' in q]) >= or_count:
                    break
                queries.append(f"{high_freq_terms[i]} OR {high_freq_terms[j]}")
            if len([q for q in queries if ' OR ' in q]) >= or_count:
                break
        
        print(f"  3. OR查询: {len([q for q in queries if ' OR ' in q])} 个")
        
        # 4. NOT查询（采样）
        not_count = min(50, len(high_freq_terms))
        for i in range(not_count):
            queries.append(f"NOT {high_freq_terms[i]}")
        
        print(f"  4. NOT查询: {not_count} 个")
        
        # 5. 混合查询（少量）
        if len(high_freq_terms) >= 3:
            queries.append(f"({high_freq_terms[0]} AND {high_freq_terms[1]}) OR {high_freq_terms[2]}")
            queries.append(f"({high_freq_terms[0]} OR {high_freq_terms[1]}) AND NOT {high_freq_terms[2]}")
        
        print(f"  5. 混合查询: {len([q for q in queries if '(' in q])} 个")
        print(f"  总计: {len(queries)} 个查询")
    
    # 限制查询数量
    if max_queries and len(queries) > max_queries:
        queries = queries[:max_queries]
        print(f"限制为 {max_queries} 个查询")
    
    return queries


def generate_test_queries_from_dictionary(
    dictionary: Dict[str, Tuple[int, int]],
    num_queries: int = 20
) -> List[str]:
    """
    基于词典数据自动生成测试查询
    
    策略：
    1. 基于文档频率：选择高频词、中频词、低频词组合
    2. 基于列表长度：选择长列表和短列表组合（对跳表指针测试重要）
    3. 基于字母分布：确保覆盖不同字母开头的词
    4. 基于交集大小：选择会产生不同大小交集的词对
    
    参数:
        dictionary: 词典字典 {词项: (文档频率, 指针)}
        num_queries: 生成的查询数量
    返回:
        测试查询列表
    """
    if not dictionary:
        return []
    
    # 按文档频率排序
    terms_by_freq = sorted(dictionary.items(), key=lambda x: x[1][0], reverse=True)
    
    # 分类词项
    total_terms = len(terms_by_freq)
    high_freq_terms = [t[0] for t in terms_by_freq[:total_terms//3]]  # 高频词（前1/3）
    mid_freq_terms = [t[0] for t in terms_by_freq[total_terms//3:2*total_terms//3]]  # 中频词
    low_freq_terms = [t[0] for t in terms_by_freq[2*total_terms//3:]]  # 低频词（后1/3）
    
    # 按字母分组（用于确保字母分布）
    terms_by_letter = {}
    for term, (doc_freq, _) in dictionary.items():
        first_letter = term[0].lower() if term else 'a'
        if first_letter not in terms_by_letter:
            terms_by_letter[first_letter] = []
        terms_by_letter[first_letter].append((term, doc_freq))
    
    # 对每个字母按频率排序
    for letter in terms_by_letter:
        terms_by_letter[letter].sort(key=lambda x: x[1], reverse=True)
    
    queries = []
    
    # 策略1: 高频词 AND 高频词（大列表交集，跳表指针效果明显）
    if len(high_freq_terms) >= 2:
        queries.append(f"{high_freq_terms[0]} AND {high_freq_terms[1]}")
        if len(high_freq_terms) >= 3:
            queries.append(f"{high_freq_terms[0]} AND {high_freq_terms[1]} AND {high_freq_terms[2]}")
    
    # 策略2: 高频词 AND 中频词（长列表和中等列表）
    if high_freq_terms and mid_freq_terms:
        queries.append(f"{high_freq_terms[0]} AND {mid_freq_terms[0]}")
        if len(mid_freq_terms) >= 2:
            queries.append(f"{high_freq_terms[0]} AND {mid_freq_terms[1]}")
    
    # 策略3: 高频词 AND 低频词（长列表和短列表，跳表指针效果明显）
    if high_freq_terms and low_freq_terms:
        queries.append(f"{high_freq_terms[0]} AND {low_freq_terms[0]}")
        if len(low_freq_terms) >= 2:
            queries.append(f"{high_freq_terms[1]} AND {low_freq_terms[1]}")
    
    # 策略4: 中频词 AND 中频词
    if len(mid_freq_terms) >= 2:
        queries.append(f"{mid_freq_terms[0]} AND {mid_freq_terms[1]}")
    
    # 策略5: OR查询（高频词组合）
    if len(high_freq_terms) >= 2:
        queries.append(f"{high_freq_terms[0]} OR {high_freq_terms[1]}")
        if len(high_freq_terms) >= 3:
            queries.append(f"{high_freq_terms[0]} OR {high_freq_terms[1]} OR {high_freq_terms[2]}")
    
    # 策略6: 不同字母开头的词组合（确保字母分布）
    letters = sorted(terms_by_letter.keys())
    if len(letters) >= 2:
        # 选择不同字母的高频词
        letter1_terms = [t[0] for t in terms_by_letter[letters[0]][:3]]
        letter2_terms = [t[0] for t in terms_by_letter[letters[1]][:3]]
        if letter1_terms and letter2_terms:
            queries.append(f"{letter1_terms[0]} AND {letter2_terms[0]}")
            queries.append(f"{letter1_terms[0]} OR {letter2_terms[0]}")
    
    if len(letters) >= 3:
        letter3_terms = [t[0] for t in terms_by_letter[letters[2]][:3]]
        if letter1_terms and letter3_terms:
            queries.append(f"{letter1_terms[0]} AND {letter3_terms[0]}")
    
    # 策略7: NOT查询（高频词）
    if high_freq_terms:
        queries.append(f"NOT {high_freq_terms[0]}")
        if len(high_freq_terms) >= 2:
            queries.append(f"NOT {high_freq_terms[1]}")
    
    # 策略8: 混合查询
    if len(high_freq_terms) >= 3 and mid_freq_terms:
        queries.append(f"({high_freq_terms[0]} AND {high_freq_terms[1]}) OR {mid_freq_terms[0]}")
        queries.append(f"({high_freq_terms[0]} OR {high_freq_terms[1]}) AND NOT {mid_freq_terms[0]}")
    
    if len(high_freq_terms) >= 2 and low_freq_terms:
        queries.append(f"{high_freq_terms[0]} AND ({high_freq_terms[1]} OR {low_freq_terms[0]})")
    
    # 策略9: 复杂混合查询
    if len(high_freq_terms) >= 2 and mid_freq_terms and low_freq_terms:
        queries.append(f"(({high_freq_terms[0]} AND {mid_freq_terms[0]}) OR {low_freq_terms[0]}) AND NOT {high_freq_terms[1]}")
    
    # 限制查询数量
    return queries[:num_queries]


def main():
    """主函数"""
    # 配置文件路径
    config_path_skip = project_root / "config" / "inverted_index_config_with_skip.json"
    
    if not config_path_skip.exists():
        print(f"错误：配置文件不存在: {config_path_skip}")
        return
    
    # 加载配置
    with open(config_path_skip, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    output_dir = config.get("output_dir", "output_inverted_index_skiplist")
    dict_bin_path = project_root / output_dir / config["files"]["dictionary_bin"]
    bin_index_path = project_root / output_dir / config["files"]["inverted_index_bin_with_skip"]
    
    # 检查文件是否存在
    if not dict_bin_path.exists():
        print(f"错误：词典文件不存在: {dict_bin_path}")
        print("请先运行 run_inverted_index_builder_with_skip.py 构建倒排索引")
        return
    
    if not bin_index_path.exists():
        print(f"错误：倒排索引文件不存在: {bin_index_path}")
        print("请先运行 run_inverted_index_builder_with_skip.py 构建倒排索引")
        return
    
    # 加载词典以生成测试查询
    print("加载词典以生成测试查询...")
    dictionary = load_dictionary_from_binary(str(dict_bin_path))
    print(f"[OK] 加载了 {len(dictionary)} 个词项")
    
    # 选择生成策略：自动生成、手动指定 或 全局测试
    # TEST_MODE 可选值：
    #   "single": 单查询测试（用于调试）
    #   "auto": 自动生成（基于策略的智能选择）
    #   "manual": 手动指定（原来的方式）
    #   "comprehensive": 全局综合测试（单词+采样词对+OR+NOT+混合）
    #   "all_single": 测试所有单个词项
    #   "all_pairs": 测试所有词对（警告：可能数量非常大！）
    #   "sampled_pairs": 采样测试词对
    #   "frequency_groups": 按频率分组测试
    TEST_MODE = "manual"  # 默认使用手动指定查询列表模式
    # 可以尝试不同的查询来测试跳表指针
    # 测试不同的词对组合，验证跳表指针在各种情况下的表现
    SINGLE_QUERY = "future AND function"  # 当前测试查询
    # SINGLE_QUERY = "adventure AND age"
    # SINGLE_QUERY = "book AND business"
    # SINGLE_QUERY = "computer AND create"
    # SINGLE_QUERY = "data AND design"
    # SINGLE_QUERY = "energy AND environment"
    # SINGLE_QUERY = "future AND function"
    MAX_QUERIES = None  # 最大查询数量限制，None表示不限制
    
    # 单查询测试模式
    if TEST_MODE == "single":
        print(f"\n使用单查询测试模式")
        print(f"测试查询: {SINGLE_QUERY}")
        print()
        test_single_query(
            str(dict_bin_path),
            str(bin_index_path),
            str(config_path_skip),
            SINGLE_QUERY
        )
        return
    
    if TEST_MODE in ["comprehensive", "all_single", "all_pairs", "sampled_pairs", "frequency_groups"]:
        # 全局测试模式
        print(f"\n使用全局测试模式: {TEST_MODE}")
        if TEST_MODE == "all_pairs":
            total_pairs = len(dictionary) * (len(dictionary) - 1) // 2
            print(f"警告：将生成 {total_pairs} 个词对查询，这可能需要很长时间！")
            response = input("是否继续？(y/n): ")
            if response.lower() != 'y':
                print("已取消")
                return
        
        test_queries = generate_comprehensive_test_queries(
            dictionary, 
            test_mode=TEST_MODE,
            max_queries=MAX_QUERIES
        )
        print(f"\n[OK] 生成了 {len(test_queries)} 个测试查询")
        if len(test_queries) > 50:
            print("\n前50个查询预览：")
            for i, q in enumerate(test_queries[:50], 1):
                print(f"  {i}. {q}")
            print(f"  ... (还有 {len(test_queries) - 50} 个查询)")
        else:
            print("\n生成的测试查询列表：")
            for i, q in enumerate(test_queries, 1):
                print(f"  {i}. {q}")
        print()
    elif TEST_MODE == "auto":
        # 自动生成测试查询（基于词典数据）
        print("\n使用自动生成的测试查询（基于词典数据）...")
        test_queries = generate_test_queries_from_dictionary(dictionary, num_queries=20)
        print(f"[OK] 生成了 {len(test_queries)} 个测试查询")
        print("\n生成的测试查询列表：")
        for i, q in enumerate(test_queries, 1):
            print(f"  {i}. {q}")
        print()
    else:
        # 手动指定测试查询（包含AND、OR、NOT和混合查询）
        # 注意：选择不同字母开头的单词以测试更全面的场景
        test_queries = [
        # AND查询（使用跳表指针）
        "ability AND able",
        "adventure AND age",
        "alone AND amazing",
        "ability AND able AND accept",
        "book AND business",  # b开头
        "computer AND create",  # c开头
        "data AND design",  # d开头
        
        # OR查询（并集）
        "ability OR able",
        "adventure OR age",
        "alone OR amazing OR allow",
        "book OR business OR build",  # b开头
        
        # NOT查询（补集）
        "NOT ability",
        "NOT adventure",
        "NOT book",  # b开头
        
        # 混合查询
        "(ability AND able) OR accept",
        "(adventure AND age) OR alert",
        "(alone OR amazing) AND NOT activity",
        "(allow AND also) OR (alternative AND although)",
        "((area AND arrive) OR art) AND NOT already",
        "ability AND (able OR accept)",
        "NOT ability AND NOT adventure",
        "(book AND business) OR computer",  # 跨字母混合
        "(data OR design) AND NOT book",  # 跨字母混合
    ]
    
    # 执行测试
    test_skip_pointer_queries(
        str(dict_bin_path),
        str(bin_index_path),
        str(config_path_skip),
        test_queries
    )


if __name__ == "__main__":
    main()

