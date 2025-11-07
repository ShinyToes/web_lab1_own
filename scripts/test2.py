"""
跳表指针测试脚本
测试带跳表指针的倒排索引查询，统计跳表指针的尝试次数和成功次数
"""

import os
import sys
import struct
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

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
    从带跳表指针的二进制倒排索引文件中读取posting list
    
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
    with open(bin_index_path, 'rb') as f:
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
    result = []
    i, j = 0, 0
    
    # 将跳表指针转换为按位置索引排序的列表，方便查找
    # 跳表指针格式：(位置索引, 目标doc_id, 目标file_pos)
    sorted_skip_pointers = sorted(long_skip_pointers, key=lambda x: x[0])
    skip_ptr_idx = 0
    
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
            can_skip = False
            
            # 查找当前位置之后可用的跳表指针
            # 跳过已经使用过的跳表指针
            while skip_ptr_idx < len(sorted_skip_pointers):
                skip_pos, target_doc_id, _ = sorted_skip_pointers[skip_ptr_idx]
                
                # 如果跳表指针位置在当前或之后的位置
                if skip_pos >= j:
                    stats.skip_attempts += 1
                    
                    # 检查是否可以跳过
                    # 如果short_list[i] < target_doc_id，说明可以跳过中间的元素
                    if short_list[i] < target_doc_id:
                        # 可以跳过！找到目标doc_id在long_list中的实际位置
                        # 由于列表已排序，可以使用二分查找或线性查找
                        target_j = j
                        for k in range(j, len(long_list)):
                            if long_list[k] >= target_doc_id:
                                target_j = k
                                break
                        
                        if target_j > j:
                            # 成功跳过
                            stats.skip_successes += 1
                            stats.comparisons_with_skip += (target_j - j)
                            j = target_j
                            can_skip = True
                            # 不增加skip_ptr_idx，因为可能还需要使用这个跳表指针
                            break
                        else:
                            # 目标位置就是当前位置，不能跳过
                            skip_ptr_idx += 1
                            break
                    else:
                        # short_list[i] >= target_doc_id，不能跳过这个跳表指针
                        # 继续检查下一个跳表指针
                        skip_ptr_idx += 1
                        continue
                else:
                    # 跳表指针位置在当前位置之前，跳过这个跳表指针
                    skip_ptr_idx += 1
            
            if not can_skip:
                # 没有可用的跳表指针或无法跳过，正常前进
                j += 1
                stats.comparisons_without_skip += 1
    
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
    print(f"✓ 加载了 {len(dictionary)} 个词项")
    print()
    
    # 构建指针到词项的映射（用于确定读取范围）
    pointer_to_term = {}
    for term, (doc_freq, pointer) in dictionary.items():
        pointer_to_term[pointer] = term
    
    # 获取所有文档ID集合（用于NOT操作）
    print("获取所有文档ID集合（用于NOT操作）...")
    all_docs = get_all_document_ids(dictionary, bin_index_path, builder, pointer_to_term)
    print(f"✓ 总文档数: {len(all_docs)}")
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
            
            print(f"\n  【跳表版本】")
            print(f"    执行时间: {query_stats.execution_time_skip*1000:.3f} ms")
            print(f"    比较次数: {query_stats.comparisons_without_skip + query_stats.comparisons_with_skip}")
            print(f"    跳表指针尝试次数: {query_stats.skip_attempts}")
            print(f"    跳表指针成功次数: {query_stats.skip_successes}")
            if query_stats.skip_attempts > 0:
                print(f"    跳表指针成功率: {query_stats.skip_successes/query_stats.skip_attempts*100:.2f}%")
            
            print(f"\n  【无跳表版本（对照）】")
            print(f"    执行时间: {query_stats.execution_time_no_skip*1000:.3f} ms")
            print(f"    比较次数: {query_stats.comparisons_no_skip_version}")
            
            # 计算性能提升
            if query_stats.execution_time_no_skip > 0:
                speedup = ((query_stats.execution_time_no_skip - query_stats.execution_time_skip) / query_stats.execution_time_no_skip) * 100
                print(f"\n  【性能对比】")
                print(f"    时间提升: {speedup:+.2f}%")
            
            if query_stats.comparisons_no_skip_version > 0:
                comparison_reduction = ((query_stats.comparisons_no_skip_version - (query_stats.comparisons_without_skip + query_stats.comparisons_with_skip)) / query_stats.comparisons_no_skip_version) * 100
                print(f"    比较次数减少: {comparison_reduction:+.2f}%")
            
            # 结果一致性检查
            if result_skip == result_no_skip:
                print(f"    结果一致性: ✓ 一致")
            else:
                print(f"    结果一致性: ✗ 不一致")
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
    
    print(f"总执行时间:")
    print(f"  - 跳表版本: {total_stats.execution_time_skip*1000:.3f} ms")
    print(f"  - 无跳表版本: {total_stats.execution_time_no_skip*1000:.3f} ms")
    print(f"  - 总体性能提升: {total_speedup:+.2f}%")
    print()
    print(f"总比较次数:")
    print(f"  - 跳表版本: {total_stats.comparisons_without_skip + total_stats.comparisons_with_skip}")
    print(f"  - 无跳表版本: {total_stats.comparisons_no_skip_version}")
    print(f"  - 比较次数减少: {total_comparison_reduction:+.2f}%")
    print()
    print(f"跳表指针统计:")
    print(f"  - 总尝试次数: {total_stats.skip_attempts}")
    print(f"  - 总成功次数: {total_stats.skip_successes}")
    if total_stats.skip_attempts > 0:
        print(f"  - 总成功率: {total_stats.skip_successes/total_stats.skip_attempts*100:.2f}%")
    print()


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
    
    # 设计测试查询（包含AND、OR、NOT和混合查询）
    test_queries = [
        # AND查询（使用跳表指针）
        "ability AND able",
        "adventure AND age",
        "alone AND amazing",
        "ability AND able AND accept",
        
        # OR查询（并集）
        "ability OR able",
        "adventure OR age",
        "alone OR amazing OR allow",
        
        # NOT查询（补集）
        "NOT ability",
        "NOT adventure",
        
        # 混合查询
        "(ability AND able) OR accept",
        "(adventure AND age) OR alert",
        "(alone OR amazing) AND NOT activity",
        "(allow AND also) OR (alternative AND although)",
        "((area AND arrive) OR art) AND NOT already",
        "ability AND (able OR accept)",
        "NOT ability AND NOT adventure",
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

