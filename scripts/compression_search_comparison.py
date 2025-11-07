"""
索引压缩前后检索效率对比实验
对比未压缩索引和压缩索引在布尔检索中的性能差异
包括：查询时间、空间占用、解码开销等指标
"""

import time
import struct
import os
from typing import List, Dict, Set, Tuple


# ==================== 配置项 ====================

BASE_DIR = "D:/2025_2/web/Lab/web_lab1"

# 未压缩索引文件路径
DICT_BIN_PATH = f"{BASE_DIR}/output_inverted_index/dictionary.bin"
INDEX_BIN_PATH = f"{BASE_DIR}/output_inverted_index/inverted_index.bin"

# 压缩索引文件路径
COMPRESSED_DICT_PATH = f"{BASE_DIR}/output_inverted_index/compressed_dictionary.bin"
COMPRESSED_INDEX_PATH = f"{BASE_DIR}/output_inverted_index/inverted_index_compressed.bin"

# 输出目录
OUTPUT_DIR = f"{BASE_DIR}/part_4/output"
COMPARISON_OUTPUT_FILE = f"{OUTPUT_DIR}/compression_comparison_results.txt"


# ==================== 常量定义 ====================

# 未压缩索引格式常量
TERM_MAX_LENGTH = 20  # 词项最大长度（字节）
DOC_FREQ_SIZE = 4  # 文档频率字段大小（字节）
POINTER_SIZE = 4  # 倒排表指针字段大小（字节）
ENTRY_SIZE = TERM_MAX_LENGTH + DOC_FREQ_SIZE + POINTER_SIZE  # 每个词项总开销（28字节）
DOC_ID_SIZE = 4  # 文档ID存储大小（字节）

# 压缩索引格式常量
TERM_PREFIX_BYTES = 8  # 词项前缀固定 8 字节
TERM_PTR_BYTES = 3  # 词项指针字段，3 字节
TERM_LEN_BYTES = 1  # 词项长度字段，1 字节
FREQ_BYTES = 4  # 文档频率字段，4 字节
PTR_BYTES = 4  # 倒排表指针字段，4 字节


# ==================== 模块 1：未压缩索引加载 ====================

def load_uncompressed_dictionary(dict_bin_path: str) -> Dict[str, Tuple[int, int]]:
    """
    从二进制文件加载未压缩词典
    
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
            term = term_bytes.rstrip(b'\x00').decode('utf-8', errors='ignore')
            
            # 解析文档频率（4字节）
            doc_freq = struct.unpack('I', entry_data[TERM_MAX_LENGTH:TERM_MAX_LENGTH+DOC_FREQ_SIZE])[0]
            
            # 解析倒排表指针（4字节）
            pointer = struct.unpack('I', entry_data[TERM_MAX_LENGTH+DOC_FREQ_SIZE:ENTRY_SIZE])[0]
            
            dictionary[term] = (doc_freq, pointer)
    
    return dictionary


def load_uncompressed_posting_list(bin_index_path: str, pointer: int, doc_freq: int) -> List[int]:
    """
    从未压缩倒排索引文件中读取posting list
    
    参数:
        bin_index_path: 二进制倒排索引文件路径
        pointer: 倒排表指针（字节偏移）
        doc_freq: 文档频率（posting list长度）
    返回:
        文档ID列表（已排序）
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
    
    # 写入即为升序，此处不再二次排序，避免不必要的开销
    return doc_ids


# ==================== 模块 2：压缩索引VB+Gap解码 ====================

def vb_decode_from_file(file_handle, doc_freq: int) -> List[int]:
    """
    从文件中读取VB编码的数据并解码为指定数量的整数
    
    参数:
        file_handle: 文件句柄（已定位到起始位置）
        doc_freq: 需要解码的整数数量
    返回:
        解码后的整数列表（Gap编码）
    """
    gaps = []
    n = 0
    
    while len(gaps) < doc_freq:
        byte_data = file_handle.read(1)
        if not byte_data:
            break
        
        byte = byte_data[0]
        if byte < 128:
            n = 128 * n + byte
        else:
            n = 128 * n + (byte - 128)
            gaps.append(n)
            n = 0
    
    return gaps


def decode_gaps(gaps: List[int]) -> List[int]:
    """
    Gap解码：将差值编码转换为绝对值列表
    
    参数:
        gaps: 差值编码列表
    返回:
        绝对值列表
    """
    if not gaps:
        return []
    
    numbers = [gaps[0]]
    for i in range(1, len(gaps)):
        numbers.append(numbers[i-1] + gaps[i])
    
    return numbers


def load_compressed_dictionary(dict_bin_path: str) -> Tuple[Dict[str, Tuple[int, int]], str, int]:
    """
    加载压缩词典
    
    参数:
        dict_bin_path: 压缩词典文件路径
    返回:
        (词典, 单一字符串, K值)
        词典: {term: (doc_freq, pointer)}
    """
    with open(dict_bin_path, 'rb') as f:
        # 读取单一字符串长度
        string_len = struct.unpack('I', f.read(4))[0]
        
        # 读取单一字符串内容
        single_string = f.read(string_len).decode('utf-8', errors='ignore')
        
        # 读取词项数量
        num_terms = struct.unpack('I', f.read(4))[0]
        
        # 读取K值
        K = struct.unpack('B', f.read(1))[0]
        
        # 读取所有元数据
        metadata_list = []
        
        for idx in range(num_terms):
            # 读取词项前缀
            prefix = f.read(TERM_PREFIX_BYTES).rstrip(b'\x00').decode('utf-8', errors='ignore')
            
            # 判断是否存储指针
            store_pointer = ((idx + 1) % K == 0) or (idx == num_terms - 1)
            
            if store_pointer:
                # 读取指针
                term_ptr = int.from_bytes(f.read(TERM_PTR_BYTES), byteorder='little')
                term_len = None
            else:
                # 读取长度
                term_len = struct.unpack('B', f.read(TERM_LEN_BYTES))[0]
                term_ptr = None
            
            # 读取文档频率
            freq = struct.unpack('I', f.read(FREQ_BYTES))[0]
            
            # 读取倒排表指针
            inv_ptr = struct.unpack('I', f.read(PTR_BYTES))[0]
            
            metadata_list.append({
                'prefix': prefix,
                'store_pointer': store_pointer,
                'term_ptr': term_ptr,
                'term_len': term_len,
                'freq': freq,
                'inv_ptr': inv_ptr
            })
        
        # 从单一字符串中提取完整词项
        dictionary = {}
        current_pos = 0
        
        for i, meta in enumerate(metadata_list):
            if meta['store_pointer']:
                # 锚点：跳转到指定位置
                current_pos = meta['term_ptr']
                
                # 计算长度
                next_anchor_pos = len(single_string)
                intermediate_len = 0
                
                for j in range(i+1, len(metadata_list)):
                    if metadata_list[j]['store_pointer']:
                        next_anchor_pos = metadata_list[j]['term_ptr']
                        break
                    else:
                        intermediate_len += metadata_list[j]['term_len']
                
                term_len = next_anchor_pos - current_pos - intermediate_len
            else:
                # 非锚点：使用长度
                term_len = meta['term_len']
            
            # 提取词项
            if current_pos + term_len <= len(single_string):
                term = single_string[current_pos:current_pos+term_len]
            else:
                term = meta['prefix']
            
            current_pos += term_len
            
            dictionary[term] = (meta['freq'], meta['inv_ptr'])
        
        return dictionary, single_string, K


def load_compressed_posting_list(index_path: str, pointer: int, doc_freq: int) -> List[int]:
    """
    从压缩倒排表中加载posting list
    
    参数:
        index_path: 压缩倒排索引文件路径
        pointer: 倒排表指针
        doc_freq: 文档频率
    返回:
        文档ID列表
    """
    with open(index_path, 'rb') as f:
        f.seek(pointer)
        
        # VB解码
        gaps = vb_decode_from_file(f, doc_freq)
        
        # Gap解码
        doc_ids = decode_gaps(gaps)
        
        return doc_ids


# ==================== 模块 3：布尔查询操作 ====================

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


# ==================== 模块 4：查询执行引擎 ====================

def execute_query_on_uncompressed(query: str, dictionary: Dict[str, Tuple[int, int]], 
                                   index_path: str) -> Tuple[List[int], float]:
    """
    在未压缩索引上执行查询
    
    参数:
        query: 查询字符串（如 "apple AND banana"）
        dictionary: 未压缩词典
        index_path: 未压缩倒排索引路径
    返回:
        (结果文档列表, 执行时间)
    """
    start_time = time.perf_counter()
    
    query = query.strip()
    terms = query.split()
    
    if 'AND' in terms:
        # AND 查询
        query_terms = [t for t in terms if t not in ['AND', 'OR']]
        
        if len(query_terms) == 0:
            return [], time.perf_counter() - start_time
        
        # 加载第一个词项
        term1 = query_terms[0].lower()
        if term1 not in dictionary:
            return [], time.perf_counter() - start_time
        
        doc_freq1, pointer1 = dictionary[term1]
        result = load_uncompressed_posting_list(index_path, pointer1, doc_freq1)
        
        # 与后续词项求交集
        for term in query_terms[1:]:
            term = term.lower()
            if term not in dictionary:
                result = []
                break
            
            doc_freq, pointer = dictionary[term]
            doc_ids = load_uncompressed_posting_list(index_path, pointer, doc_freq)
            result = intersect(result, doc_ids)
        
    elif 'OR' in terms:
        # OR 查询
        query_terms = [t for t in terms if t not in ['AND', 'OR']]
        
        if len(query_terms) == 0:
            return [], time.perf_counter() - start_time
        
        result = []
        
        for term in query_terms:
            term = term.lower()
            if term in dictionary:
                doc_freq, pointer = dictionary[term]
                doc_ids = load_uncompressed_posting_list(index_path, pointer, doc_freq)
                result = union(result, doc_ids)
    
    else:
        # 单个词项查询
        term = query.lower()
        if term in dictionary:
            doc_freq, pointer = dictionary[term]
            result = load_uncompressed_posting_list(index_path, pointer, doc_freq)
        else:
            result = []
    
    execution_time = time.perf_counter() - start_time
    
    return result, execution_time


def execute_query_on_compressed(query: str, dictionary: Dict[str, Tuple[int, int]], 
                                 index_path: str) -> Tuple[List[int], float, int]:
    """
    在压缩索引上执行查询
    
    参数:
        query: 查询字符串（如 "apple AND banana"）
        dictionary: 压缩词典
        index_path: 压缩倒排索引路径
    返回:
        (结果文档列表, 执行时间, 解码次数)
    """
    start_time = time.perf_counter()
    decode_count = 0
    
    query = query.strip()
    terms = query.split()
    
    if 'AND' in terms:
        # AND 查询
        query_terms = [t for t in terms if t not in ['AND', 'OR']]
        
        if len(query_terms) == 0:
            return [], time.perf_counter() - start_time, decode_count
        
        # 加载第一个词项
        term1 = query_terms[0].lower()
        if term1 not in dictionary:
            return [], time.perf_counter() - start_time, decode_count
        
        doc_freq1, pointer1 = dictionary[term1]
        result = load_compressed_posting_list(index_path, pointer1, doc_freq1)
        decode_count += 1
        
        # 与后续词项求交集
        for term in query_terms[1:]:
            term = term.lower()
            if term not in dictionary:
                result = []
                break
            
            doc_freq, pointer = dictionary[term]
            doc_ids = load_compressed_posting_list(index_path, pointer, doc_freq)
            decode_count += 1
            result = intersect(result, doc_ids)
        
    elif 'OR' in terms:
        # OR 查询
        query_terms = [t for t in terms if t not in ['AND', 'OR']]
        
        if len(query_terms) == 0:
            return [], time.perf_counter() - start_time, decode_count
        
        result = []
        
        for term in query_terms:
            term = term.lower()
            if term in dictionary:
                doc_freq, pointer = dictionary[term]
                doc_ids = load_compressed_posting_list(index_path, pointer, doc_freq)
                decode_count += 1
                result = union(result, doc_ids)
    
    else:
        # 单个词项查询
        term = query.lower()
        if term in dictionary:
            doc_freq, pointer = dictionary[term]
            result = load_compressed_posting_list(index_path, pointer, doc_freq)
            decode_count += 1
        else:
            result = []
    
    execution_time = time.perf_counter() - start_time
    
    return result, execution_time, decode_count


# ==================== 模块 5：测试查询设计 ====================

def get_top_frequency_terms(dictionary: Dict[str, Tuple[int, int]], top_n: int = 20) -> List[str]:
    """
    从词典中提取文档频率最高的词项（高频词）
    
    参数:
        dictionary: 词典，键为词项，值为(文档频率, 倒排表指针)元组
        top_n: 返回前N个高频词
    返回:
        高频词列表（按文档频率降序排列）
    """
    # 按文档频率降序排序
    sorted_terms = sorted(dictionary.items(), key=lambda x: x[1][0], reverse=True)
    top_terms = [term for term, _ in sorted_terms[:top_n]]
    return top_terms


def design_comparison_queries(dictionary: Dict[str, Tuple[int, int]] = None) -> List[str]:
    """
    设计用于对比实验的查询（使用高频词）
    
    参数:
        dictionary: 词典，用于提取高频词（可选）
    返回:
        查询字符串列表
    """
    # 如果提供了词典，使用高频词生成查询
    if dictionary:
        top_terms = get_top_frequency_terms(dictionary, top_n=15)
        
        if len(top_terms) >= 3:
            queries = [
                # 单词查询（使用前3个高频词）
                top_terms[0],
                top_terms[1],
                top_terms[2],
                
                # AND 查询（2词项，使用高频词）
                f"{top_terms[0]} AND {top_terms[1]}",
                f"{top_terms[2]} AND {top_terms[3]}" if len(top_terms) > 3 else f"{top_terms[1]} AND {top_terms[2]}",
                f"{top_terms[1]} AND {top_terms[3]}" if len(top_terms) > 3 else f"{top_terms[0]} AND {top_terms[2]}",
                
                # AND 查询（3词项）
                f"{top_terms[0]} AND {top_terms[1]} AND {top_terms[2]}",
                f"{top_terms[1]} AND {top_terms[2]} AND {top_terms[3]}" if len(top_terms) > 3 else f"{top_terms[0]} AND {top_terms[1]} AND {top_terms[2]}",
                
                # OR 查询
                f"{top_terms[0]} OR {top_terms[1]}",
                f"{top_terms[2]} OR {top_terms[3]}" if len(top_terms) > 3 else f"{top_terms[1]} OR {top_terms[2]}",
                f"{top_terms[0]} OR {top_terms[1]} OR {top_terms[2]}",
            ]
        else:
            # 如果高频词不足，使用默认查询
            queries = [
                top_terms[0] if len(top_terms) > 0 else "default",
            ]
    else:
        # 如果没有提供词典，使用默认查询
        queries = [
            # 单词查询
            "group",
            "event",
            "dinner",
            
            # AND 查询（2词项）
            "group AND please",
            "come AND free",
            "dinner AND event",
            
            # AND 查询（3词项）
            "get AND join AND free",
            
            # OR 查询
            "group OR please",
            "dinner OR event",
            "come OR free OR see",
        ]
    
    return queries


# ==================== 模块 6：对比实验与结果分析 ====================

def get_file_size(file_path: str) -> int:
    """
    获取文件大小（字节）
    
    参数:
        file_path: 文件路径
    返回:
        文件大小（字节）
    """
    if os.path.exists(file_path):
        return os.path.getsize(file_path)
    return 0


def format_size(size_bytes: int) -> str:
    """
    格式化文件大小为可读字符串
    
    参数:
        size_bytes: 字节数
    返回:
        格式化的大小字符串
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    else:
        return f"{size_bytes / (1024 * 1024):.2f} MB"


def run_comparison_experiments(uncompressed_dict: Dict[str, Tuple[int, int]],
                                compressed_dict: Dict[str, Tuple[int, int]]) -> List[Dict]:
    """
    运行对比实验
    
    参数:
        uncompressed_dict: 未压缩词典
        compressed_dict: 压缩词典
    返回:
        实验结果列表
    """
    # 使用未压缩词典获取高频词来生成查询
    queries = design_comparison_queries(uncompressed_dict)
    results = []
    
    print("=" * 80)
    print("索引压缩前后检索效率对比实验")
    print("=" * 80)
    print()
    
    # 显示使用的高频词信息
    top_terms = get_top_frequency_terms(uncompressed_dict, top_n=15)
    print(f"使用的高频词（前15个，按文档频率降序）:")
    for i, term in enumerate(top_terms[:10], 1):
        doc_freq = uncompressed_dict[term][0]
        print(f"  {i}. {term} (文档频率: {doc_freq})")
    print()
    
    for idx, query in enumerate(queries, 1):
        print(f"查询 {idx}: {query}")
        print("-" * 80)
        
        # 在未压缩索引上执行
        result_uncompressed, time_uncompressed = execute_query_on_uncompressed(
            query, uncompressed_dict, INDEX_BIN_PATH
        )
        
        # 在压缩索引上执行
        result_compressed, time_compressed, decode_count = execute_query_on_compressed(
            query, compressed_dict, COMPRESSED_INDEX_PATH
        )
        
        # 计算性能指标
        if time_uncompressed > 0:
            speedup_percent = ((time_compressed - time_uncompressed) / time_uncompressed) * 100
        else:
            speedup_percent = 0
        
        # 验证结果一致性（确保都排序后比较）
        sorted_uncompressed = sorted(result_uncompressed)
        sorted_compressed = sorted(result_compressed)
        results_match = sorted_uncompressed == sorted_compressed
        
        # 输出结果
        print(f"未压缩索引查询时间:   {time_uncompressed:.6f}s")
        print(f"压缩索引查询时间:     {time_compressed:.6f}s")
        print(f"时间差异:             {speedup_percent:+.2f}%")
        print(f"VB+Gap解码次数:       {decode_count}")
        print(f"结果文档数量 (未压缩): {len(result_uncompressed)}")
        print(f"结果文档数量 (压缩):   {len(result_compressed)}")
        print(f"结果一致性:           {'✓ 一致' if results_match else '✗ 不一致'}")
        
        # 如果结果不一致，显示差异
        if not results_match:
            print(f"【调试信息】结果不一致！")
            print(f"  未压缩结果: {sorted_uncompressed[:20]}")
            print(f"  压缩结果:   {sorted_compressed[:20]}")
            
            # 找出差异
            set_uncompressed = set(result_uncompressed)
            set_compressed = set(result_compressed)
            only_in_uncompressed = set_uncompressed - set_compressed
            only_in_compressed = set_compressed - set_uncompressed
            
            if only_in_uncompressed:
                print(f"  仅在未压缩中: {sorted(list(only_in_uncompressed))[:10]}")
            if only_in_compressed:
                print(f"  仅在压缩中:   {sorted(list(only_in_compressed))[:10]}")
        
        print()
        
        # 记录结果
        results.append({
            'query': query,
            'result_count': len(result_uncompressed),
            'result_count_compressed': len(result_compressed),
            'time_uncompressed': time_uncompressed,
            'time_compressed': time_compressed,
            'decode_count': decode_count,
            'speedup_percent': speedup_percent,
            'results_match': results_match,
            'result_uncompressed': sorted_uncompressed[:20],
            'result_compressed': sorted_compressed[:20]
        })
    
    print("=" * 80)
    
    return results


def save_comparison_results(results: List[Dict], output_file: str):
    """
    保存对比实验结果到文件
    
    参数:
        results: 实验结果列表
        output_file: 输出文件路径
    """
    # 获取文件大小信息
    dict_size = get_file_size(DICT_BIN_PATH)
    index_size = get_file_size(INDEX_BIN_PATH)
    compressed_dict_size = get_file_size(COMPRESSED_DICT_PATH)
    compressed_index_size = get_file_size(COMPRESSED_INDEX_PATH)
    
    total_uncompressed = dict_size + index_size
    total_compressed = compressed_dict_size + compressed_index_size
    
    compression_ratio = (total_compressed / total_uncompressed * 100) if total_uncompressed > 0 else 0
    space_saved = total_uncompressed - total_compressed
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("索引压缩前后检索效率对比实验报告\n")
        f.write("=" * 80 + "\n\n")
        
        # 空间占用对比
        f.write("【一、存储空间对比】\n")
        f.write("-" * 80 + "\n")
        f.write(f"未压缩索引:\n")
        f.write(f"  - 词典文件:       {format_size(dict_size):>15} ({dict_size:,} 字节)\n")
        f.write(f"  - 倒排表文件:     {format_size(index_size):>15} ({index_size:,} 字节)\n")
        f.write(f"  - 总大小:         {format_size(total_uncompressed):>15} ({total_uncompressed:,} 字节)\n")
        f.write(f"\n")
        f.write(f"压缩索引:\n")
        f.write(f"  - 词典文件:       {format_size(compressed_dict_size):>15} ({compressed_dict_size:,} 字节)\n")
        f.write(f"  - 倒排表文件:     {format_size(compressed_index_size):>15} ({compressed_index_size:,} 字节)\n")
        f.write(f"  - 总大小:         {format_size(total_compressed):>15} ({total_compressed:,} 字节)\n")
        f.write(f"\n")
        f.write(f"压缩效果:\n")
        f.write(f"  - 压缩率:         {compression_ratio:.2f}%\n")
        f.write(f"  - 节省空间:       {format_size(space_saved)} ({space_saved:,} 字节)\n")
        f.write(f"\n\n")
        
        # 查询性能对比
        f.write("【二、查询性能对比】\n")
        f.write("-" * 80 + "\n\n")
        
        for idx, res in enumerate(results, 1):
            f.write(f"查询 {idx}: {res['query']}\n")
            f.write(f"  未压缩索引查询时间:   {res['time_uncompressed']:.6f}s\n")
            f.write(f"  压缩索引查询时间:     {res['time_compressed']:.6f}s\n")
            f.write(f"  时间差异:             {res['speedup_percent']:+.2f}%\n")
            f.write(f"  VB+Gap解码次数:       {res['decode_count']}\n")
            f.write(f"  结果文档数量 (未压缩): {res['result_count']}\n")
            f.write(f"  结果文档数量 (压缩):   {res['result_count_compressed']}\n")
            f.write(f"  结果一致性:           {'✓ 一致' if res['results_match'] else '✗ 不一致'}\n")
            
            # 如果结果不一致，添加详细信息
            if not res['results_match']:
                f.write(f"  【差异详情】\n")
                f.write(f"    未压缩结果示例: {res['result_uncompressed']}\n")
                f.write(f"    压缩结果示例:   {res['result_compressed']}\n")
            
            f.write("\n")
        
        # 统计汇总
        f.write("-" * 80 + "\n")
        f.write("【三、性能统计汇总】\n")
        f.write("-" * 80 + "\n")
        
        avg_time_uncompressed = sum(r['time_uncompressed'] for r in results) / len(results)
        avg_time_compressed = sum(r['time_compressed'] for r in results) / len(results)
        avg_speedup = sum(r['speedup_percent'] for r in results) / len(results)
        total_decodes = sum(r['decode_count'] for r in results)
        
        f.write(f"查询总数:               {len(results)}\n")
        f.write(f"平均未压缩查询时间:     {avg_time_uncompressed:.6f}s\n")
        f.write(f"平均压缩查询时间:       {avg_time_compressed:.6f}s\n")
        f.write(f"平均时间差异:           {avg_speedup:+.2f}%\n")
        f.write(f"总VB+Gap解码次数:       {total_decodes}\n")
        f.write("\n")
        
        # 分析结论
        f.write("=" * 80 + "\n")
        f.write("【四、实验结论】\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"1. 存储效率:\n")
        f.write(f"   压缩索引相比未压缩索引节省了 {100 - compression_ratio:.2f}% 的存储空间。\n\n")
        
        f.write(f"2. 查询效率:\n")
        if avg_speedup > 0:
            f.write(f"   压缩索引的平均查询时间比未压缩索引慢 {avg_speedup:.2f}%，\n")
            f.write(f"   这是由于需要额外的VB+Gap解码开销。\n\n")
        elif avg_speedup < 0:
            f.write(f"   压缩索引的平均查询时间比未压缩索引快 {-avg_speedup:.2f}%，\n")
            f.write(f"   这可能是由于压缩后数据更紧凑，提高了缓存命中率。\n\n")
        else:
            f.write(f"   两种索引的查询性能基本相当。\n\n")
        
        f.write(f"3. 综合评价:\n")
        f.write(f"   压缩索引通过牺牲少量查询性能（约{avg_speedup:.2f}%），换取了显著的存储空间节省。\n")
        f.write(f"   在存储资源受限或需要处理大规模数据的场景下，压缩索引是更优的选择。\n\n")
        
        f.write("=" * 80 + "\n")


# ==================== 主程序 ====================

def main():
    """
    主函数：执行索引压缩前后检索效率对比实验
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("\n" + "=" * 80)
    print("索引压缩前后检索效率对比实验")
    print("=" * 80)
    print()
    
    # 加载未压缩索引
    print("【步骤1】加载未压缩索引...")
    print(f"  词典文件: {DICT_BIN_PATH}")
    print(f"  倒排表文件: {INDEX_BIN_PATH}")
    uncompressed_dict = load_uncompressed_dictionary(DICT_BIN_PATH)
    print(f"  ✓ 加载完成，共 {len(uncompressed_dict)} 个词项")
    print()
    
    # 加载压缩索引
    print("【步骤2】加载压缩索引...")
    print(f"  词典文件: {COMPRESSED_DICT_PATH}")
    print(f"  倒排表文件: {COMPRESSED_INDEX_PATH}")
    compressed_dict, single_string, K = load_compressed_dictionary(COMPRESSED_DICT_PATH)
    print(f"  ✓ 加载完成，共 {len(compressed_dict)} 个词项")
    print(f"  压缩参数 K = {K}")
    print()
    
    # 显示存储空间对比
    print("【步骤3】存储空间对比...")
    dict_size = get_file_size(DICT_BIN_PATH)
    index_size = get_file_size(INDEX_BIN_PATH)
    compressed_dict_size = get_file_size(COMPRESSED_DICT_PATH)
    compressed_index_size = get_file_size(COMPRESSED_INDEX_PATH)
    
    total_uncompressed = dict_size + index_size
    total_compressed = compressed_dict_size + compressed_index_size
    
    print(f"  未压缩总大小: {format_size(total_uncompressed)}")
    print(f"  压缩总大小:   {format_size(total_compressed)}")
    print(f"  压缩率:       {total_compressed / total_uncompressed * 100:.2f}%")
    print(f"  节省空间:     {format_size(total_uncompressed - total_compressed)}")
    print()
    
    # 运行对比实验
    print("【步骤4】执行检索效率对比实验...")
    print()
    results = run_comparison_experiments(uncompressed_dict, compressed_dict)
    
    # 保存结果
    print("【步骤5】保存实验结果...")
    save_comparison_results(results, COMPARISON_OUTPUT_FILE)
    print(f"  ✓ 实验结果已保存到: {COMPARISON_OUTPUT_FILE}")
    print()
    
    print("=" * 80)
    print("实验完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()

