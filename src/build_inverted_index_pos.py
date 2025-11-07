"""
位置型倒排索引构建工具
用于对规范化词项构建带位置信息的倒排表（Positional Inverted Index）
采用三步流程：
1. 检索每篇文档，获得<词项，文档ID，位置>三元组，并写入临时索引
2. 对临时索引中的词项进行排序
3. 遍历临时索引，对于相同词项的文档ID和位置信息进行合并
输出格式：词项 总频率; 文档1:位置1,位置2,...; 文档2:位置1,位置2,...; ...
"""

import os
import shutil
import struct
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


# ============ 配置加载 ============
def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    """
    if config_path is None:
        # 默认配置文件路径
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config" / "inverted_index_pos_config.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 加载配置
CONFIG = load_config()

# ============ 配置项 ============
DEFAULT_INPUT_DIR = CONFIG['input_dir']  # 输入文件夹路径
DEFAULT_OUTPUT_DIR = CONFIG['output_dir']  # 输出文件夹路径
INVERTED_INDEX_FILE = CONFIG['files']['inverted_index']  # 倒排索引文件名（文本格式）
INVERTED_INDEX_BIN_FILE = CONFIG['files']['inverted_index_bin']  # 倒排索引二进制文件
INVERTED_INDEX_COMPRESSED_FILE = CONFIG['files']['inverted_index_compressed']  # VB压缩倒排索引文件
DICTIONARY_FILE = CONFIG['files']['dictionary']  # 词典文件名（文本格式）
DICTIONARY_BIN_FILE = CONFIG['files']['dictionary_bin']  # 词典文件名（二进制格式）
COMPRESSED_DICTIONARY_FILE = CONFIG['files']['compressed_dictionary']  # 压缩词典文件名

# 词典定长存储配置
TERM_MAX_LENGTH = CONFIG['storage']['term_max_length']  # 词项最大长度（字节）
DOC_FREQ_SIZE = CONFIG['storage']['doc_freq_size']  # 文档频率字段大小（字节）
POINTER_SIZE = CONFIG['storage']['pointer_size']  # 倒排表指针字段大小（字节）
ENTRY_SIZE = TERM_MAX_LENGTH + DOC_FREQ_SIZE + POINTER_SIZE  # 每个词项总开销

# 文档ID和位置存储参数
DOC_ID_SIZE = CONFIG['storage']['doc_id_size']  # 文档ID存储大小（字节）
POSITION_SIZE = CONFIG['storage']['position_size']  # 位置存储大小（字节）
FREQ_SIZE = CONFIG['storage']['freq_size']  # 词频存储大小（字节）
POSITION_MAX_VALUE = CONFIG['storage']['position_max_value']  # 位置最大值

# 压缩配置
K_INTERVAL = CONFIG['compression']['k_interval']  # K-间隔指针压缩参数
TERM_BYTES = CONFIG['compression']['term_bytes']  # 词项前缀字节数
FREQ_BYTES = CONFIG['compression']['freq_bytes']  # 频率字段字节数
PTR_BYTES = CONFIG['compression']['ptr_bytes']  # 指针字段字节数
TERM_PTR_BYTES = CONFIG['compression']['term_ptr_bytes']  # 词项指针字节数
TERM_LEN_BYTES = CONFIG['compression']['term_len_bytes']  # 词项长度字节数


# ============ VB编码核心函数 ============

def vb_encode_number(n: int) -> bytes:
    """
    对单个整数进行VB编码
    
    原理：
    - 每个字节的低7位存储数据
    - 最高位为标志位：0表示后续还有字节，1表示这是最后一个字节
    
    参数:
        n: 要编码的非负整数
        
    返回:
        VB编码后的字节序列
    
    示例:
        5 -> b'\x85' (10000101)
        130 -> b'\x01\x82' (00000001 10000010)
    """
    if n == 0:
        return bytes([128])  # 10000000
    
    bytes_list = []
    while n > 0:
        bytes_list.insert(0, n % 128)  # 取低7位
        n //= 128
    
    # 最后一个字节的最高位设为1
    bytes_list[-1] += 128
    
    return bytes(bytes_list)


def vb_encode_list(numbers: List[int]) -> bytes:
    """
    对整数列表进行VB编码
    
    参数:
        numbers: 整数列表
        
    返回:
        VB编码后的字节序列
    """
    result = b''
    for num in numbers:
        result += vb_encode_number(num)
    return result


def vb_decode(byte_stream: bytes) -> List[int]:
    """
    解码VB编码的字节流
    
    参数:
        byte_stream: VB编码的字节序列
        
    返回:
        解码后的整数列表
    """
    numbers = []
    current = 0
    
    for byte in byte_stream:
        if byte < 128:
            # 最高位为0，继续累积
            current = current * 128 + byte
        else:
            # 最高位为1，这是最后一个字节
            current = current * 128 + (byte - 128)
            numbers.append(current)
            current = 0
    
    return numbers


# ============ Gap编码函数 ============

def encode_gaps(numbers: List[int]) -> List[int]:
    """
    将绝对值列表转换为Gap编码（差值编码）
    
    参数:
        numbers: 有序的整数列表
        
    返回:
        Gap编码后的列表（第一个值保持不变，后续为差值）
    
    示例:
        [5, 10, 15, 23] -> [5, 5, 5, 8]
    """
    if not numbers:
        return []
    
    gaps = [numbers[0]]
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])
    
    return gaps


def decode_gaps(gaps: List[int]) -> List[int]:
    """
    将Gap编码还原为绝对值
    
    参数:
        gaps: Gap编码的列表
        
    返回:
        绝对值列表
    """
    if not gaps:
        return []
    
    numbers = [gaps[0]]
    for i in range(1, len(gaps)):
        numbers.append(numbers[i-1] + gaps[i])
    
    return numbers


# ============ 文档ID转换函数 ============

def doc_id_to_int(doc_id: str) -> int:
    """
    将文档ID字符串转换为整数
    
    参数:
        doc_id: 文档ID字符串（如 "10000527"）
        
    返回:
        整数形式的文档ID
    """
    try:
        return int(doc_id)
    except ValueError:
        # 如果无法转换，使用哈希值
        return hash(doc_id) & 0x7FFFFFFF


def int_to_doc_id(doc_int: int) -> str:
    """
    将整数转换回文档ID字符串
    
    参数:
        doc_int: 整数形式的文档ID
        
    返回:
        文档ID字符串
    """
    return str(doc_int)


# ============ 原有函数 ============

def clear_output_directory(directory: str):
    """
    清空输出目录，移除所有旧文件
    
    参数:
        directory: 目录路径
    """
    if os.path.exists(directory):
        print(f"清空输出目录: {directory}")
        # 删除目录及其所有内容
        shutil.rmtree(directory)
        print(f"✓ 已清除旧文件\n")
    # 重新创建空目录
    os.makedirs(directory, exist_ok=True)


def get_text_files_from_directory(directory: str) -> List[str]:
    """
    从指定目录获取所有 .txt 文件（排除 mwe_phrases.txt）
    
    参数:
        directory: 目录路径
        
    返回:
        文本文件路径列表
    """
    txt_files = []
    
    if not os.path.exists(directory):
        print(f"目录不存在: {directory}")
        return txt_files
    
    for file_name in os.listdir(directory):
        # 排除短语词典文件
        if file_name.lower().endswith('.txt') and file_name != 'mwe_phrases.txt':
            file_path = os.path.join(directory, file_name)
            txt_files.append(file_path)
    
    return txt_files


def read_tokens_from_file(file_path: str) -> List[str]:
    """
    从文件中读取所有 token（每行一个token）
    
    参数:
        file_path: 文件路径
        
    返回:
        token 列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            tokens = [line.strip() for line in f if line.strip()]
        return tokens
    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        return []


def extract_document_id(file_path: str) -> str:
    """
    从文件路径中提取文档 ID（即文件名不含扩展名）
    
    参数:
        file_path: 文件路径
        
    返回:
        文档 ID
    """
    file_name = os.path.basename(file_path)
    doc_id = os.path.splitext(file_name)[0]
    return doc_id


def build_temp_index(input_dir: str, output_dir: str) -> Dict[str, Dict[str, List[int]]]:
    """
    第一步：检索每篇文档，在内存中构建位置型倒排索引
    
    参数:
        input_dir: 输入目录路径
        output_dir: 输出目录路径
        
    返回:
        倒排索引字典 {词项: {文档ID: [位置列表]}}
    """
    print("=" * 60)
    print("第一步：构建内存倒排索引（含位置信息）")
    print("=" * 60)
    print()
    
    # 获取所有文档文件
    file_paths = get_text_files_from_directory(input_dir)
    
    if not file_paths:
        print(f"错误：在目录 {input_dir} 中未找到任何文档文件")
        return {}
    
    print(f"找到 {len(file_paths)} 篇文档")
    
    # 在内存中构建倒排索引：{词项: {文档ID: [位置列表]}}
    inverted_index = defaultdict(lambda: defaultdict(list))
    total_pairs = 0
    
    for idx, file_path in enumerate(file_paths, 1):
        doc_id = extract_document_id(file_path)
        tokens = read_tokens_from_file(file_path)
        
        if not tokens:
            print(f"[{idx}/{len(file_paths)}] 文档 {doc_id}: 无 token，跳过")
            continue
        
        # 记录每个词项在文档中的位置（位置从1开始计数）
        for position, token in enumerate(tokens, 1):
            inverted_index[token][doc_id].append(position)
            total_pairs += 1
        
        print(f"[{idx}/{len(file_paths)}] 文档 {doc_id}: 提取 {len(tokens)} 个词项")
    
    print(f"\n✓ 内存索引构建完成")
    print(f"  - 总文档数: {len(file_paths)}")
    print(f"  - 唯一词项数: {len(inverted_index)}")
    print(f"  - 总词项出现次数: {total_pairs}\n")
    
    return dict(inverted_index)



def write_text_inverted_index(inverted_index: Dict[str, Dict[str, List[int]]], output_dir: str) -> Tuple[str, Dict]:
    """
    第二步：将内存中的倒排索引写入文本文件
    
    参数:
        inverted_index: 内存中的倒排索引 {词项: {文档ID: [位置列表]}}
        output_dir: 输出目录路径
        
    返回:
        (倒排索引文件路径, 统计信息字典)
    """
    print("=" * 60)
    print("第二步：写入文本倒排索引")
    print("=" * 60)
    print()
    
    # 写入最终的倒排索引
    inverted_index_path = os.path.join(output_dir, INVERTED_INDEX_FILE)
    
    print("写入倒排索引文件...")
    total_frequency = 0
    with open(inverted_index_path, 'w', encoding='utf-8') as f:
        for term in sorted(inverted_index.keys()):
            doc_positions = inverted_index[term]
            
            # 计算词项的总频率（所有文档中出现的总次数）
            term_frequency = sum(len(positions) for positions in doc_positions.values())
            total_frequency += term_frequency
            
            # 构建倒排列表：文档ID:位置1,位置2,...
            postings = []
            for doc_id in sorted(doc_positions.keys()):
                positions = sorted(doc_positions[doc_id])
                positions_str = ','.join(map(str, positions))
                postings.append(f"{doc_id}:{positions_str}")
            
            # 格式：词项 总频率; 文档1:位置1,位置2,...; 文档2:位置1,位置2,...; ...
            postings_str = '; '.join(postings)
            f.write(f"{term} {term_frequency}; {postings_str}\n")
    
    # 统计信息
    total_postings = sum(len(doc_positions) for doc_positions in inverted_index.values())
    avg_postings = total_postings / len(inverted_index) if inverted_index else 0
    avg_frequency = total_frequency / len(inverted_index) if inverted_index else 0
    
    stats = {
        'unique_terms': len(inverted_index),
        'total_postings': total_postings,
        'total_frequency': total_frequency,
        'avg_postings_per_term': avg_postings,
        'avg_frequency_per_term': avg_frequency
    }
    
    print(f"\n✓ 文本倒排索引写入完成")
    print(f"  - 唯一词项数: {stats['unique_terms']}")
    print(f"  - 总倒排项数（文档数）: {stats['total_postings']}")
    print(f"  - 总词频（所有出现次数）: {stats['total_frequency']}")
    print(f"  - 平均每个词项的文档数: {stats['avg_postings_per_term']:.2f}")
    print(f"  - 平均每个词项的总频率: {stats['avg_frequency_per_term']:.2f}")
    print(f"  - 保存路径: {inverted_index_path}\n")
    
    return inverted_index_path, stats


def generate_binary_index_and_dictionary(inverted_index: Dict[str, Dict[str, List[int]]], output_dir: str) -> Tuple[str, str, str]:
    """
    生成二进制格式的倒排索引和定长词典
    
    二进制倒排表格式（每个词项）：
    [文档数量(4字节)] + 
      [文档ID1(4字节)] + [该文档词频(4字节)] + [位置1(2字节)] + [位置2(2字节)] + ... +
      [文档ID2(4字节)] + [该文档词频(4字节)] + [位置1(2字节)] + [位置2(2字节)] + ...
    
    参数:
        inverted_index: 内存中的倒排索引 {词项: {文档ID: [位置列表]}}
        output_dir: 输出目录路径
        
    返回:
        (二进制索引文件路径, 二进制词典文件路径, 文本词典文件路径)
    """
    print("=" * 60)
    print("第三步：生成二进制倒排索引和词典")
    print("=" * 60)
    print()
    
    # 生成二进制倒排索引和词典
    bin_index_path = os.path.join(output_dir, INVERTED_INDEX_BIN_FILE)
    dict_bin_path = os.path.join(output_dir, DICTIONARY_BIN_FILE)
    dict_text_path = os.path.join(output_dir, DICTIONARY_FILE)
    
    print(f"写入二进制文件...")
    
    total_positions = 0
    overflow_positions = 0
    truncated_terms = 0
    
    with open(bin_index_path, 'wb') as bin_f, \
         open(dict_bin_path, 'wb') as dict_bin_f, \
         open(dict_text_path, 'w', encoding='utf-8') as dict_text_f:
        
        # 写入文本词典头部
        dict_text_f.write("=" * 80 + "\n")
        dict_text_f.write("位置型倒排索引词典（定长存储）\n")
        dict_text_f.write("=" * 80 + "\n\n")
        dict_text_f.write(f"词项固定长度: {TERM_MAX_LENGTH} 字节\n")
        dict_text_f.write(f"文档频率字段: {DOC_FREQ_SIZE} 字节\n")
        dict_text_f.write(f"倒排表指针: {POINTER_SIZE} 字节\n")
        dict_text_f.write(f"每词项开销: {ENTRY_SIZE} 字节\n\n")
        dict_text_f.write("=" * 80 + "\n")
        dict_text_f.write(f"{'词项':<30}{'文档数':<10}{'总频率':<10}{'倒排表指针':<15}\n")
        dict_text_f.write("=" * 80 + "\n")
        
        # 按词项顺序处理
        for term in sorted(inverted_index.keys()):
            doc_positions = inverted_index[term]
            
            # 记录当前偏移量
            offset = bin_f.tell()
            
            # 计算总频率
            total_freq = sum(len(positions) for positions in doc_positions.values())
            
            # 写入文档数量（4字节）
            doc_count = len(doc_positions)
            bin_f.write(struct.pack('I', doc_count))
            
            # 写入每个文档的倒排列表
            for doc_id in sorted(doc_positions.keys()):
                positions = doc_positions[doc_id]
                
                # 转换文档ID为整数
                try:
                    doc_id_int = int(doc_id)
                except ValueError:
                    doc_id_int = hash(doc_id) & 0xFFFFFFFF
                
                # 写入文档ID（4字节）
                bin_f.write(struct.pack('I', doc_id_int))
                
                # 写入该文档中的词频（4字节）
                freq = len(positions)
                bin_f.write(struct.pack('I', freq))
                
                # 写入位置列表（每个位置2字节）
                for pos in positions:
                    if pos > POSITION_MAX_VALUE:
                        pos = POSITION_MAX_VALUE  # 截断到最大值
                        overflow_positions += 1
                    bin_f.write(struct.pack('H', pos))  # 'H' = unsigned short (2 bytes)
                    total_positions += 1
            
            # === 写入二进制词典（定长存储） ===
            term_bytes = term.encode('utf-8')
            if len(term_bytes) > TERM_MAX_LENGTH:
                term_bytes = term_bytes[:TERM_MAX_LENGTH]
                truncated_terms += 1
            else:
                term_bytes = term_bytes.ljust(TERM_MAX_LENGTH, b'\x00')
            
            # 写入：词项(20字节) + 文档频率(4字节) + 偏移量(4字节)
            dict_bin_f.write(term_bytes)
            dict_bin_f.write(struct.pack('I', doc_count))
            dict_bin_f.write(struct.pack('I', offset))
            
            # 写入文本词典
            display_term = term if len(term) <= 28 else term[:25] + "..."
            dict_text_f.write(f"{display_term:<30}{doc_count:<10}{total_freq:<10}{offset:<15}\n")
        
        dict_text_f.write("=" * 80 + "\n")
        dict_text_f.write(f"总词项数: {len(inverted_index)}\n")
        dict_text_f.write("=" * 80 + "\n")
    
    print(f"✓ 二进制文件生成完成")
    
    # 计算文件大小
    bin_index_size = os.path.getsize(bin_index_path)
    dict_bin_size = os.path.getsize(dict_bin_path)
    
    print(f"\n【二进制倒排索引统计】")
    print(f"  - 词项数量: {len(inverted_index)}")
    print(f"  - 总位置数: {total_positions}")
    print(f"  - 文档ID大小: {DOC_ID_SIZE} 字节")
    print(f"  - 位置大小: {POSITION_SIZE} 字节（最大值: {POSITION_MAX_VALUE:,}）")
    print(f"  - 词频大小: {FREQ_SIZE} 字节")
    print(f"  - 二进制索引文件大小: {bin_index_size:,} 字节 ({bin_index_size / 1024:.2f} KB)")
    if overflow_positions > 0:
        print(f"  - 警告: {overflow_positions} 个位置超过{POSITION_MAX_VALUE}被截断")
    
    print(f"\n【二进制词典统计】")
    print(f"  - 词项固定长度: {TERM_MAX_LENGTH} 字节")
    print(f"  - 每词项开销: {ENTRY_SIZE} 字节")
    print(f"  - 词典文件大小: {dict_bin_size:,} 字节 ({dict_bin_size / 1024:.2f} KB)")
    if truncated_terms > 0:
        print(f"  - 警告: {truncated_terms} 个词项被截断")
    print()
    
    return bin_index_path, dict_bin_path, dict_text_path


def generate_vb_compressed_index(inverted_index: Dict[str, Dict[str, List[int]]], 
                                 inverted_index_path: str, output_dir: str) -> Tuple[str, Dict]:
    """
    生成VB+Gap编码压缩的倒排索引
    
    参数:
        inverted_index: 内存中的倒排索引 {词项: {文档ID: [位置列表]}}
        inverted_index_path: 文本格式倒排索引文件路径（用于获取文件大小对比）
        output_dir: 输出目录路径
        
    返回:
        (压缩索引文件路径, 压缩统计信息字典)
    """
    print("=" * 60)
    print("第四步：生成VB压缩倒排索引")
    print("=" * 60)
    print()
    
    # 生成压缩文件
    compressed_index_path = os.path.join(output_dir, INVERTED_INDEX_COMPRESSED_FILE)
    
    print(f"正在使用VB+Gap编码压缩...")
    
    # 统计信息
    stats = {
        'total_terms': 0,
        'total_postings': 0,
        'total_positions': 0,
        'uncompressed_size': 0,  # 未压缩大小（整数个数 × 4字节）
        'compressed_size': 0,     # VB编码后的大小
        'compression_ratio': 0.0,
        'text_size': 0            # 文本格式大小
    }
    
    with open(compressed_index_path, 'wb') as bin_file:
        
        for term in sorted(inverted_index.keys()):
            doc_positions = inverted_index[term]
            
            # 准备编码数据
            doc_ids = sorted([doc_id_to_int(doc_id) for doc_id in doc_positions.keys()])
            total_frequency = 0
            
            # 编码格式：
            # 1. 文档数量（VB编码）
            # 2. 对每个文档：
            #    - 文档ID（Gap编码 + VB编码）
            #    - 位置数量（VB编码）
            #    - 位置列表（Gap编码 + VB编码）
            
            encoded_data = vb_encode_number(len(doc_ids))
            
            # 文档ID使用Gap编码
            doc_id_gaps = encode_gaps(doc_ids)
            
            for doc_id_int, doc_id_gap in zip(doc_ids, doc_id_gaps):
                # 获取位置列表
                positions = None
                for orig_doc_id, pos_list in doc_positions.items():
                    if doc_id_to_int(orig_doc_id) == doc_id_int:
                        positions = sorted(pos_list)
                        break
                
                if positions is None:
                    continue
                
                total_frequency += len(positions)
                
                # 编码文档ID（Gap）
                encoded_data += vb_encode_number(doc_id_gap)
                
                # 编码位置数量
                encoded_data += vb_encode_number(len(positions))
                
                # 编码位置列表（Gap编码）
                position_gaps = encode_gaps(positions)
                encoded_data += vb_encode_list(position_gaps)
                
                # 统计未压缩大小（文档ID + 位置数量 + 所有位置）
                stats['uncompressed_size'] += 4 * (1 + 1 + len(positions))
            
            # 写入二进制数据
            bin_file.write(encoded_data)
            
            # 记录编码后的长度
            length = len(encoded_data)
            stats['compressed_size'] += length
            
            # 更新统计
            stats['total_terms'] += 1
            stats['total_postings'] += len(doc_ids)
            stats['total_positions'] += total_frequency
            
            if stats['total_terms'] % 100 == 0:
                print(f"  已处理 {stats['total_terms']} 个词项...")
    
    # 计算压缩率
    if stats['uncompressed_size'] > 0:
        stats['compression_ratio'] = stats['compressed_size'] / stats['uncompressed_size']
    
    # 获取文本索引文件大小用于对比
    stats['text_size'] = os.path.getsize(inverted_index_path)
    
    print(f"\n✓ VB压缩完成")
    print(f"  - 压缩索引: {compressed_index_path}")
    print(f"  - 未压缩大小: {stats['uncompressed_size']:,} 字节")
    print(f"  - 压缩后大小: {stats['compressed_size']:,} 字节")
    print(f"  - 压缩率（相对整数）: {stats['compression_ratio']:.2%}")
    print(f"  - 文本格式大小: {stats['text_size']:,} 字节")
    print(f"  - 压缩率（相对文本）: {stats['compressed_size']/stats['text_size']:.2%}\n")
    
    return compressed_index_path, stats


def generate_compressed_dictionary(dict_text_path: str, compressed_index_path: str, output_dir: str) -> str:
    """
    生成压缩词典（使用K-间隔指针压缩方法）
    
    参数:
        dict_text_path: 文本格式词典文件路径
        compressed_index_path: VB+Gap压缩倒排表文件路径
        output_dir: 输出目录路径
        
    返回:
        压缩词典文件路径
    """
    print("=" * 60)
    print("第五步：生成压缩词典（K-间隔指针压缩）")
    print("=" * 60)
    print()
    
    # 使用全局配置
    K = K_INTERVAL
    TERM_ENTRY_WITH_PTR_SIZE = TERM_BYTES + TERM_PTR_BYTES + FREQ_BYTES + PTR_BYTES  # 带指针
    TERM_ENTRY_WITH_LEN_SIZE = TERM_BYTES + TERM_LEN_BYTES + FREQ_BYTES + PTR_BYTES  # 带长度
    
    # 1. 从文本词典加载数据
    print("读取文本格式词典...")
    dictionary = []
    with open(dict_text_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            # 跳过空行、注释行、分隔线、标题行
            if (not line or line.startswith('=') or 
                '词项' in line or '总词项数' in line or 
                '词项固定长度' in line or '文档频率字段' in line or
                '倒排表指针' in line or '每词项开销' in line or
                '位置型倒排索引词典' in line):
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                term = parts[0]
                try:
                    freq = int(parts[1])   # 文档频率
                    ptr = int(parts[3])    # 倒排表指针（基于二进制倒排表）
                    dictionary.append((term, freq, ptr))
                except ValueError:
                    continue
    
    print(f"读取了 {len(dictionary)} 个词项")
    
    # 2. 重新计算VB压缩倒排表的指针
    print("\n计算VB+Gap压缩倒排表的词项偏移...")
    new_dictionary = []
    current_offset = 0
    
    with open(compressed_index_path, 'rb') as f:
        for term, doc_freq, old_ptr in dictionary:
            # 记录当前词项在压缩倒排表中的偏移
            new_dictionary.append((term, doc_freq, current_offset))
            
            # 计算这个词项的VB编码posting list长度
            f.seek(current_offset)
            byte_count = 0
            num_decoded = 0
            while num_decoded < doc_freq:
                byte_data = f.read(1)
                if not byte_data:
                    break
                byte_count += 1
                # VB编码：最高位为1表示一个整数的结束
                if byte_data[0] >= 128:
                    num_decoded += 1
            
            current_offset += byte_count
    
    dictionary = new_dictionary
    print(f"计算完成，共处理 {len(dictionary)} 个词项")
    
    # 3. 构建单一字符串和元数据
    print("\n构建压缩词典数据结构...")
    single_string = ""
    term_entries = []
    
    for idx, (term, freq, ptr) in enumerate(dictionary):
        term_bytes = term.encode('utf-8')
        term_byte_len = len(term_bytes)
        term_start_pos = len(single_string)
        single_string += term
        term_len = len(term)
        
        # 构建8字节前缀
        if term_byte_len <= TERM_BYTES:
            term_prefix = term_bytes + b'\x00' * (TERM_BYTES - term_byte_len)
        else:
            term_prefix = term_bytes[:TERM_BYTES]
        
        # 判断是否存储指针
        store_pointer = ((idx + 1) % K == 0) or (idx == len(dictionary) - 1)
        
        term_entries.append({
            'term_prefix': term_prefix,
            'term_ptr': term_start_pos,
            'term_len': term_len,
            'freq': freq,
            'ptr': ptr,
            'store_pointer': store_pointer
        })
    
    # 4. 写入压缩词典文件
    compressed_dict_path = os.path.join(output_dir, COMPRESSED_DICTIONARY_FILE)
    single_string_bytes = single_string.encode('utf-8')
    
    print(f"\n写入压缩词典: {compressed_dict_path}")
    with open(compressed_dict_path, 'wb') as f:
        # 写入文件头
        f.write(struct.pack('I', len(single_string_bytes)))  # 单一字符串长度
        f.write(single_string_bytes)                         # 单一字符串内容
        f.write(struct.pack('I', len(dictionary)))           # 词项数量
        f.write(struct.pack('B', K))                         # K值
        
        # 写入每个词项的元数据
        for entry in term_entries:
            f.write(entry['term_prefix'])  # 8字节前缀
            
            if entry['store_pointer']:
                # 存储指针（3字节）
                term_ptr_bytes = entry['term_ptr'].to_bytes(TERM_PTR_BYTES, byteorder='little')
                f.write(term_ptr_bytes)
            else:
                # 存储长度（1字节）
                f.write(struct.pack('B', entry['term_len']))
            
            f.write(struct.pack('I', entry['freq']))  # 4字节频率
            f.write(struct.pack('I', entry['ptr']))   # 4字节倒排表指针
    
    # 5. 统计信息
    single_string_size = len(single_string_bytes)
    num_with_ptr = sum(1 for e in term_entries if e['store_pointer'])
    num_with_len = len(dictionary) - num_with_ptr
    metadata_size = num_with_ptr * TERM_ENTRY_WITH_PTR_SIZE + num_with_len * TERM_ENTRY_WITH_LEN_SIZE
    total_size = single_string_size + metadata_size + 9  # +9字节文件头
    actual_size = os.path.getsize(compressed_dict_path)
    
    print(f"\n✓ 压缩词典生成完成")
    print(f"\n【压缩词典统计】")
    print(f"  - 词项总数: {len(dictionary):,}")
    print(f"  - 压缩参数 K: {K}")
    print(f"  - 存储指针的词项: {num_with_ptr:,} 个 ({TERM_ENTRY_WITH_PTR_SIZE}字节/个)")
    print(f"  - 存储长度的词项: {num_with_len:,} 个 ({TERM_ENTRY_WITH_LEN_SIZE}字节/个)")
    print(f"  - 单一字符串块: {single_string_size:,} 字节")
    print(f"  - 元数据块: {metadata_size:,} 字节")
    print(f"  - 文件大小: {actual_size:,} 字节 ({actual_size/1024:.2f} KB)")
    print(f"  - 指向倒排表: {os.path.basename(compressed_index_path)} (VB+Gap压缩)")
    
    # 计算压缩率
    uncompressed_metadata = len(dictionary) * TERM_ENTRY_WITH_PTR_SIZE
    saved = uncompressed_metadata - metadata_size
    print(f"  - 元数据节省: {saved:,} 字节 ({saved/uncompressed_metadata*100:.2f}%)")
    print()
    
    return compressed_dict_path


def display_sample_index(inverted_index_path: str, sample_size: int = 10):
    """
    在控制台显示部分倒排索引样例
    
    参数:
        inverted_index_path: 倒排索引文件路径
        sample_size: 显示的样例数量
    """
    print("=" * 60)
    print(f"位置型倒排索引样例（前 {sample_size} 个词项）")
    print("=" * 60)
    print()
    
    with open(inverted_index_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            line = line.strip()
            if line:
                # 格式：词项 频率; 文档1:位置...; 文档2:位置...
                if ' ' in line and '; ' in line:
                    parts = line.split(' ', 1)
                    if len(parts) == 2:
                        term = parts[0]
                        rest = parts[1]
                        
                        freq_and_docs = rest.split('; ', 1)
                        if len(freq_and_docs) >= 1:
                            frequency = int(freq_and_docs[0])
                            
                            print(f"词项: {term}")
                            print(f"  总频率: {frequency}")
                            
                            if len(freq_and_docs) == 2:
                                doc_postings = freq_and_docs[1].split('; ')
                                print(f"  文档数: {len(doc_postings)}")
                                print(f"  倒排列表:")
                                
                                # 显示前3个文档的详细信息
                                for j, posting in enumerate(doc_postings[:3]):
                                    if ':' in posting:
                                        doc_id, positions = posting.split(':', 1)
                                        pos_list = positions.split(',')
                                        print(f"    - 文档 {doc_id}: 位置 {positions} (出现{len(pos_list)}次)")
                                
                                if len(doc_postings) > 3:
                                    print(f"    ... 还有 {len(doc_postings) - 3} 个文档")
                            print()


def main():
    """
    主入口函数
    """
    print("\n" + "=" * 60)
    print("位置型倒排索引构建工具")
    print("=" * 60 + "\n")
    
    # 清空输出目录
    clear_output_directory(DEFAULT_OUTPUT_DIR)
    
    # 第一步：在内存中构建倒排索引（含位置信息）
    inverted_index = build_temp_index(DEFAULT_INPUT_DIR, DEFAULT_OUTPUT_DIR)
    if not inverted_index:
        print("错误：倒排索引构建失败")
        return
    
    # 第二步：写入文本倒排索引
    inverted_index_path, stats = write_text_inverted_index(inverted_index, DEFAULT_OUTPUT_DIR)
    if not inverted_index_path:
        print("错误：倒排索引写入失败")
        return
    
    # 第三步：生成二进制倒排索引和词典
    bin_index_path, dict_bin_path, dict_text_path = generate_binary_index_and_dictionary(inverted_index, DEFAULT_OUTPUT_DIR)
    
    # 第四步：生成VB压缩倒排索引
    compressed_index_path, vb_stats = generate_vb_compressed_index(inverted_index, inverted_index_path, DEFAULT_OUTPUT_DIR)
    
    # 第五步：生成压缩词典
    compressed_dict_path = generate_compressed_dictionary(dict_text_path, compressed_index_path, DEFAULT_OUTPUT_DIR)
    
    # 显示样例
    display_sample_index(inverted_index_path, sample_size=10)
    
    print("=" * 60)
    print("位置型倒排索引构建完成！")
    print("=" * 60)
    print(f"\n输出格式说明：")
    print(f"  文本格式：词项 总频率; 文档ID1:位置1,位置2,...; 文档ID2:位置1,位置2,...; ...")
    print(f"  二进制格式：定长词典 + 二进制倒排表（位置用2字节存储）")
    print(f"  VB压缩格式：VB+Gap编码（文档ID和位置差值编码，大幅压缩）")
    print(f"\n输出文件：")
    print(f"  - 倒排索引（文本）: {os.path.join(DEFAULT_OUTPUT_DIR, INVERTED_INDEX_FILE)}")
    print(f"  - 倒排索引（二进制）: {os.path.join(DEFAULT_OUTPUT_DIR, INVERTED_INDEX_BIN_FILE)}")
    print(f"  - 倒排索引（VB压缩）: {os.path.join(DEFAULT_OUTPUT_DIR, INVERTED_INDEX_COMPRESSED_FILE)}")
    print(f"  - 词典（文本）: {os.path.join(DEFAULT_OUTPUT_DIR, DICTIONARY_FILE)}")
    print(f"  - 词典（二进制）: {os.path.join(DEFAULT_OUTPUT_DIR, DICTIONARY_BIN_FILE)}")
    print(f"  - 词典（压缩，K-间隔指针）: {compressed_dict_path}")
    
    print(f"\n【基本统计】")
    print(f"  - 唯一词项数: {stats['unique_terms']:,}")
    print(f"  - 总倒排项数（文档数）: {stats['total_postings']:,}")
    print(f"  - 总词频（所有出现次数）: {stats['total_frequency']:,}")
    print(f"  - 平均每个词项的文档数: {stats['avg_postings_per_term']:.2f}")
    print(f"  - 平均每个词项的总频率: {stats['avg_frequency_per_term']:.2f}")
    
    print(f"\n【二进制存储格式说明】")
    print(f"  词典（定长存储）：")
    print(f"    - 词项：{TERM_MAX_LENGTH} 字节")
    print(f"    - 文档频率：{DOC_FREQ_SIZE} 字节")
    print(f"    - 倒排表指针：{POINTER_SIZE} 字节")
    print(f"    - 每词项总计：{ENTRY_SIZE} 字节")
    print(f"  倒排表（每个词项）：")
    print(f"    - 文档数量：{DOC_ID_SIZE} 字节")
    print(f"    - 每个文档：文档ID({DOC_ID_SIZE}字节) + 词频({FREQ_SIZE}字节) + 位置列表(每个{POSITION_SIZE}字节)")
    
    print(f"\n【VB压缩效果总结】")
    print(f"  - 未压缩大小（32位整数）: {vb_stats['uncompressed_size']:,} 字节 ({vb_stats['uncompressed_size']/1024/1024:.2f} MB)")
    print(f"  - VB编码后大小: {vb_stats['compressed_size']:,} 字节 ({vb_stats['compressed_size']/1024/1024:.2f} MB)")
    print(f"  - 压缩率（相对整数）: {vb_stats['compression_ratio']:.2%}")
    print(f"  - 压缩率（相对文本）: {vb_stats['compressed_size']/vb_stats['text_size']:.2%}")
    print(f"  - 节省空间: {vb_stats['text_size'] - vb_stats['compressed_size']:,} 字节 ({(vb_stats['text_size'] - vb_stats['compressed_size'])/1024/1024:.2f} MB)")
    print()


if __name__ == "__main__":
    main()

