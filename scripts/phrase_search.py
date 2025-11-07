"""
短语检索系统（Phrase Query）
利用带位置信息的倒排索引实现精确短语匹配
"""

import time
import struct
import json
import os
from pathlib import Path
from typing import List, Dict, Set, Tuple
from collections import defaultdict


# ==================== 配置加载 ====================
def load_config(config_path: str = None) -> dict:
    """
    加载配置文件
    """
    if config_path is None:
        # 默认配置文件路径
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config" / "phrase_search_config.json"
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


# 加载配置
CONFIG = load_config()

# 获取项目根目录
PROJECT_ROOT = Path(__file__).parent.parent

# ==================== 配置项 ====================
DICT_BIN_PATH = str(PROJECT_ROOT / CONFIG['input']['dictionary_bin'])
INDEX_BIN_PATH = str(PROJECT_ROOT / CONFIG['input']['inverted_index_bin'])
OUTPUT_DIR = str(PROJECT_ROOT / CONFIG['output']['output_dir'])
OUTPUT_FILE = str(PROJECT_ROOT / CONFIG['output']['output_dir'] / CONFIG['output']['results_file'])

# 二进制文件格式常量
TERM_MAX_LENGTH = CONFIG['storage']['term_max_length']  # 词项最大长度（字节）
DOC_FREQ_SIZE = CONFIG['storage']['doc_freq_size']  # 文档频率字段大小（字节）
POINTER_SIZE = CONFIG['storage']['pointer_size']  # 倒排表指针字段大小（字节）
ENTRY_SIZE = TERM_MAX_LENGTH + DOC_FREQ_SIZE + POINTER_SIZE  # 每个词项总开销
DOC_ID_SIZE = CONFIG['storage']['doc_id_size']  # 文档ID存储大小（字节）
POSITION_SIZE = CONFIG['storage']['position_size']  # 位置信息存储大小（字节）


# ==================== 模块 1：数据加载 ====================

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


def load_positional_posting_list(bin_index_path: str, pointer: int, doc_freq: int) -> Dict[int, List[int]]:
    """
    从二进制倒排索引文件中读取带位置信息的posting list
    
    参数:
        bin_index_path: 二进制倒排索引文件路径
        pointer: 倒排表指针（字节偏移）
        doc_freq: 文档频率（包含该词的文档数）
    返回:
        字典，键为文档ID，值为位置列表
    """
    doc_positions = {}
    
    with open(bin_index_path, 'rb') as f:
        # 定位到倒排表起始位置
        f.seek(pointer)

        # 先读取文档数量（4字节，构建器写入的doc_count）
        doc_count_bytes = f.read(4)
        if not doc_count_bytes or len(doc_count_bytes) < 4:
            return doc_positions
        actual_doc_count = struct.unpack('I', doc_count_bytes)[0]

        # 读取actual_doc_count个文档的信息
        for _ in range(actual_doc_count):
            # 读取文档ID（4字节）
            doc_id_bytes = f.read(DOC_ID_SIZE)
            if len(doc_id_bytes) < DOC_ID_SIZE:
                break
            doc_id = struct.unpack('I', doc_id_bytes)[0]

            # 读取该文档中的词频/位置数量（4字节）
            freq_bytes = f.read(4)
            if len(freq_bytes) < 4:
                break
            pos_count = struct.unpack('I', freq_bytes)[0]

            # 读取所有位置（每个位置2字节，unsigned short）
            positions = []
            for _ in range(pos_count):
                pos_bytes = f.read(POSITION_SIZE)
                if len(pos_bytes) < POSITION_SIZE:
                    break
                position = struct.unpack('H', pos_bytes)[0]
                positions.append(position)

            doc_positions[doc_id] = sorted(positions)
    
    return doc_positions


def load_positional_index_from_binary(dict_bin_path: str, bin_index_path: str) -> Dict[str, Dict[int, List[int]]]:
    """
    从二进制文件加载带位置信息的完整倒排索引
    
    参数:
        dict_bin_path: 词典二进制文件路径
        bin_index_path: 倒排索引二进制文件路径
    返回:
        字典结构: {term: {doc_id: [pos1, pos2, ...]}}
    
    示例:
        {"apple": {1: [3, 10], 2: [5]}}
        表示 "apple" 在文档1的位置3和10，在文档2的位置5
    """
    # 加载词典
    dictionary = load_dictionary_from_binary(dict_bin_path)
    
    # 根据词典中的指针加载所有带位置信息的倒排表
    positional_index = {}
    for term, (doc_freq, pointer) in dictionary.items():
        doc_positions = load_positional_posting_list(bin_index_path, pointer, doc_freq)
        positional_index[term] = doc_positions
    
    return positional_index


# ==================== 模块 2：短语查询算法 ====================

def phrase_query(phrase: str, positional_index: Dict[str, Dict[int, List[int]]]) -> Tuple[Set[int], float]:
    """
    执行短语查询（精确匹配）
    
    参数:
        phrase: 查询短语，如 "new york"
        positional_index: 带位置信息的倒排索引
    返回:
        (匹配的文档ID集合, 执行时间)
    """
    start_time = time.perf_counter()
    
    # 将短语分词
    terms = phrase.lower().strip().split()
    if not terms:
        return set(), 0
    
    # 如果只有一个词，直接返回包含该词的文档
    if len(terms) == 1:
        term = terms[0]
        if term in positional_index:
            result_docs = set(positional_index[term].keys())
        else:
            result_docs = set()
        end_time = time.perf_counter()
        return result_docs, end_time - start_time
    
    # 兼容：若短语以空格分隔，但词典中存储为下划线合并形式（如 "come over" -> "come_over"）
    underscored_term = "_".join(terms)
    if underscored_term in positional_index:
        result_docs = set(positional_index[underscored_term].keys())
        end_time = time.perf_counter()
        return result_docs, end_time - start_time
    
    # 获取第一个词的文档列表
    first_term = terms[0]
    if first_term not in positional_index:
        end_time = time.perf_counter()
        return set(), end_time - start_time
    
    # 候选文档：包含所有词项的文档（交集）
    candidate_docs = set(positional_index[first_term].keys())
    for term in terms[1:]:
        if term not in positional_index:
            end_time = time.perf_counter()
            return set(), end_time - start_time
        candidate_docs &= set(positional_index[term].keys())
    
    # 对每个候选文档，检查位置是否连续
    result_docs = set()
    for doc_id in candidate_docs:
        if is_phrase_in_document(terms, doc_id, positional_index):
            result_docs.add(doc_id)
    
    end_time = time.perf_counter()
    return result_docs, end_time - start_time


def is_phrase_in_document(terms: List[str], doc_id: int, 
                          positional_index: Dict[str, Dict[int, List[int]]]) -> bool:
    """
    检查短语是否在指定文档中以连续形式出现
    
    参数:
        terms: 词项列表
        doc_id: 文档ID
        positional_index: 位置索引
    返回:
        是否找到匹配
    """
    # 获取第一个词在该文档中的所有位置
    first_positions = positional_index[terms[0]][doc_id]
    
    # 对于第一个词的每个位置，检查后续词是否连续出现
    for start_pos in first_positions:
        match = True
        for i, term in enumerate(terms[1:], 1):
            expected_pos = start_pos + i
            if expected_pos not in positional_index[term][doc_id]:
                match = False
                break
        
        if match:
            return True
    
    return False


def proximity_search(term1: str, term2: str, max_distance: int,
                    positional_index: Dict[str, Dict[int, List[int]]]) -> Tuple[Set[int], float]:
    """
    邻近搜索：查找两个词在指定距离内出现的文档
    
    参数:
        term1: 第一个词
        term2: 第二个词
        max_distance: 最大距离（词数）
        positional_index: 位置索引
    返回:
        (匹配的文档ID集合, 执行时间)
    """
    start_time = time.perf_counter()
    
    # 检查词项是否存在
    if term1 not in positional_index or term2 not in positional_index:
        end_time = time.perf_counter()
        return set(), end_time - start_time
    
    # 找到同时包含两个词的文档
    docs1 = set(positional_index[term1].keys())
    docs2 = set(positional_index[term2].keys())
    candidate_docs = docs1 & docs2
    
    # 检查距离
    result_docs = set()
    for doc_id in candidate_docs:
        positions1 = positional_index[term1][doc_id]
        positions2 = positional_index[term2][doc_id]
        
        # 检查任意两个位置的距离
        for pos1 in positions1:
            for pos2 in positions2:
                if abs(pos2 - pos1) <= max_distance:
                    result_docs.add(doc_id)
                    break
            if doc_id in result_docs:
                break
    
    end_time = time.perf_counter()
    return result_docs, end_time - start_time


# ==================== 模块 3：查询设计 ====================

def design_phrase_queries() -> List[Dict]:
    """
    设计测试查询（包括短语查询和邻近查询）
    
    返回:
        查询字典列表
    """
    queries = [
        # 短语查询
        {
            "type": "phrase",
            "query": "favorite recipes",
            "description": "短语查询：favorite recipes"
        },
        {
            "type": "phrase",
            "query": "special recipe",
            "description": "短语查询：special recipe"
        },
        {
            "type": "phrase",
            "query": "Please note",
            "description": "短语查询：Please note"
        },
        {
            "type": "phrase",
            "query": "come over",
            "description": "短语查询：come over"
        },
        {
            "type": "phrase",
            "query": "happy hour",
            "description": "短语查询：happy hour"
        },
        {
            "type": "phrase",
            "query": "parking lot",
            "description": "短语查询：parking lot"
        },
        
        # 邻近查询
        {
            "type": "proximity",
            "term1": "food",
            "term2": "wine",
            "distance": 5,
            "description": "邻近查询：food 和 wine 在5个词以内"
        },
        {
            "type": "proximity",
            "term1": "mountain",
            "term2": "hike",
            "distance": 3,
            "description": "邻近查询：mountain 和 hike 在3个词以内"
        },
        {
            "type": "proximity",
            "term1": "free",
            "term2": "event",
            "distance": 10,
            "description": "邻近查询：free 和 event 在10个词以内"
        }
    ]
    
    return queries


# ==================== 模块 4：结果输出 ====================

def display_results(query_info: Dict, result_docs: Set[int], exec_time: float):
    """
    显示查询结果
    
    参数:
        query_info: 查询信息
        result_docs: 结果文档集合
        exec_time: 执行时间
    """
    print(f"查询: {query_info['description']}")
    print(f"{'=' * 80}")
    print(f"执行时间:     {exec_time:.6f}s")
    print(f"匹配文档数:   {len(result_docs)}")
    
    if result_docs:
        doc_list = sorted(list(result_docs))
        print(f"匹配文档ID列表:")
        print(" ".join(map(str, doc_list)))
    else:
        print(f"结果文档ID示例:       无匹配结果")
    
    print()


def save_results(queries: List[Dict], results: List[Tuple], output_file: str):
    """
    保存结果到文件
    
    参数:
        queries: 查询列表
        results: 结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("短语检索实验结果\n")
        f.write("=" * 80 + "\n\n")
        
        for query_info, (result_docs, exec_time) in zip(queries, results):
            f.write(f"查询: {query_info['description']}\n")
            f.write("-" * 80 + "\n")
            f.write(f"执行时间:     {exec_time:.6f}s\n")
            f.write(f"匹配文档数:   {len(result_docs)}\n")
            
            if result_docs:
                doc_list = sorted(list(result_docs))
                f.write("匹配文档ID列表:\n")
                # 每行最多写入若干ID，便于阅读
                line = []
                for i, doc_id in enumerate(doc_list, 1):
                    line.append(str(doc_id))
                    if i % 20 == 0:
                        f.write(" ".join(line) + "\n")
                        line = []
                if line:
                    f.write(" ".join(line) + "\n")
            else:
                f.write(f"结果文档ID示例:       无匹配结果\n")
            
            f.write("\n")
        
        f.write("=" * 80 + "\n")


# ==================== 模块 5：性能统计 ====================

def analyze_performance(queries: List[Dict], results: List[Tuple]):
    """
    分析查询性能
    
    参数:
        queries: 查询列表
        results: 结果列表
    """
    print("=" * 80)
    print("性能统计分析")
    print("=" * 80)
    
    phrase_times = []
    proximity_times = []
    
    for query_info, (result_docs, exec_time) in zip(queries, results):
        if query_info['type'] == 'phrase':
            phrase_times.append(exec_time)
        else:
            proximity_times.append(exec_time)
    
    if phrase_times:
        avg_phrase_time = sum(phrase_times) / len(phrase_times)
        print(f"短语查询平均时间:     {avg_phrase_time:.6f}s")
        print(f"短语查询最快时间:     {min(phrase_times):.6f}s")
        print(f"短语查询最慢时间:     {max(phrase_times):.6f}s")
        print()
    
    if proximity_times:
        avg_proximity_time = sum(proximity_times) / len(proximity_times)
        print(f"邻近查询平均时间:     {avg_proximity_time:.6f}s")
        print(f"邻近查询最快时间:     {min(proximity_times):.6f}s")
        print(f"邻近查询最慢时间:     {max(proximity_times):.6f}s")
        print()
    
    print("=" * 80)
    print()


# ==================== 主程序 ====================

def main():
    """
    主函数：执行短语检索实验
    """
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("正在从二进制文件加载位置倒排索引...")
    print(f"  - 词典文件: {DICT_BIN_PATH}")
    print(f"  - 倒排索引文件: {INDEX_BIN_PATH}")
    positional_index = load_positional_index_from_binary(DICT_BIN_PATH, INDEX_BIN_PATH)
    print(f"加载完成！共 {len(positional_index)} 个词项\n")
    
    # 统计文档数量
    all_docs = set()
    for doc_dict in positional_index.values():
        all_docs.update(doc_dict.keys())
    print(f"文档总数: {len(all_docs)}\n")
    
    # 设计查询
    queries = design_phrase_queries()
    
    # 执行查询
    print("=" * 80)
    print("开始执行短语检索实验")
    print("=" * 80)
    print()
    
    results = []
    for query_info in queries:
        if query_info['type'] == 'phrase':
            result_docs, exec_time = phrase_query(query_info['query'], positional_index)
        else:  # proximity
            result_docs, exec_time = proximity_search(
                query_info['term1'], 
                query_info['term2'], 
                query_info['distance'],
                positional_index
            )
        
        results.append((result_docs, exec_time))
        display_results(query_info, result_docs, exec_time)
    
    # 性能分析
    analyze_performance(queries, results)
    
    # 保存结果
    save_results(queries, results, OUTPUT_FILE)
    print(f"实验结果已保存到: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

