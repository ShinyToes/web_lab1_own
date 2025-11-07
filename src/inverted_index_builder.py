"""
倒排索引构建器类
用于对规范化词项构建倒排表（Inverted Index）
采用三步流程：
1. 检索每篇文档，获得<词项，文档ID>对，并写入临时索引
2. 对临时索引中的词项进行排序
3. 遍历临时索引，对于相同词项的文档ID进行合并
"""

import os
import shutil
import struct
import json
from pathlib import Path
from typing import List, Dict, Tuple
from collections import defaultdict


class InvertedIndexBuilder:
    """倒排索引构建器类"""
    
    def __init__(self, config_path: str = None):
        """
        初始化倒排索引构建器
        
        参数:
            config_path: 配置文件路径，默认使用 config/inverted_index_config.json
        """
        # 如果没有指定config_path，使用默认的配置文件
        if config_path is None:
            script_dir = Path(__file__).parent
            project_root = script_dir.parent
            config_path = str(project_root / "config" / "inverted_index_config.json")
        
        # 加载配置文件
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print(f"✓ 已加载配置文件: {config_path}")
        else:
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        # 计算派生配置
        self._calculate_derived_config()
    
    def _calculate_derived_config(self):
        """计算派生配置项"""
        storage = self.config["storage"]
        compression = self.config["compression"]
        
        # 存储配置
        storage["entry_size"] = storage["term_max_length"] + storage["doc_freq_size"] + storage["pointer_size"]
        storage["doc_id_max_value"] = (1 << storage["doc_id_bits"]) - 1
        
        # 压缩配置
        compression["term_entry_with_ptr_size"] = (
            compression["term_bytes"] + compression["term_ptr_bytes"] + 
            compression["freq_bytes"] + compression["ptr_bytes"]
        )
        compression["term_entry_with_len_size"] = (
            compression["term_bytes"] + compression["term_len_bytes"] + 
            compression["freq_bytes"] + compression["ptr_bytes"]
        )
    
    def clear_output_directory(self):
        """清空输出目录，移除所有旧文件"""
        output_dir = self.config["output_dir"]
        if os.path.exists(output_dir):
            print(f"清空输出目录: {output_dir}")
            shutil.rmtree(output_dir)
            print(f"✓ 已清除旧文件\n")
        os.makedirs(output_dir, exist_ok=True)
    
    def get_text_files_from_directory(self, directory: str) -> List[str]:
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
            if file_name.lower().endswith('.txt') and file_name != 'mwe_phrases.txt':
                file_path = os.path.join(directory, file_name)
                txt_files.append(file_path)
        
        return txt_files
    
    def read_tokens_from_file(self, file_path: str) -> List[str]:
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
    
    def extract_document_id(self, file_path: str) -> str:
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
    
    def build_temp_index(self) -> List[Tuple[str, str]]:
        """
        第一步：检索每篇文档，获得<词项，文档ID>对（内存处理）
        
        返回:
            词项-文档ID对列表
        """
        print("=" * 60)
        print("第一步：构建临时索引（内存处理）")
        print("=" * 60)
        print()
        
        input_dir = self.config["input_dir"]
        file_paths = self.get_text_files_from_directory(input_dir)
        
        if not file_paths:
            print(f"错误：在目录 {input_dir} 中未找到任何文档文件")
            return []
        
        print(f"找到 {len(file_paths)} 篇文档")
        
        term_doc_pairs = []
        total_pairs = 0
        
        for idx, file_path in enumerate(file_paths, 1):
            doc_id = self.extract_document_id(file_path)
            tokens = self.read_tokens_from_file(file_path)
            
            if not tokens:
                print(f"[{idx}/{len(file_paths)}] 文档 {doc_id}: 无 token，跳过")
                continue
            
            for token in tokens:
                term_doc_pairs.append((token, doc_id))
                total_pairs += 1
            
            print(f"[{idx}/{len(file_paths)}] 文档 {doc_id}: 提取 {len(tokens)} 个词项")
        
        print(f"\n✓ 临时索引构建完成")
        print(f"  - 总文档数: {len(file_paths)}")
        print(f"  - 总词项对数: {total_pairs}\n")
        
        return term_doc_pairs
    
    def sort_temp_index(self, term_doc_pairs: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """
        第二步：对临时索引中的词项进行排序（内存处理）
        
        参数:
            term_doc_pairs: 词项-文档ID对列表
            
        返回:
            排序后的词项-文档ID对列表
        """
        print("=" * 60)
        print("第二步：对临时索引进行排序（内存处理）")
        print("=" * 60)
        print()
        
        print(f"处理 {len(term_doc_pairs)} 个词项对")
        print("正在排序...")
        term_doc_pairs.sort(key=lambda x: x[0])
        print(f"\n✓ 排序完成\n")
        
        return term_doc_pairs
    
    def merge_inverted_index(self, term_doc_pairs: List[Tuple[str, str]]) -> Tuple[str, Dict]:
        """
        第三步：遍历临时索引，对于相同词项的文档ID进行合并
        
        参数:
            term_doc_pairs: 排序后的词项-文档ID对列表
            
        返回:
            (倒排索引文件路径, 统计信息字典)
        """
        print("=" * 60)
        print("第三步：合并倒排索引")
        print("=" * 60)
        print()
        
        print("合并词项...")
        
        inverted_index = defaultdict(list)
        for term, doc_id in term_doc_pairs:
            inverted_index[term].append(doc_id)
        
        print(f"合并完成，共 {len(inverted_index)} 个唯一词项")
        
        # 写入最终的倒排索引
        output_dir = self.config["output_dir"]
        inverted_index_path = os.path.join(output_dir, self.config["files"]["inverted_index"])
        
        print("\n写入倒排索引文件...")
        with open(inverted_index_path, 'w', encoding='utf-8') as f:
            for term in sorted(inverted_index.keys()):
                doc_ids = inverted_index[term]
                # 数值去重并按数值升序排序，确保后续二进制/压缩写入都是数值有序
                def to_int(s: str) -> int:
                    return int(s.replace('event_', '')) if s.startswith('event_') else int(s)
                unique_sorted = sorted({to_int(x) for x in doc_ids})
                # 写回为字符串形式
                unique_doc_ids = [str(x) for x in unique_sorted]
                f.write(f"{term}\t{','.join(unique_doc_ids)}\n")
        
        # 统计信息
        total_postings = sum(len(set(doc_ids)) for doc_ids in inverted_index.values())
        avg_postings = total_postings / len(inverted_index) if inverted_index else 0
        
        stats = {
            'unique_terms': len(inverted_index),
            'total_postings': total_postings,
            'avg_postings_per_term': avg_postings
        }
        
        print(f"\n✓ 倒排索引构建完成")
        print(f"  - 唯一词项数: {stats['unique_terms']}")
        print(f"  - 总倒排项数: {stats['total_postings']}")
        print(f"  - 平均每个词项的文档数: {stats['avg_postings_per_term']:.2f}")
        print(f"  - 保存路径: {inverted_index_path}\n")
        
        return inverted_index_path, stats
    
    def generate_dictionary(self, inverted_index_path: str) -> Tuple[str, str]:
        """
        生成定长存储的词典文件（文本格式和二进制格式）
        
        参数:
            inverted_index_path: 倒排索引文件路径
            
        返回:
            (词典文本文件路径, 词典二进制文件路径)
        """
        print("=" * 60)
        print("生成词典文件（定长存储）")
        print("=" * 60)
        print()
        
        # 读取倒排索引
        print("读取倒排索引...")
        inverted_index = {}
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        term, doc_ids_str = parts
                        doc_ids = doc_ids_str.split(',')
                        inverted_index[term] = doc_ids
        
        print(f"读取了 {len(inverted_index)} 个词项")
        
        # 准备词典数据
        print("构建词典数据...")
        dictionary_entries = []
        text_pointer = 0
        binary_pointer = 0
        
        storage = self.config["storage"]
        
        for term in sorted(inverted_index.keys()):
            doc_ids = inverted_index[term]
            doc_freq = len(doc_ids)
            
            posting_list_str = ','.join(doc_ids)
            text_posting_size = len(posting_list_str.encode('utf-8')) + 1
            binary_posting_size = doc_freq * storage["doc_id_size"]
            
            dictionary_entries.append({
                'term': term,
                'doc_freq': doc_freq,
                'text_pointer': text_pointer,
                'binary_pointer': binary_pointer
            })
            
            text_pointer += text_posting_size
            binary_pointer += binary_posting_size
        
        # 生成文本格式词典
        output_dir = self.config["output_dir"]
        dict_text_path = os.path.join(output_dir, self.config["files"]["dictionary"])
        
        print(f"\n写入文本格式词典: {dict_text_path}")
        with open(dict_text_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("词典文件（定长存储格式 - 文本倒排索引）\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"词项固定长度: {storage['term_max_length']} 字节\n")
            f.write(f"文档频率字段: {storage['doc_freq_size']} 字节\n")
            f.write(f"倒排表指针: {storage['pointer_size']} 字节（基于文本格式）\n")
            f.write(f"每词项总开销: {storage['entry_size']} 字节\n\n")
            f.write("=" * 80 + "\n")
            f.write(f"{'词项':<30}{'文档频率':<15}{'倒排表指针':<15}\n")
            f.write("=" * 80 + "\n")
            
            for entry in dictionary_entries:
                term = entry['term']
                display_term = term if len(term) <= 28 else term[:25] + "..."
                f.write(f"{display_term:<30}{entry['doc_freq']:<15}{entry['text_pointer']:<15}\n")
            
            f.write("=" * 80 + "\n")
            f.write(f"总词项数: {len(dictionary_entries)}\n")
            f.write("=" * 80 + "\n")
        
        print(f"✓ 文本格式词典已生成")
        
        # 生成二进制格式词典
        dict_bin_path = os.path.join(output_dir, self.config["files"]["dictionary_bin"])
        print(f"\n写入二进制格式词典: {dict_bin_path}")
        
        truncated_count = 0
        with open(dict_bin_path, 'wb') as f:
            for entry in dictionary_entries:
                term = entry['term']
                doc_freq = entry['doc_freq']
                pointer = entry['binary_pointer']
                
                term_bytes = term.encode('utf-8')
                if len(term_bytes) > storage['term_max_length']:
                    term_bytes = term_bytes[:storage['term_max_length']]
                    truncated_count += 1
                else:
                    term_bytes = term_bytes.ljust(storage['term_max_length'], b'\x00')
                
                f.write(term_bytes)
                f.write(struct.pack('I', doc_freq))
                f.write(struct.pack('I', pointer))
        
        print(f"✓ 二进制格式词典已生成")
        
        # 计算文件大小
        bin_file_size = os.path.getsize(dict_bin_path)
        expected_size = len(dictionary_entries) * storage['entry_size']
        
        print(f"\n【词典统计】")
        print(f"  - 词项数量: {len(dictionary_entries)}")
        print(f"  - 词项最大长度: {storage['term_max_length']} 字节")
        print(f"  - 每词项开销: {storage['entry_size']} 字节")
        print(f"  - 二进制文件大小: {bin_file_size:,} 字节 ({bin_file_size / 1024:.2f} KB)")
        print(f"  - 预期文件大小: {expected_size:,} 字节")
        if truncated_count > 0:
            print(f"  - 警告: {truncated_count} 个词项被截断（超过 {storage['term_max_length']} 字节）")
        print()
        
        return dict_text_path, dict_bin_path
    
    def vb_encode_number(self, n: int) -> bytes:
        """VB编码：对单个整数进行可变字节编码"""
        if n == 0:
            return bytes([128])
        
        bytes_list = []
        while n > 0:
            bytes_list.insert(0, n % 128)
            n //= 128
        
        bytes_list[-1] += 128
        return bytes(bytes_list)
    
    def encode_gaps(self, numbers: List[int]) -> List[int]:
        """Gap编码：将绝对值列表转换为差值编码"""
        if not numbers:
            return []
        
        gaps = [numbers[0]]
        for i in range(1, len(numbers)):
            gaps.append(numbers[i] - numbers[i-1])
        
        return gaps
    
    def generate_compressed_inverted_index(self, inverted_index_path: str) -> Tuple[str, Dict]:
        """
        生成VB+Gap压缩的倒排表文件
        
        参数:
            inverted_index_path: 文本格式倒排索引文件路径
            
        返回:
            (压缩倒排索引文件路径, 压缩统计信息字典)
        """
        print("=" * 60)
        print("生成VB+Gap压缩倒排表")
        print("=" * 60)
        print()
        
        # 读取文本格式倒排索引
        print("读取文本格式倒排索引...")
        inverted_index = {}
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        term, doc_ids_str = parts
                        doc_ids = doc_ids_str.split(',')
                        inverted_index[term] = doc_ids
        
        print(f"读取了 {len(inverted_index)} 个词项")
        
        # 生成压缩倒排表
        output_dir = self.config["output_dir"]
        compressed_path = os.path.join(output_dir, "inverted_index_compressed.bin")
        print(f"\n写入VB+Gap压缩倒排表: {compressed_path}")
        
        stats = {
            'total_terms': 0,
            'total_doc_ids': 0,
            'uncompressed_size': 0,
            'compressed_size': 0
        }
        
        with open(compressed_path, 'wb') as f:
            for term in sorted(inverted_index.keys()):
                doc_ids_str = inverted_index[term]
                
                doc_ids_int = []
                for doc_id_str in doc_ids_str:
                    try:
                        doc_ids_int.append(int(doc_id_str))
                    except ValueError:
                        doc_ids_int.append(hash(doc_id_str) & 0x7FFFFFFF)
                
                doc_ids_int.sort()
                doc_id_gaps = self.encode_gaps(doc_ids_int)
                
                encoded_data = b''
                for gap in doc_id_gaps:
                    encoded_data += self.vb_encode_number(gap)
                
                f.write(encoded_data)
                
                stats['total_terms'] += 1
                stats['total_doc_ids'] += len(doc_ids_int)
                stats['uncompressed_size'] += len(doc_ids_int) * 4
                stats['compressed_size'] += len(encoded_data)
                
                if stats['total_terms'] % 100 == 0:
                    print(f"  已处理 {stats['total_terms']} 个词项...")
        
        # 计算压缩率
        stats['compression_ratio'] = stats['compressed_size'] / stats['uncompressed_size'] if stats['uncompressed_size'] > 0 else 0
        
        print(f"\n✓ VB+Gap压缩完成")
        print(f"  - 压缩文件: {compressed_path}")
        print(f"  - 未压缩大小: {stats['uncompressed_size']:,} 字节 ({stats['uncompressed_size']/1024:.2f} KB)")
        print(f"  - 压缩后大小: {stats['compressed_size']:,} 字节 ({stats['compressed_size']/1024:.2f} KB)")
        print(f"  - 压缩率: {stats['compression_ratio']:.2%}")
        print(f"  - 节省空间: {stats['uncompressed_size']-stats['compressed_size']:,} 字节 ({(stats['uncompressed_size']-stats['compressed_size'])/1024:.2f} KB)")
        print()
        
        return compressed_path, stats
    
    def generate_binary_inverted_index(self, inverted_index_path: str) -> str:
        """
        生成二进制格式的倒排表文件
        每个文档ID使用固定28bit（存储为4字节）
        
        参数:
            inverted_index_path: 文本格式倒排索引文件路径
            
        返回:
            二进制倒排索引文件路径
        """
        print("=" * 60)
        print("生成二进制格式倒排表")
        print("=" * 60)
        print()
        
        # 读取文本格式倒排索引
        print("读取文本格式倒排索引...")
        inverted_index = {}
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        term, doc_ids_str = parts
                        doc_ids = doc_ids_str.split(',')
                        inverted_index[term] = doc_ids
        
        print(f"读取了 {len(inverted_index)} 个词项")
        
        # 生成二进制倒排表
        output_dir = self.config["output_dir"]
        bin_index_path = os.path.join(output_dir, self.config["files"]["inverted_index_bin"])
        print(f"\n写入二进制倒排表: {bin_index_path}")
        
        storage = self.config["storage"]
        total_doc_ids = 0
        overflow_count = 0
        max_doc_id_seen = 0
        
        with open(bin_index_path, 'wb') as f:
            # 按照词项顺序（与词典一致）
            for term in sorted(inverted_index.keys()):
                doc_ids = inverted_index[term]
                
                # 直接写入文档ID列表（不写入文档频率，频率在词典中）
                # 写入每个文档ID（4字节，但限制在28bit范围）
                for doc_id_str in doc_ids:
                    try:
                        # 尝试将文档ID转换为整数
                        # 如果文档ID是字符串格式（如"event_123"），提取数字部分
                        if doc_id_str.startswith('event_'):
                            doc_id = int(doc_id_str.replace('event_', ''))
                        else:
                            # 尝试直接转换为整数
                            doc_id = int(doc_id_str)
                        
                        # 检查是否超过28bit范围
                        if doc_id > storage["doc_id_max_value"]:
                            doc_id = doc_id & storage["doc_id_max_value"]  # 截断到28bit
                            overflow_count += 1
                        
                        max_doc_id_seen = max(max_doc_id_seen, doc_id)
                        
                        # 写入文档ID（使用4字节无符号整数）
                        f.write(struct.pack('I', doc_id))
                        total_doc_ids += 1
                        
                    except ValueError:
                        # 如果无法转换为整数，使用hash作为备选方案
                        doc_id = hash(doc_id_str) & storage["doc_id_max_value"]
                        f.write(struct.pack('I', doc_id))
                        total_doc_ids += 1
        
        print(f"✓ 二进制倒排表已生成")
        
        # 计算文件大小
        bin_file_size = os.path.getsize(bin_index_path)
        
        print(f"\n【二进制倒排表统计】")
        print(f"  - 词项数量: {len(inverted_index)}")
        print(f"  - 总文档ID数: {total_doc_ids}")
        print(f"  - 文档ID位数: {storage['doc_id_bits']} bit")
        print(f"  - 文档ID存储大小: {storage['doc_id_size']} 字节")
        print(f"  - 文档ID最大值: {storage['doc_id_max_value']:,}")
        print(f"  - 实际最大文档ID: {max_doc_id_seen:,}")
        print(f"  - 二进制文件大小: {bin_file_size:,} 字节 ({bin_file_size / 1024:.2f} KB)")
        if overflow_count > 0:
            print(f"  - 警告: {overflow_count} 个文档ID超过28bit范围被截断")
        print()
        
        return bin_index_path
    
    def generate_compressed_dictionary(self, dict_text_path: str, compressed_index_path: str) -> str:
        """
        生成压缩词典（使用K-间隔指针压缩方法）
        
        参数:
            dict_text_path: 文本格式词典文件路径
            compressed_index_path: VB+Gap压缩倒排表文件路径
            
        返回:
            压缩词典文件路径
        """
        print("=" * 60)
        print("生成压缩词典（K-间隔指针压缩）")
        print("=" * 60)
        print()
        
        # 词典压缩参数配置
        compression = self.config["compression"]
        TERM_BYTES = compression["term_bytes"]
        FREQ_BYTES = compression["freq_bytes"]
        PTR_BYTES = compression["ptr_bytes"]
        TERM_PTR_BYTES = compression["term_ptr_bytes"]
        TERM_LEN_BYTES = compression["term_len_bytes"]
        K = compression["k_interval"]
        
        TERM_ENTRY_WITH_PTR_SIZE = compression["term_entry_with_ptr_size"]
        TERM_ENTRY_WITH_LEN_SIZE = compression["term_entry_with_len_size"]
        
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
                    '倒排表指针' in line or '每词项开销' in line):
                    continue
                
                parts = line.split()
                if len(parts) >= 3:
                    term = parts[0]
                    try:
                        freq = int(parts[1])
                        ptr = int(parts[2])  # 旧的指针（基于文本倒排表）
                        dictionary.append((term, freq, ptr))
                    except ValueError:
                        continue
        
        print(f"读取了 {len(dictionary)} 个词项")
        
        # 2. 重新计算压缩倒排表的指针
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
        output_dir = self.config["output_dir"]
        compressed_dict_path = os.path.join(output_dir, "compressed_dictionary.bin")
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
    
    def display_sample_index(self, inverted_index_path: str, sample_size: int = 10):
        """
        在控制台显示部分倒排索引样例
        
        参数:
            inverted_index_path: 倒排索引文件路径
            sample_size: 显示的样例数量
        """
        print("=" * 60)
        print(f"倒排索引样例（前 {sample_size} 个词项）")
        print("=" * 60)
        print()
        
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_size:
                    break
                line = line.strip()
                if line:
                    parts = line.split('\t')
                    if len(parts) == 2:
                        term, doc_ids = parts
                        doc_list = doc_ids.split(',')
                        print(f"词项: {term}")
                        print(f"  文档数: {len(doc_list)}")
                        print(f"  文档ID: {', '.join(doc_list[:10])}")
                        if len(doc_list) > 10:
                            print(f"           ... 还有 {len(doc_list) - 10} 个文档")
                        print()
    
    def build(self):
        """
        执行完整的倒排索引构建流程
        """
        print("\n" + "=" * 60)
        print("倒排索引构建工具")
        print("=" * 60 + "\n")
        
        # 清空输出目录
        self.clear_output_directory()
        
        # 第一步：构建临时索引
        term_doc_pairs = self.build_temp_index()
        if not term_doc_pairs:
            print("错误：临时索引构建失败")
            return
        
        # 第二步：排序临时索引
        sorted_pairs = self.sort_temp_index(term_doc_pairs)
        if not sorted_pairs:
            print("错误：索引排序失败")
            return
        
        # 第三步：合并倒排索引
        inverted_index_path, stats = self.merge_inverted_index(sorted_pairs)
        if not inverted_index_path:
            print("错误：倒排索引合并失败")
            return
        
        # 生成词典文件
        dict_text_path, dict_bin_path = self.generate_dictionary(inverted_index_path)
        
        # 生成二进制倒排表
        bin_index_path = self.generate_binary_inverted_index(inverted_index_path)
        
        # 生成VB+Gap压缩的倒排表
        compressed_index_path, compressed_stats = self.generate_compressed_inverted_index(inverted_index_path)
        
        # 生成压缩词典
        compressed_dict_path = self.generate_compressed_dictionary(dict_text_path, compressed_index_path)
        
        # 显示样例
        self.display_sample_index(inverted_index_path, sample_size=10)
        
        print("=" * 60)
        print("倒排索引构建完成！")
        print("=" * 60)
        print(f"\n输出文件：")
        output_dir = self.config["output_dir"]
        print(f"  - 倒排索引（文本）: {os.path.join(output_dir, self.config['files']['inverted_index'])}")
        print(f"  - 倒排索引（二进制）: {bin_index_path}")
        print(f"  - 倒排索引（VB+Gap压缩）: {compressed_index_path}")
        print(f"  - 词典（文本）: {os.path.join(output_dir, self.config['files']['dictionary'])}")
        print(f"  - 词典（二进制）: {os.path.join(output_dir, self.config['files']['dictionary_bin'])}")
        print(f"  - 词典（压缩，K-间隔指针）: {compressed_dict_path}")
        
        print(f"\n【基本统计】")
        print(f"  - 唯一词项数: {stats['unique_terms']:,}")
        print(f"  - 总倒排项数: {stats['total_postings']:,}")
        print(f"  - 平均每个词项的文档数: {stats['avg_postings_per_term']:.2f}")
        
        print(f"\n【压缩效果总结】")
        text_index_size = os.path.getsize(inverted_index_path)
        print(f"  - 文本格式: {text_index_size:,} 字节 ({text_index_size/1024:.2f} KB)")
        print(f"  - 未压缩二进制: {compressed_stats['uncompressed_size']:,} 字节 ({compressed_stats['uncompressed_size']/1024:.2f} KB)")
        print(f"  - VB+Gap压缩: {compressed_stats['compressed_size']:,} 字节 ({compressed_stats['compressed_size']/1024:.2f} KB)")
        print(f"  - 压缩率: {compressed_stats['compression_ratio']:.2%}")
        print()
