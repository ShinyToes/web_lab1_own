"""
带跳表指针的倒排索引构建器
实现二进制格式和VB+Gap压缩格式的跳表指针支持
"""

import os
import struct
import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from .inverted_index_builder import InvertedIndexBuilder


class InvertedIndexBuilderWithSkip(InvertedIndexBuilder):
    """带跳表指针的倒排索引构建器"""
    
    # 跳表指针标记值
    SKIP_MARKER_BINARY = 0xFFFFFFFF  # 二进制格式：4字节标记
    SKIP_MARKER_COMPRESSED = 0xFD   # 压缩格式：1字节标记（0xFD = 253）
    
    def __init__(self, config_path: str = None):
        """初始化：继承父类，添加跳表指针支持"""
        # 调用父类初始化
        super().__init__(config_path)
        
        # 检查是否启用跳表指针
        skip_config = self.config.get("skip_pointer", {})
        self.skip_enabled = skip_config.get("enabled", True)
        
        # 从配置文件读取标记值（如果存在）
        self.SKIP_MARKER_BINARY = skip_config.get("skip_marker_binary", 0xFFFFFFFF)
        self.SKIP_MARKER_COMPRESSED = skip_config.get("skip_marker_compressed", 0xFD)
        self.min_list_length = skip_config.get("min_list_length", 3)
        
        if not self.skip_enabled:
            print("警告：跳表指针未启用")
        else:
            print(f"跳表指针配置：")
            print(f"  - 二进制格式标记: 0x{self.SKIP_MARKER_BINARY:08X}")
            print(f"  - 压缩格式标记: 0x{self.SKIP_MARKER_COMPRESSED:02X}")
            print(f"  - 最小列表长度: {self.min_list_length}")
    
    def calculate_skip_pointers(self, doc_ids: List[int], file_positions: List[int] = None) -> List[Tuple[int, int, int]]:
        """
        计算跳表指针位置和目标值（间隔√L均匀放置）
        
        参数:
            doc_ids: 排序后的文档ID列表
            file_positions: 每个文档ID对应的文件位置（file_pos）列表，如果为None则只返回doc_id
            
        返回:
            跳表指针列表，每个元素为(位置索引, 目标文档ID值, 目标文件位置)
            如果file_positions为None，则file_pos为-1
        """
        L = len(doc_ids)
        if L < self.min_list_length:
            return []  # 太短的列表不需要跳表指针
        
        # 计算跳表步长：√L
        skip_step = int(math.sqrt(L))
        if skip_step < 1:
            skip_step = 1
        
        skip_pointers = []
        
        # 每隔 skip_step 个位置放置跳表指针
        # 跳表指针指向当前位置后 skip_step 个位置的文档ID值
        i = skip_step - 1  # 从第 skip_step 个位置开始（0索引）
        while i < L - 1:  # 最后一个位置不放置跳表指针
            target_index = min(i + skip_step, L - 1)
            target_doc_id = doc_ids[target_index]
            target_file_pos = file_positions[target_index] if file_positions and target_index < len(file_positions) else -1
            skip_pointers.append((i, target_doc_id, target_file_pos))
            i += skip_step
        
        return skip_pointers
    
    def vb_decode_number(self, data: bytes, start_pos: int) -> Tuple[int, int]:
        """
        VB解码：从字节流中解码一个整数
        
        返回:
            (解码后的值, 读取的字节数)
        """
        value = 0
        pos = start_pos
        byte_count = 0
        
        while pos < len(data):
            byte = data[pos]
            byte_count += 1
            value = value * 128 + (byte & 127)
            
            if byte >= 128:  # 最高位为1，表示结束
                break
            
            pos += 1
        
        return value, byte_count
    
    def decode_gaps(self, gaps: List[int]) -> List[int]:
        """Gap解码：将差值编码转换回文档ID列表"""
        if not gaps:
            return []
        
        doc_ids = [gaps[0]]
        for i in range(1, len(gaps)):
            doc_ids.append(doc_ids[i-1] + gaps[i])
        
        return doc_ids
    
    def generate_dictionary(self, inverted_index_path: str) -> Tuple[str, str]:
        """
        生成定长存储的词典文件（重写父类方法，考虑跳表指针）
        
        关键修改：计算binary_pointer时，需要考虑跳表指针占用的空间
        每个跳表指针占用：标记(4字节) + 目标doc_id(4字节) + 目标file_pos(4字节) = 12字节
        
        参数:
            inverted_index_path: 倒排索引文件路径
            
        返回:
            (词典文本文件路径, 词典二进制文件路径)
        """
        print("=" * 60)
        print("生成词典文件（定长存储，带跳表指针）")
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
        
        # 准备词典数据（考虑跳表指针）
        print("构建词典数据（计算跳表指针开销）...")
        dictionary_entries = []
        text_pointer = 0
        binary_pointer = 0  # 带跳表指针的二进制指针
        
        storage = self.config["storage"]
        
        for term in sorted(inverted_index.keys()):
            doc_ids = inverted_index[term]
            doc_freq = len(set(doc_ids))  # 去重后的文档频率
            
            posting_list_str = ','.join(doc_ids)
            text_posting_size = len(posting_list_str.encode('utf-8')) + 1
            
            # 计算二进制posting list大小（考虑跳表指针）
            # 基础大小：文档ID数量 × 每个文档ID大小
            base_size = doc_freq * storage["doc_id_size"]
            
            # 计算跳表指针数量
            doc_ids_int = []
            for doc_id_str in doc_ids:
                try:
                    if doc_id_str.startswith('event_'):
                        doc_id = int(doc_id_str.replace('event_', ''))
                    else:
                        doc_id = int(doc_id_str)
                    doc_ids_int.append(doc_id)
                except ValueError:
                    doc_ids_int.append(hash(doc_id_str) & storage["doc_id_max_value"])
            
            doc_ids_int = sorted(set(doc_ids_int))
            # 计算跳表指针（这里只用于估算大小，不包含file_pos信息）
            # 实际写入时会在generate_binary_inverted_index中计算准确的file_pos
            skip_pointers = self.calculate_skip_pointers(doc_ids_int, None)
            skip_overhead = len(skip_pointers) * 12  # 每个跳表指针12字节（标记4字节+doc_id 4字节+file_pos 4字节）
            
            # 总大小 = 基础大小 + 跳表指针开销
            binary_posting_size = base_size + skip_overhead
            
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
            f.write(f"每词项总开销: {storage['entry_size']} 字节\n")
            f.write(f"注意：二进制倒排表包含跳表指针，词典中的指针已考虑跳表指针开销\n\n")
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
        
        # 生成二进制格式词典（指针指向带跳表指针的二进制文件）
        dict_bin_path = os.path.join(output_dir, self.config["files"]["dictionary_bin"])
        print(f"\n写入二进制格式词典: {dict_bin_path}")
        print("注意：词典中的指针指向带跳表指针的二进制倒排表文件")
        
        truncated_count = 0
        with open(dict_bin_path, 'wb') as f:
            for entry in dictionary_entries:
                term = entry['term']
                doc_freq = entry['doc_freq']
                pointer = entry['binary_pointer']  # 这个指针考虑了跳表指针开销
                
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
    
    def generate_binary_inverted_index(self, inverted_index_path: str) -> str:
        """
        生成带跳表指针的二进制格式倒排表（重写父类方法）
        
        存储格式：
        [文档ID: 4字节] 
        [文档ID: 4字节] 
        [SKIP标记: 0xFFFFFFFF (4字节)] [跳表目标 doc_id: 4字节] [跳表目标 file_pos: 4字节]  ← 跳表指针
        [文档ID: 4字节]
        ...
        
        参数:
            inverted_index_path: 文本格式倒排索引文件路径
            
        返回:
            输出文件路径
        """
        print("=" * 60)
        print("生成带跳表指针的二进制格式倒排表")
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
        
        # 确定输出路径（使用带跳表指针的文件名）
        output_dir = self.config["output_dir"]
        file_name = self.config.get("files", {}).get("inverted_index_bin_with_skip", "inverted_index_skiplist.bin")
        output_path = os.path.join(output_dir, file_name)
        
        print(f"\n写入带跳表指针的二进制倒排表: {output_path}")
        
        storage = self.config["storage"]
        total_doc_ids = 0
        total_skip_pointers = 0
        skip_stats = {}
        
        with open(output_path, 'wb') as f:
            # 按照词项顺序（与词典一致）
            for term in sorted(inverted_index.keys()):
                doc_ids = inverted_index[term]
                
                # 转换为整数列表（复用父类逻辑）
                doc_ids_int = []
                for doc_id_str in doc_ids:
                    try:
                        if doc_id_str.startswith('event_'):
                            doc_id = int(doc_id_str.replace('event_', ''))
                        else:
                            doc_id = int(doc_id_str)
                        
                        if doc_id > storage["doc_id_max_value"]:
                            doc_id = doc_id & storage["doc_id_max_value"]
                        
                        doc_ids_int.append(doc_id)
                    except ValueError:
                        doc_id = hash(doc_id_str) & storage["doc_id_max_value"]
                        doc_ids_int.append(doc_id)
                
                # 确保排序
                doc_ids_int = sorted(set(doc_ids_int))
                
                # 记录term的起始位置
                term_start_pos = f.tell()
                
                # 先计算跳表指针位置（不考虑file_pos）
                skip_step = int(math.sqrt(len(doc_ids_int))) if len(doc_ids_int) >= self.min_list_length else 1
                skip_positions = []
                if skip_step >= 1 and len(doc_ids_int) >= self.min_list_length:
                    i = skip_step - 1
                    while i < len(doc_ids_int) - 1:
                        skip_positions.append(i)
                        i += skip_step
                
                # 统计
                total_doc_ids += len(doc_ids_int)
                total_skip_pointers += len(skip_positions)
                if len(skip_positions) > 0:
                    skip_stats[term] = len(skip_positions)
                
                # 写入文档ID和跳表指针
                skip_pos_idx = 0
                inserted_skip_count = 0  # 已经插入的跳表指针数量（用于计算后续位置的偏移）
                
                for i, doc_id in enumerate(doc_ids_int):
                    # 写入文档ID
                    f.write(struct.pack('I', doc_id))
                    
                    # 检查当前位置是否需要跳表指针
                    if skip_pos_idx < len(skip_positions) and i == skip_positions[skip_pos_idx]:
                        # 计算目标位置
                        target_index = min(i + skip_step, len(doc_ids_int) - 1)
                        target_doc_id = doc_ids_int[target_index]
                        
                        # 计算目标doc_id的file_pos
                        # 目标位置 = term_start_pos + 目标doc_id的偏移（考虑已插入的跳表指针）
                        # 目标doc_id之前插入的跳表指针数量
                        skip_count_before_target = sum(1 for pos in skip_positions if pos < target_index)
                        # 目标doc_id的file_pos = term起始位置 + target_index * 4（doc_id大小） + skip_count_before_target * 12（跳表指针大小）
                        target_file_pos = term_start_pos + target_index * 4 + skip_count_before_target * 12
                        
                        # 写入跳表指针：标记 + 目标doc_id + 目标file_pos
                        f.write(struct.pack('I', self.SKIP_MARKER_BINARY))
                        f.write(struct.pack('I', target_doc_id))
                        f.write(struct.pack('I', target_file_pos))
                        
                        skip_pos_idx += 1
                        inserted_skip_count += 1
        
        print(f"✓ 带跳表指针的二进制倒排表已生成")
        
        # 统计信息
        bin_file_size = os.path.getsize(output_path)
        
        print(f"\n【带跳表指针的二进制倒排表统计】")
        print(f"  - 词项数量: {len(inverted_index)}")
        print(f"  - 总文档ID数: {total_doc_ids}")
        print(f"  - 总跳表指针数: {total_skip_pointers}")
        print(f"  - 平均每词项跳表指针数: {total_skip_pointers/len(inverted_index):.2f}" if inverted_index else "0")
        print(f"  - 二进制文件大小: {bin_file_size:,} 字节 ({bin_file_size / 1024:.2f} KB)")
        print(f"  - 跳表指针额外开销: {total_skip_pointers * 12:,} 字节 ({total_skip_pointers * 12 / 1024:.2f} KB)")
        print()
        
        return output_path
    
    def generate_compressed_inverted_index(self, inverted_index_path: str) -> Tuple[str, Dict]:
        """
        生成带跳表指针的VB+Gap压缩倒排表（重写父类方法）
        
        存储格式：
        [Gap1: VB编码] [Gap2: VB编码] [SKIP标记: 0xFD (1字节)] [目标值: VB编码] [Gap3: VB编码] ...
        
        参数:
            inverted_index_path: 文本格式倒排索引文件路径
            
        返回:
            (压缩倒排索引文件路径, 压缩统计信息字典)
        """
        print("=" * 60)
        print("生成带跳表指针的VB+Gap压缩倒排表")
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
        
        # 确定输出路径（使用带跳表指针的文件名）
        output_dir = self.config["output_dir"]
        file_name = self.config.get("files", {}).get("inverted_index_compressed_with_skip", "inverted_index_compressed_skiplist.bin")
        compressed_path = os.path.join(output_dir, file_name)
        
        print(f"\n写入带跳表指针的VB+Gap压缩倒排表: {compressed_path}")
        
        stats = {
            'total_terms': 0,
            'total_doc_ids': 0,
            'total_skip_pointers': 0,
            'uncompressed_size': 0,
            'compressed_size': 0,
            'skip_overhead': 0
        }
        
        with open(compressed_path, 'wb') as f:
            for term in sorted(inverted_index.keys()):
                doc_ids_str = inverted_index[term]
                
                # 转换为整数列表
                doc_ids_int = []
                for doc_id_str in doc_ids_str:
                    try:
                        doc_ids_int.append(int(doc_id_str))
                    except ValueError:
                        doc_ids_int.append(hash(doc_id_str) & 0x7FFFFFFF)
                
                doc_ids_int = sorted(set(doc_ids_int))
                
                # Gap编码
                doc_id_gaps = self.encode_gaps(doc_ids_int)
                
                # 计算跳表指针（基于原始文档ID位置）
                # 对于压缩格式，不需要file_pos，所以传入None
                skip_pointers = self.calculate_skip_pointers(doc_ids_int, None)
                
                # 构建跳表指针在gap序列中的映射
                # skip_pointers存储的是(原始位置, 目标doc_id, 目标file_pos)
                # 对于压缩格式，只需要doc_id，不需要file_pos
                # 需要转换为(gap位置, 目标doc_id)
                gap_skip_pointers = []
                for orig_pos, target_doc_id, _ in skip_pointers:
                    # 在gap序列中，位置需要减1（因为第一个gap是绝对值）
                    if orig_pos > 0:
                        gap_skip_pointers.append((orig_pos - 1, target_doc_id))
                
                # 编码：插入跳表指针标记
                encoded_data = []
                skip_ptr_idx = 0
                
                for i, gap in enumerate(doc_id_gaps):
                    # 写入gap的VB编码
                    gap_bytes = self.vb_encode_number(gap)
                    encoded_data.append(gap_bytes)
                    
                    # 检查是否需要插入跳表指针
                    if skip_ptr_idx < len(gap_skip_pointers):
                        gap_pos, target_doc_id = gap_skip_pointers[skip_ptr_idx]
                        if i == gap_pos:
                            # 插入跳表指针标记
                            encoded_data.append(bytes([self.SKIP_MARKER_COMPRESSED]))
                            # 插入目标doc_id的VB编码（压缩格式不需要file_pos）
                            target_bytes = self.vb_encode_number(target_doc_id)
                            encoded_data.append(target_bytes)
                            skip_ptr_idx += 1
                            
                            stats['skip_overhead'] += 1 + len(target_bytes)  # 标记 + 目标doc_id
                
                # 写入文件
                final_data = b''.join(encoded_data)
                f.write(final_data)
                
                # 统计
                stats['total_terms'] += 1
                stats['total_doc_ids'] += len(doc_ids_int)
                stats['total_skip_pointers'] += len(skip_pointers)
                stats['uncompressed_size'] += len(doc_ids_int) * 4
                stats['compressed_size'] += len(final_data)
                
                if stats['total_terms'] % 100 == 0:
                    print(f"  已处理 {stats['total_terms']} 个词项...")
        
        # 计算压缩率
        stats['compression_ratio'] = stats['compressed_size'] / stats['uncompressed_size'] if stats['uncompressed_size'] > 0 else 0
        
        print(f"\n✓ 带跳表指针的VB+Gap压缩完成")
        print(f"  - 压缩文件: {compressed_path}")
        print(f"  - 未压缩大小: {stats['uncompressed_size']:,} 字节 ({stats['uncompressed_size']/1024:.2f} KB)")
        print(f"  - 压缩后大小: {stats['compressed_size']:,} 字节 ({stats['compressed_size']/1024:.2f} KB)")
        print(f"  - 压缩率: {stats['compression_ratio']:.2%}")
        print(f"  - 总跳表指针数: {stats['total_skip_pointers']}")
        print(f"  - 跳表指针开销: {stats['skip_overhead']:,} 字节 ({stats['skip_overhead']/1024:.2f} KB)")
        print(f"  - 跳表指针占总大小: {stats['skip_overhead']/stats['compressed_size']*100:.2f}%" if stats['compressed_size'] > 0 else "0%")
        print()
        
        return compressed_path, stats
    
    def decode_compressed_with_skip(
        self, 
        compressed_data: bytes, 
        doc_freq: int
    ) -> Tuple[List[int], List[Tuple[int, int]]]:
        """
        解码带跳表指针的压缩倒排列表
        
        参数:
            compressed_data: 压缩的字节流
            doc_freq: 文档频率（文档ID数量）
            
        返回:
            (文档ID列表, 跳表指针列表)
            跳表指针列表：[(位置索引, 目标值), ...]
        """
        doc_ids = []
        skip_pointers = []
        gaps = []
        
        i = 0
        current_gap_pos = 0
        
        while i < len(compressed_data) and len(doc_ids) < doc_freq:
            byte = compressed_data[i]
            
            # 检查是否是SKIP标记
            if byte == self.SKIP_MARKER_COMPRESSED:
                # 读取目标值
                target_value, bytes_read = self.vb_decode_number(compressed_data, i + 1)
                # 记录跳表指针：当前位置对应的文档ID位置
                skip_pointers.append((len(doc_ids) - 1, target_value))
                i += 1 + bytes_read
                continue
            
            # 正常VB解码
            gap, bytes_read = self.vb_decode_number(compressed_data, i)
            gaps.append(gap)
            i += bytes_read
        
        # Gap解码得到文档ID列表
        doc_ids = self.decode_gaps(gaps)
        
        return doc_ids, skip_pointers
    
    def decode_binary_with_skip(
        self, 
        binary_data: bytes,
        doc_freq: int
    ) -> Tuple[List[int], List[Tuple[int, int, int]]]:
        """
        解码带跳表指针的二进制倒排列表
        
        参数:
            binary_data: 二进制数据
            doc_freq: 文档频率（文档ID数量）
            
        返回:
            (文档ID列表, 跳表指针列表)
            跳表指针列表：[(位置索引, 目标doc_id, 目标file_pos), ...]
        """
        doc_ids = []
        skip_pointers = []
        
        i = 0
        current_pos = 0
        
        while i < len(binary_data) and len(doc_ids) < doc_freq:
            # 读取4字节
            if i + 4 > len(binary_data):
                break
            
            value = struct.unpack('I', binary_data[i:i+4])[0]
            i += 4
            
            # 检查是否是SKIP标记
            if value == self.SKIP_MARKER_BINARY:
                # 读取目标doc_id和目标file_pos（各4字节）
                if i + 8 <= len(binary_data):
                    target_doc_id = struct.unpack('I', binary_data[i:i+4])[0]
                    target_file_pos = struct.unpack('I', binary_data[i+4:i+8])[0]
                    skip_pointers.append((current_pos - 1, target_doc_id, target_file_pos))
                    i += 8
                continue
            
            # 普通文档ID
            doc_ids.append(value)
            current_pos += 1
        
        return doc_ids, skip_pointers

