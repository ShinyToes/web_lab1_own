"""
向量空间模型文档检索系统
基于 TF-IDF 和余弦相似度的文档检索实现
"""

import os
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict, Counter
import time

from .event_text_processor import EventTextProcessor


class VectorSpaceRetrieval:
    """
    向量空间模型文档检索系统
    支持多种 TF-IDF 计算方法和余弦相似度检索
    """
    
    def __init__(self, config_path: str = None):
        """
        初始化检索系统
        
        参数:
            config_path: 配置文件路径
        """
        # 默认配置
        self.config = {
            "inverted_index_dir": "output_inverted_index",
            "documents_dir": "output/normalized_events", 
            "output_dir": "output_retrieval",
            "tfidf_method": "standard",
            "top_k": 10,
            "min_similarity": 0.0,
            "test_queries": [
                "music concert performance",
                "food recipe cooking", 
                "art exhibition gallery",
                "technology innovation",
                "education learning"
            ]
        }
        
        # 加载配置文件
        if config_path and os.path.exists(config_path):
            self.load_config(config_path)
        
        # 初始化组件
        self.text_processor = EventTextProcessor()
        
        # 数据存储
        self.inverted_index = {}
        self.documents = {}
        self.document_vectors = {}
        self.vocabulary = set()
        self.doc_frequencies = {}
        self.total_documents = 0
        
        # 统计信息
        self.stats = {
            'total_documents': 0,
            'total_terms': 0,
            'avg_doc_length': 0,
            'vocabulary_size': 0
        }
    
    def load_config(self, config_path: str):
        """
        从配置文件加载配置
        
        参数:
            config_path: 配置文件路径
        """
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                user_config = json.load(f)
            
            # 递归更新配置
            self._update_config(self.config, user_config)
            print(f"已加载检索配置文件: {config_path}")
        except Exception as e:
            print(f"警告: 配置文件加载失败 {config_path}, 使用默认配置: {e}")
    
    def _update_config(self, base_config: dict, user_config: dict):
        """递归更新配置字典"""
        for key, value in user_config.items():
            if key in base_config and isinstance(base_config[key], dict) and isinstance(value, dict):
                self._update_config(base_config[key], value)
            else:
                base_config[key] = value
    
    def load_inverted_index(self, inverted_index_path: str):
        """
        加载倒排索引文件
        
        参数:
            inverted_index_path: 倒排索引文件路径
        """
        print("=" * 60)
        print("加载倒排索引")
        print("=" * 60)
        
        if not os.path.exists(inverted_index_path):
            print(f"错误: 倒排索引文件不存在: {inverted_index_path}")
            return False
        
        print(f"读取倒排索引文件: {inverted_index_path}")
        
        self.inverted_index = {}
        self.doc_frequencies = {}
        
        with open(inverted_index_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split('\t')
                    if len(parts) != 2:
                        print(f"警告: 第 {line_num} 行格式错误: {line}")
                        continue
                    
                    term, doc_ids_str = parts
                    doc_ids = doc_ids_str.split(',')
                    
                    # 存储倒排索引
                    self.inverted_index[term] = doc_ids
                    self.doc_frequencies[term] = len(doc_ids)
                    self.vocabulary.add(term)
                    
                except Exception as e:
                    print(f"警告: 处理第 {line_num} 行时出错: {e}")
                    continue
        
        print(f"倒排索引加载完成")
        print(f"  - 词项数量: {len(self.inverted_index):,}")
        print(f"  - 词汇表大小: {len(self.vocabulary):,}")
        
        return True
    
    def load_documents(self, documents_dir: str):
        """
        加载规范化文档
        
        参数:
            documents_dir: 文档目录路径
        """
        print("=" * 60)
        print("加载文档集合")
        print("=" * 60)
        
        if not os.path.exists(documents_dir):
            print(f"错误: 文档目录不存在: {documents_dir}")
            return False
        
        print(f"读取文档目录: {documents_dir}")
        
        self.documents = {}
        doc_files = [f for f in os.listdir(documents_dir) if f.endswith('.txt')]
        
        print(f"找到 {len(doc_files)} 个文档文件")
        
        total_tokens = 0
        doc_lengths = []
        
        for doc_file in doc_files:
            doc_id = os.path.splitext(doc_file)[0]
            doc_path = os.path.join(documents_dir, doc_file)
            
            try:
                with open(doc_path, 'r', encoding='utf-8') as f:
                    tokens = [line.strip() for line in f if line.strip()]
                
                if tokens:
                    self.documents[doc_id] = tokens
                    doc_length = len(tokens)
                    doc_lengths.append(doc_length)
                    total_tokens += doc_length
                    
                    if len(self.documents) % 20 == 0:
                        print(f"  已加载 {len(self.documents)} 个文档...")
                
            except Exception as e:
                print(f"警告: 加载文档 {doc_file} 失败: {e}")
                continue
        
        self.total_documents = len(self.documents)
        
        # 更新统计信息
        self.stats.update({
            'total_documents': self.total_documents,
            'total_terms': total_tokens,
            'avg_doc_length': total_tokens / self.total_documents if self.total_documents > 0 else 0,
            'vocabulary_size': len(self.vocabulary)
        })
        
        print(f"\n文档集合加载完成")
        print(f"  - 文档数量: {self.total_documents:,}")
        print(f"  - 总词项数: {total_tokens:,}")
        print(f"  - 平均文档长度: {self.stats['avg_doc_length']:.1f}")
        print(f"  - 词汇表大小: {self.stats['vocabulary_size']:,}")
        
        return True
    
    def build_document_vectors(self):
        """
        构建所有文档的 TF-IDF 向量
        """
        print("=" * 60)
        print("构建文档向量")
        print("=" * 60)
        
        if not self.documents or not self.inverted_index:
            print("错误: 请先加载文档和倒排索引")
            return False
        
        print(f"使用 TF-IDF 方法: {self.config['tfidf_method']}")
        print(f"为 {self.total_documents} 个文档构建向量...")
        
        self.document_vectors = {}
        
        # 创建词汇表到索引的映射
        vocab_list = sorted(list(self.vocabulary))
        vocab_to_index = {term: idx for idx, term in enumerate(vocab_list)}
        
        for doc_idx, (doc_id, tokens) in enumerate(self.documents.items()):
            if doc_idx % 20 == 0:
                print(f"  已处理 {doc_idx}/{self.total_documents} 个文档...")
            
            # 计算文档中每个词项的频率
            term_freq = Counter(tokens)
            max_freq = max(term_freq.values()) if term_freq else 1
            
            # 构建 TF-IDF 向量
            vector = np.zeros(len(vocab_list))
            
            for term, tf in term_freq.items():
                if term in vocab_to_index:
                    idx = vocab_to_index[term]
                    df = self.doc_frequencies.get(term, 1)
                    
                    # 计算 TF-IDF 权重
                    tfidf_weight = self._calculate_tfidf(tf, df, max_freq)
                    vector[idx] = tfidf_weight
            
            self.document_vectors[doc_id] = vector
        
        print(f"\n文档向量构建完成")
        print(f"  - 向量维度: {len(vocab_list):,}")
        print(f"  - 文档向量数量: {len(self.document_vectors):,}")
        
        return True
    
    def _calculate_tfidf(self, tf: int, df: int, max_freq: int = 1) -> float:
        """
        计算 TF-IDF 权重
        
        参数:
            tf: 词项频率
            df: 文档频率
            max_freq: 文档中最大词频
            
        返回:
            TF-IDF 权重值
        """
        method = self.config['tfidf_method']
        N = self.total_documents
        
        # 计算 IDF
        idf = math.log(N / df) if df > 0 else 0
        
        # 根据方法计算 TF 权重
        if method == "standard":
            tf_weight = tf
        elif method == "log":
            tf_weight = 1 + math.log(tf) if tf > 0 else 0
        elif method == "augmented":
            tf_weight = 0.5 + 0.5 * (tf / max_freq) if max_freq > 0 else 0
        elif method == "binary":
            tf_weight = 1 if tf > 0 else 0
        else:
            # 默认使用标准方法
            tf_weight = tf
        
        return tf_weight * idf
    
    def set_tfidf_method(self, method: str):
        """
        设置 TF-IDF 计算方法
        
        参数:
            method: 计算方法 ("standard", "log", "augmented", "binary")
        """
        valid_methods = ["standard", "log", "augmented", "binary"]
        if method in valid_methods:
            self.config['tfidf_method'] = method
            print(f"TF-IDF 方法已设置为: {method}")
        else:
            print(f"错误: 无效的 TF-IDF 方法 '{method}'")
            print(f"支持的方法: {', '.join(valid_methods)}")
    
    def get_available_tfidf_methods(self) -> List[str]:
        """
        获取可用的 TF-IDF 计算方法
        
        返回:
            方法列表
        """
        return ["standard", "log", "augmented", "binary"]
    
    def get_tfidf_method_description(self, method: str) -> str:
        """
        获取 TF-IDF 方法的描述
        
        参数:
            method: 方法名称
            
        返回:
            方法描述
        """
        descriptions = {
            "standard": "标准 TF-IDF: tf × log(N/df)",
            "log": "对数频率: (1 + log(tf)) × log(N/df)", 
            "augmented": "增强频率: (0.5 + 0.5×tf/max_tf) × log(N/df)",
            "binary": "二元频率: (1 if tf>0 else 0) × log(N/df)"
        }
        return descriptions.get(method, "未知方法")
    
    def preprocess_query(self, query_text: str) -> List[str]:
        """
        预处理查询文本
        
        参数:
            query_text: 原始查询文本
            
        返回:
            规范化后的词项列表
        """
        # 创建临时事件数据用于处理
        temp_event = {
            'description': query_text,
            'event_info': {'id': 'query'}
        }
        
        # 使用 EventTextProcessor 进行预处理
        normalized_tokens, _ = self.text_processor.process_single_event(temp_event)
        
        return normalized_tokens
    
    def build_query_vector(self, query_tokens: List[str]) -> np.ndarray:
        """
        构建查询向量
        
        参数:
            query_tokens: 预处理后的查询词项
            
        返回:
            查询的 TF-IDF 向量
        """
        if not self.vocabulary:
            print("错误: 词汇表为空，请先加载数据")
            return np.array([])
        
        # 创建词汇表到索引的映射
        vocab_list = sorted(list(self.vocabulary))
        vocab_to_index = {term: idx for idx, term in enumerate(vocab_list)}
        
        # 计算查询中每个词项的频率
        query_term_freq = Counter(query_tokens)
        max_freq = max(query_term_freq.values()) if query_term_freq else 1
        
        # 构建查询向量
        query_vector = np.zeros(len(vocab_list))
        
        for term, tf in query_term_freq.items():
            if term in vocab_to_index:
                idx = vocab_to_index[term]
                df = self.doc_frequencies.get(term, 1)
                
                # 计算 TF-IDF 权重
                tfidf_weight = self._calculate_tfidf(tf, df, max_freq)
                query_vector[idx] = tfidf_weight
        
        return query_vector
    
    def calculate_cosine_similarity(self, query_vector: np.ndarray, doc_vector: np.ndarray) -> float:
        """
        计算余弦相似度
        
        参数:
            query_vector: 查询向量
            doc_vector: 文档向量
            
        返回:
            余弦相似度值
        """
        # 计算点积
        dot_product = np.dot(query_vector, doc_vector)
        
        # 计算向量模长
        query_norm = np.linalg.norm(query_vector)
        doc_norm = np.linalg.norm(doc_vector)
        
        # 避免除零错误
        if query_norm == 0 or doc_norm == 0:
            return 0.0
        
        # 计算余弦相似度
        similarity = dot_product / (query_norm * doc_norm)
        
        return similarity
    
    def search(self, query_text: str, top_k: int = None) -> List[Tuple[str, float, Dict]]:
        """
        执行文档检索
        
        参数:
            query_text: 查询文本
            top_k: 返回的文档数量
            
        返回:
            检索结果列表 [(doc_id, similarity, metadata), ...]
        """
        if top_k is None:
            top_k = self.config['top_k']
        
        print(f"\n执行查询: '{query_text}'")
        print(f"返回前 {top_k} 个最相关文档")
        
        # 1. 预处理查询
        query_tokens = self.preprocess_query(query_text)
        print(f"查询词项: {query_tokens}")
        
        if not query_tokens:
            print("警告: 查询预处理后为空")
            return []
        
        # 2. 构建查询向量
        query_vector = self.build_query_vector(query_tokens)
        
        if query_vector.size == 0:
            print("错误: 查询向量构建失败")
            return []
        
        # 3. 计算与所有文档的相似度
        similarities = []
        
        for doc_id, doc_vector in self.document_vectors.items():
            similarity = self.calculate_cosine_similarity(query_vector, doc_vector)
            
            if similarity >= self.config['min_similarity']:
                # 获取文档元数据
                doc_tokens = self.documents.get(doc_id, [])
                metadata = {
                    'doc_length': len(doc_tokens),
                    'matched_terms': [term for term in query_tokens if term in doc_tokens],
                    'match_count': len([term for term in query_tokens if term in doc_tokens])
                }
                
                similarities.append((doc_id, similarity, metadata))
        
        # 4. 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # 5. 返回 Top-K 结果
        results = similarities[:top_k]
        
        print(f"找到 {len(similarities)} 个相关文档，返回前 {len(results)} 个")
        
        return results
    
    def batch_search(self, queries: List[str], top_k: int = None) -> Dict[str, List[Tuple[str, float, Dict]]]:
        """
        批量执行多个查询
        
        参数:
            queries: 查询文本列表
            top_k: 每个查询返回的文档数量
            
        返回:
            查询结果字典 {query: results}
        """
        if top_k is None:
            top_k = self.config['top_k']
        
        print(f"\n批量执行 {len(queries)} 个查询")
        
        results = {}
        for i, query in enumerate(queries, 1):
            print(f"\n[{i}/{len(queries)}] 处理查询: '{query}'")
            results[query] = self.search(query, top_k)
        
        return results
    
    def get_query_statistics(self, query_text: str) -> Dict:
        """
        获取查询的统计信息
        
        参数:
            query_text: 查询文本
            
        返回:
            查询统计信息
        """
        query_tokens = self.preprocess_query(query_text)
        
        if not query_tokens:
            return {}
        
        # 统计查询词项在文档集合中的分布
        term_stats = {}
        for term in query_tokens:
            df = self.doc_frequencies.get(term, 0)
            term_stats[term] = {
                'document_frequency': df,
                'collection_frequency': sum(1 for doc_tokens in self.documents.values() if term in doc_tokens),
                'idf': math.log(self.total_documents / df) if df > 0 else 0
            }
        
        return {
            'query_text': query_text,
            'processed_tokens': query_tokens,
            'token_count': len(query_tokens),
            'unique_tokens': len(set(query_tokens)),
            'term_statistics': term_stats,
            'vocabulary_coverage': len([t for t in query_tokens if t in self.vocabulary]) / len(query_tokens) if query_tokens else 0
        }
    
    def display_results(self, query_text: str, results: List[Tuple[str, float, Dict]], show_details: bool = True):
        """
        在控制台显示检索结果
        
        参数:
            query_text: 查询文本
            results: 检索结果列表
            show_details: 是否显示详细信息
        """
        print("\n" + "=" * 80)
        print(f"查询: '{query_text}'")
        print("=" * 80)
        
        if not results:
            print("未找到相关文档")
            return
        
        print(f"找到 {len(results)} 个相关文档:")
        print()
        
        for i, (doc_id, similarity, metadata) in enumerate(results, 1):
            print(f"{i:2d}. 文档ID: {doc_id}")
            print(f"    相似度: {similarity:.4f}")
            print(f"    文档长度: {metadata['doc_length']} 词项")
            print(f"    匹配词项: {metadata['match_count']}/{len(self.preprocess_query(query_text))}")
            
            if show_details and metadata['matched_terms']:
                print(f"    匹配的词: {', '.join(metadata['matched_terms'])}")
            
            # 显示文档的规范化原文
            doc_tokens = self.documents.get(doc_id, [])
            if doc_tokens:
                print(f"    规范化原文: {' '.join(doc_tokens)}")
            
            print()
    
    def save_results_to_file(self, query_text: str, results: List[Tuple[str, float, Dict]], 
                            output_dir: str, query_index: int = None) -> Tuple[str, str]:
        """
        保存检索结果到文件
        
        参数:
            query_text: 查询文本
            results: 检索结果列表
            output_dir: 输出目录
            query_index: 查询索引（用于文件命名）
            
        返回:
            (JSON文件路径, 文本文件路径)
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成文件名
        if query_index is not None:
            base_name = f"query_{query_index}_results"
        else:
            # 使用查询文本的前20个字符作为文件名
            safe_query = "".join(c for c in query_text[:20] if c.isalnum() or c in (' ', '-', '_')).strip()
            safe_query = safe_query.replace(' ', '_')
            base_name = f"query_{safe_query}_results"
        
        json_path = os.path.join(output_dir, f"{base_name}.json")
        txt_path = os.path.join(output_dir, f"{base_name}.txt")
        
        # 准备JSON数据
        json_data = {
            'query': query_text,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tfidf_method': self.config['tfidf_method'],
            'total_results': len(results),
            'results': []
        }
        
        for doc_id, similarity, metadata in results:
            # 获取文档的规范化原文
            doc_tokens = self.documents.get(doc_id, [])
            normalized_text = ' '.join(doc_tokens) if doc_tokens else ""
            
            json_data['results'].append({
                'doc_id': doc_id,
                'similarity': similarity,
                'doc_length': metadata['doc_length'],
                'match_count': metadata['match_count'],
                'matched_terms': metadata['matched_terms'],
                'normalized_text': normalized_text
            })
        
        # 保存JSON文件
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        
        # 保存可读文本文件
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"查询: '{query_text}'\n")
            f.write("=" * 80 + "\n")
            f.write(f"TF-IDF 方法: {self.config['tfidf_method']}\n")
            f.write(f"检索时间: {json_data['timestamp']}\n")
            f.write(f"结果数量: {len(results)}\n\n")
            
            if not results:
                f.write("未找到相关文档\n")
            else:
                f.write("检索结果:\n")
                f.write("-" * 80 + "\n")
                
                for i, (doc_id, similarity, metadata) in enumerate(results, 1):
                    f.write(f"{i:2d}. 文档ID: {doc_id}\n")
                    f.write(f"    相似度: {similarity:.4f}\n")
                    f.write(f"    文档长度: {metadata['doc_length']} 词项\n")
                    f.write(f"    匹配词项: {metadata['match_count']}/{len(self.preprocess_query(query_text))}\n")
                    
                    if metadata['matched_terms']:
                        f.write(f"    匹配的词: {', '.join(metadata['matched_terms'])}\n")
                    
                    # 添加规范化原文
                    doc_tokens = self.documents.get(doc_id, [])
                    if doc_tokens:
                        f.write(f"    规范化原文: {' '.join(doc_tokens)}\n")
                    
                    f.write("\n")
        
        print(f"结果已保存到:")
        print(f"  - JSON: {json_path}")
        print(f"  - 文本: {txt_path}")
        
        return json_path, txt_path
    
    def generate_retrieval_report(self, all_results: Dict[str, List[Tuple[str, float, Dict]]], 
                                output_dir: str) -> str:
        """
        生成详细的检索报告
        
        参数:
            all_results: 所有查询的结果字典
            output_dir: 输出目录
            
        返回:
            报告文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "retrieval_report.txt")
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("=" * 100 + "\n")
            f.write("向量空间模型文档检索报告\n")
            f.write("=" * 100 + "\n")
            f.write(f"生成时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"TF-IDF 方法: {self.config['tfidf_method']}\n")
            f.write(f"最小相似度阈值: {self.config['min_similarity']}\n")
            f.write(f"Top-K 设置: {self.config['top_k']}\n\n")
            
            # 系统统计信息
            f.write("系统统计信息:\n")
            f.write("-" * 50 + "\n")
            f.write(f"总文档数: {self.stats['total_documents']:,}\n")
            f.write(f"词汇表大小: {self.stats['vocabulary_size']:,}\n")
            f.write(f"平均文档长度: {self.stats['avg_doc_length']:.1f} 词项\n")
            f.write(f"总词项数: {self.stats['total_terms']:,}\n\n")
            
            # 查询结果统计
            f.write("查询结果统计:\n")
            f.write("-" * 50 + "\n")
            f.write(f"查询总数: {len(all_results)}\n")
            
            total_results = sum(len(results) for results in all_results.values())
            avg_results = total_results / len(all_results) if all_results else 0
            
            f.write(f"总结果数: {total_results}\n")
            f.write(f"平均每查询结果数: {avg_results:.1f}\n\n")
            
            # 详细查询结果
            f.write("详细查询结果:\n")
            f.write("=" * 100 + "\n")
            
            for i, (query, results) in enumerate(all_results.items(), 1):
                f.write(f"\n查询 {i}: '{query}'\n")
                f.write("-" * 80 + "\n")
                
                # 查询统计
                query_stats = self.get_query_statistics(query)
                f.write(f"预处理词项: {query_stats.get('processed_tokens', [])}\n")
                f.write(f"词项数量: {query_stats.get('token_count', 0)}\n")
                f.write(f"唯一词项: {query_stats.get('unique_tokens', 0)}\n")
                f.write(f"词汇表覆盖率: {query_stats.get('vocabulary_coverage', 0):.2%}\n")
                f.write(f"找到结果: {len(results)} 个\n\n")
                
                # 结果列表
                if results:
                    f.write("检索结果:\n")
                    for j, (doc_id, similarity, metadata) in enumerate(results, 1):
                        f.write(f"  {j:2d}. {doc_id} (相似度: {similarity:.4f}, ")
                        f.write(f"长度: {metadata['doc_length']}, ")
                        f.write(f"匹配: {metadata['match_count']})\n")
                else:
                    f.write("未找到相关文档\n")
                
                f.write("\n")
            
            # 词项分析
            f.write("词项分析:\n")
            f.write("=" * 100 + "\n")
            
            all_query_terms = set()
            for query in all_results.keys():
                query_stats = self.get_query_statistics(query)
                all_query_terms.update(query_stats.get('processed_tokens', []))
            
            f.write(f"所有查询中的唯一词项: {len(all_query_terms)}\n")
            f.write(f"词项列表: {', '.join(sorted(all_query_terms))}\n\n")
            
            # 词项频率分析
            f.write("词项频率分析:\n")
            f.write("-" * 50 + "\n")
            for term in sorted(all_query_terms):
                df = self.doc_frequencies.get(term, 0)
                idf = math.log(self.total_documents / df) if df > 0 else 0
                f.write(f"{term:20s}: DF={df:4d}, IDF={idf:.3f}\n")
        
        print(f"检索报告已生成: {report_path}")
        return report_path
    
    def save_statistics(self, all_results: Dict[str, List[Tuple[str, float, Dict]]], 
                       output_dir: str) -> str:
        """
        保存检索统计信息到JSON文件
        
        参数:
            all_results: 所有查询的结果字典
            output_dir: 输出目录
            
        返回:
            统计文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        stats_path = os.path.join(output_dir, "retrieval_statistics.json")
        
        # 计算统计信息
        total_queries = len(all_results)
        total_results = sum(len(results) for results in all_results.values())
        
        # 相似度分布
        all_similarities = []
        for results in all_results.values():
            all_similarities.extend([sim for _, sim, _ in results])
        
        similarity_stats = {
            'min': min(all_similarities) if all_similarities else 0,
            'max': max(all_similarities) if all_similarities else 0,
            'mean': sum(all_similarities) / len(all_similarities) if all_similarities else 0,
            'count': len(all_similarities)
        }
        
        # 构建统计JSON
        statistics = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_stats': self.stats,
            'config': self.config,
            'query_stats': {
                'total_queries': total_queries,
                'total_results': total_results,
                'avg_results_per_query': total_results / total_queries if total_queries > 0 else 0
            },
            'similarity_stats': similarity_stats,
            'query_details': {}
        }
        
        # 添加每个查询的详细信息
        for query, results in all_results.items():
            query_stats = self.get_query_statistics(query)
            statistics['query_details'][query] = {
                'result_count': len(results),
                'max_similarity': max([sim for _, sim, _ in results]) if results else 0,
                'min_similarity': min([sim for _, sim, _ in results]) if results else 0,
                'query_stats': query_stats
            }
        
        # 保存JSON文件
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(statistics, f, ensure_ascii=False, indent=2)
        
        print(f"统计信息已保存: {stats_path}")
        return stats_path
    
    def get_config(self) -> Dict:
        """获取当前配置"""
        return self.config
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        return self.stats.copy()
