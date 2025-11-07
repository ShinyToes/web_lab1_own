"""
事件文本处理器 - 主类
整合 XML 解析、分词和 Token 规范化功能
"""

import os
import re
import nltk
import shutil
import xml.etree.ElementTree as ET
import html
import json
import time
import multiprocessing
from pathlib import Path
from typing import List, Set, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet as wn, stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk import pos_tag
from collections import Counter


class EventTextProcessor:
    """
    事件文本处理器主类
    整合 XML 解析、分词和 Token 规范化功能
    """
    
    def __init__(self, config_file: str = None):
        """
        初始化处理器
        
        参数:
            config_file: 配置文件路径，如果为 None 则使用默认配置
        """
        self.config = self._load_config(config_file)
        self._ensure_nltk_data()
        self._initialize_components()
        
    def _load_config(self, config_file: str) -> Dict:
        """
        加载配置文件
        
        参数:
            config_file: 配置文件路径
            
        返回:
            配置字典
        """
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                print(f"✓ 已加载配置文件: {config_file}")
                return config
            except Exception as e:
                print(f"✗ 加载配置文件失败: {e}")
                print("使用默认配置")
        
        # 默认配置
        return {
            "input": {
                "xml_dir": "All_Unpack",
                "max_files": 50
            },
            "output": {
                "base_dir": "part_1/output",
                "past_events_dir": "past_events",
                "tokenized_events_dir": "tokenized_events", 
                "normalized_events_dir": "normalized_events"
            },
            "tokenization": {
                "use_mwe": True,
                "phrase_dict_file": "mwe_phrases.txt"
            },
            "normalization": {
                "remove_stopwords": True,
                "remove_numbers": True,
                "remove_punctuation": True,
                "remove_short_words": True,
                "min_word_length": 2,
                "use_lemmatization": True,
                "use_stemming": False
            },
            "processing": {
                "use_multithread": True,
                "max_workers": multiprocessing.cpu_count()
            }
        }
    
    def _ensure_nltk_data(self):
        """
        确保必要的 NLTK 数据已下载
        """
        required_data = [
            ('tokenizers/punkt', 'punkt'),
            ('tokenizers/punkt_tab', 'punkt_tab'),
            ('corpora/wordnet', 'wordnet'),
            ('corpora/omw-1.4', 'omw-1.4'),
            ('corpora/stopwords', 'stopwords'),
            ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
            ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng')
        ]
        
        print("检查 NLTK 数据...")
        for path, name in required_data:
            try:
                nltk.data.find(path)
                print(f"NLTK 数据 '{name}' 已存在")
            except LookupError:
                print(f"下载 NLTK 数据 '{name}'...")
                nltk.download(name, quiet=False)
                # 下载后重新检查
                try:
                    nltk.data.find(path)
                    print(f"NLTK 数据 '{name}' 下载成功")
                except LookupError:
                    print(f"NLTK 数据 '{name}' 下载失败，尝试强制下载...")
                    nltk.download(name, quiet=False, force=True)
                    try:
                        nltk.data.find(path)
                        print(f"NLTK 数据 '{name}' 强制下载成功")
                    except LookupError:
                        print(f"NLTK 数据 '{name}' 最终下载失败")
        print()
    
    def _initialize_components(self):
        """
        初始化处理组件
        """
        # 初始化线程配置
        self.max_workers = self.config.get("processing", {}).get(
            "max_workers", 
            multiprocessing.cpu_count()
        )
        self.use_multithread = self.config.get("processing", {}).get(
            "use_multithread",
            True
        )
        
        # 初始化分词组件
        if self.config["tokenization"]["use_mwe"]:
            print("初始化多词短语提取器...")
            self.mwe_phrases = self._extract_mwe_from_wordnet()
            self.sorted_phrases = self._sort_phrases_by_length(self.mwe_phrases)
        
        # 初始化规范化组件
        print("初始化停用词集合...")
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            print(f"加载停用词失败: {e}")
            print("使用空停用词集合")
            self.stop_words = set()
        
        if self.config["normalization"]["use_lemmatization"]:
            self.lemmatizer = WordNetLemmatizer()
        else:
            self.lemmatizer = None
            
        if self.config["normalization"]["use_stemming"]:
            self.stemmer = PorterStemmer()
        else:
            self.stemmer = None
    
    # ============ XML 解析功能 ============
    
    def _clean_html_tags(self, text: str) -> str:
        """
        清理文本中的 HTML 标签
        """
        text = re.sub(r'<br\s*/?>', '\n', text)
        text = re.sub(r'<p>', '\n', text)
        text = re.sub(r'</p>', '\n', text)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)
        return text.strip()
    
    def _extract_description(self, xml_root: ET.Element) -> Optional[str]:
        """
        从 XML 根元素中提取 Description 内容
        """
        try:
            description_element = xml_root.find('description')
            if description_element is not None and description_element.text:
                decoded_text = html.unescape(description_element.text)
                clean_text = self._clean_html_tags(decoded_text)
                return clean_text
            return None
        except Exception as e:
            print(f"提取 Description 失败: {e}")
            return None
    
    def _extract_event_info(self, xml_root: ET.Element) -> Dict:
        """
        提取事件的基本信息
        """
        info = {'id': '', 'name': '', 'time': '', 'venue_name': ''}
        
        try:
            id_element = xml_root.find('id')
            if id_element is not None and id_element.text:
                info['id'] = id_element.text
            
            name_element = xml_root.find('name')
            if name_element is not None and name_element.text:
                info['name'] = name_element.text
            
            time_element = xml_root.find('time')
            if time_element is not None and time_element.text:
                info['time'] = time_element.text
            
            venue_element = xml_root.find('venue/name')
            if venue_element is not None and venue_element.text:
                info['venue_name'] = venue_element.text
                
        except Exception as e:
            print(f"提取事件信息失败: {e}")
        
        return info
    
    def parse_xml_file(self, file_path: str) -> Optional[Dict]:
        """
        解析单个 XML 文件
        """
        try:
            tree = ET.parse(file_path)
            root = tree.getroot()
            
            description = self._extract_description(root)
            event_info = self._extract_event_info(root)
            
            if description is None:
                return None
            
            return {
                'file_name': os.path.basename(file_path),
                'event_info': event_info,
                'description': description
            }
        except Exception as e:
            print(f"解析 XML 文件失败: {file_path}, 错误: {e}")
            return None
    
    def parse_xml_files(self, input_dir: str) -> List[Dict]:
        """
        批量解析 XML 文件
        """
        xml_files = []
        
        if not os.path.exists(input_dir):
            print(f"目录不存在: {input_dir}")
            return xml_files
        
        for file_name in os.listdir(input_dir):
            if file_name.startswith('PastEvent') and file_name.lower().endswith('.xml'):
                file_path = os.path.join(input_dir, file_name)
                xml_files.append(file_path)
        
        # 限制处理数量
        max_files = self.config["input"]["max_files"]
        if len(xml_files) > max_files:
            print(f"限制处理前 {max_files} 个文件")
            xml_files = xml_files[:max_files]
        
        # 根据配置决定是否使用多线程
        if self.use_multithread and len(xml_files) > 1:
            return self._parse_xml_files_parallel(xml_files)
        
        # 单线程处理
        results = []
        for file_path in xml_files:
            result = self.parse_xml_file(file_path)
            if result:
                results.append(result)
                print(f"✓ 成功解析: {os.path.basename(file_path)}")
            else:
                print(f"✗ 跳过文件: {os.path.basename(file_path)}")
        
        return results
    
    def _parse_xml_files_parallel(self, xml_files: List[str]) -> List[Dict]:
        """
        使用多进程并行解析 XML 文件
        """
        results = []
        print(f"使用 {self.max_workers} 个进程并行解析...")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_file = {
                executor.submit(self.parse_xml_file, fp): fp 
                for fp in xml_files
            }
            
            # 收集结果
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    result = future.result()
                    if result:
                        results.append(result)
                        print(f"✓ 成功解析: {os.path.basename(file_path)}")
                    else:
                        print(f"✗ 跳过文件: {os.path.basename(file_path)}")
                except Exception as e:
                    print(f"✗ 处理失败 {os.path.basename(file_path)}: {e}")
        
        return results
    
    # ============ 分词功能 ============
    
    def _extract_mwe_from_wordnet(self) -> Set[str]:
        """
        从 WordNet 中提取多词短语
        """
        print("正在从 WordNet 提取多词短语...")
        mwe_set = set()
        
        try:
            for synset in wn.all_synsets():
                for lemma in synset.lemmas():
                    lemma_name = lemma.name()
                    if '_' in lemma_name:
                        phrase = lemma_name.replace('_', ' ').lower()
                        mwe_set.add(phrase)
            
            print(f"成功提取 {len(mwe_set)} 个多词短语")
        except Exception as e:
            print(f"从 WordNet 提取多词短语失败: {e}")
            print("使用空的多词短语集合")
            mwe_set = set()
        
        return mwe_set
    
    def _sort_phrases_by_length(self, phrases: Set[str]) -> List[str]:
        """
        将短语按词数排序
        """
        return sorted(phrases, key=lambda x: (-len(x.split()), x))
    
    def _merge_phrases_in_text(self, text: str, phrases: List[str]) -> str:
        """
        在文本中合并多词短语
        """
        merged_text = text.lower()
        
        for phrase in phrases:
            pattern = r'\b' + re.escape(phrase) + r'\b'
            replacement = phrase.replace(' ', '_')
            merged_text = re.sub(pattern, replacement, merged_text)
        
        return merged_text
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        对文本进行分词
        """
        if self.config["tokenization"]["use_mwe"]:
            merged_text = self._merge_phrases_in_text(text, self.sorted_phrases)
        else:
            merged_text = text.lower()
        
        tokens = word_tokenize(merged_text)
        return tokens
    
    # ============ Token 规范化功能 ============
    
    def _is_punctuation(self, token: str) -> bool:
        """
        判断 token 是否为标点符号
        """
        if '_' in token:
            return False
        
        if re.match(r'^[^\w\s]+$', token, re.UNICODE):
            return True
        
        if re.search(r'[^\w\s]', token, re.UNICODE):
            return True
        
        return False
    
    def _is_number(self, token: str) -> bool:
        """
        判断 token 是否为数字
        """
        return bool(re.search(r'\d', token))
    
    def _is_url_or_special(self, token: str) -> bool:
        """
        判断 token 是否为 URL 或特殊格式
        """
        url_patterns = [
            r'http', r'https', r'www\.', r'\.com', r'\.org', r'\.net',
            r'\.html', r'\.php', r'@', r'//'
        ]
        
        for pattern in url_patterns:
            if re.search(pattern, token, re.IGNORECASE):
                return True
        
        return False
    
    def _is_only_underscores(self, token: str) -> bool:
        """
        判断 token 是否只包含下划线或无效字符（没有有效字母或数字）
        """
        # 移除所有下划线、空白字符和标点符号后，如果没有任何字母或数字，则视为无效
        cleaned = re.sub(r'[_\s\'\-\.\,\;\:\!\?]+', '', token)
        # 检查是否还有字母或数字
        if len(cleaned) == 0:
            return True
        # 如果没有字母或数字字符，也算作无效
        return not bool(re.search(r'[a-zA-Z0-9]', cleaned))
    
    def _strip_prefix_punctuation(self, token: str) -> str:
        """
        移除token开头和结尾的标点符号和下划线
        """
        # 移除开头的单引号、减号和下划线
        token = re.sub(r'^[\'\-_]+', '', token)
        # 移除结尾的单引号、减号和下划线
        token = re.sub(r'[\'\-_]+$', '', token)
        return token
    
    def _get_wordnet_pos(self, treebank_tag: str) -> str:
        """
        将 Penn Treebank 词性标签转换为 WordNet 词性标签
        """
        if treebank_tag.startswith('J'):
            return wn.ADJ
        elif treebank_tag.startswith('V'):
            return wn.VERB
        elif treebank_tag.startswith('N'):
            return wn.NOUN
        elif treebank_tag.startswith('R'):
            return wn.ADV
        else:
            return wn.NOUN
    
    def filter_tokens(self, tokens: List[str]) -> Tuple[List[str], Dict]:
        """
        过滤 token
        """
        filtered_tokens = []
        stats = {
            'original_count': len(tokens),
            'removed_stopwords': 0,
            'removed_numbers': 0,
            'removed_punctuation': 0,
            'removed_short': 0,
            'removed_special': 0,
            'removed_underscores_only': 0,
            'kept_count': 0
        }
        
        config = self.config["normalization"]
        
        for token in tokens:
            # 移除只有下划线的token（没有实际字母或数字）
            if self._is_only_underscores(token):
                stats['removed_underscores_only'] += 1
                continue
            
            # 移除 URL 和特殊格式
            if self._is_url_or_special(token):
                stats['removed_special'] += 1
                continue
            
            # 移除标点符号
            if config["remove_punctuation"] and self._is_punctuation(token):
                stats['removed_punctuation'] += 1
                continue
            
            # 移除包含数字的词
            if config["remove_numbers"] and self._is_number(token):
                stats['removed_numbers'] += 1
                continue
            
            # 移除停用词（保留短语 token）
            if config["remove_stopwords"] and '_' not in token and token.lower() in self.stop_words:
                stats['removed_stopwords'] += 1
                continue
            
            # 移除过短的词
            if config["remove_short_words"] and len(token) < config["min_word_length"]:
                stats['removed_short'] += 1
                continue
            
            filtered_tokens.append(token)
        
        stats['kept_count'] = len(filtered_tokens)
        return filtered_tokens, stats
    
    def normalize_tokens(self, tokens: List[str]) -> Tuple[List[str], Dict]:
        """
        规范化 token
        """
        normalized_tokens = []
        config = self.config["normalization"]
        
        # 词性标注（用于词形还原）
        if config["use_lemmatization"]:
            tagged_tokens = pos_tag(tokens)
        else:
            tagged_tokens = [(token, None) for token in tokens]
        
        stats = {
            'original_count': len(tokens),
            'changed_count': 0,
            'unchanged_count': 0
        }
        
        for token, pos in tagged_tokens:
            original_token = token
            
            # 移除开头的单引号和减号
            token = self._strip_prefix_punctuation(token)
            
            # 如果移除前缀后token为空，跳过
            if not token:
                continue
            
            # 如果是短语 token，保持不变（但已去掉前缀）
            if '_' in token:
                normalized_tokens.append(token)
                if token != original_token:
                    stats['changed_count'] += 1
                else:
                    stats['unchanged_count'] += 1
                continue
            
            # 词形还原
            if config["use_lemmatization"] and self.lemmatizer:
                wordnet_pos = self._get_wordnet_pos(pos) if pos else wn.NOUN
                token = self.lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
            
            # 词干提取
            elif config["use_stemming"] and self.stemmer:
                token = self.stemmer.stem(token.lower())
            
            # 转为小写
            else:
                token = token.lower()
            
            normalized_tokens.append(token)
            
            if token != original_token.lower():
                stats['changed_count'] += 1
            else:
                stats['unchanged_count'] += 1
        
        return normalized_tokens, stats
    
    # ============ 文件操作功能 ============
    
    def _clear_directory(self, directory: str):
        """
        清空目录
        """
        if os.path.exists(directory):
            shutil.rmtree(directory)
        os.makedirs(directory, exist_ok=True)
    
    def _save_tokens_to_file(self, tokens: List[str], output_path: str):
        """
        保存 token 到文件
        """
        try:
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(tokens))
            return True
        except Exception as e:
            print(f"保存文件失败: {output_path}, 错误: {e}")
            return False
    
    def _save_phrase_dict(self, phrases: List[str], output_path: str):
        """
        保存短语词典
        """
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write("\n".join(phrases))
            return True
        except Exception as e:
            print(f"保存短语词典失败: {e}")
            return False
    
    # ============ 主要处理流程 ============
    
    def process_single_event(self, event_data: Dict) -> Tuple[List[str], Dict]:
        """
        处理单个事件的完整流程
        
        参数:
            event_data: 事件数据字典
            
        返回:
            (规范化后的 tokens, 统计信息)
        """
        description = event_data['description']
        
        # 1. 分词
        tokens = self.tokenize_text(description)
        
        # 2. 过滤
        filtered_tokens, filter_stats = self.filter_tokens(tokens)
        
        # 3. 规范化
        normalized_tokens, normalize_stats = self.normalize_tokens(filtered_tokens)
        
        # 合并统计信息
        stats = {
            'filter_stats': filter_stats,
            'normalize_stats': normalize_stats,
            'final_token_count': len(normalized_tokens)
        }
        
        return normalized_tokens, stats
    
    def _process_and_save_event(self, result: Dict, normalized_dir: str) -> Tuple[bool, Dict]:
        """
        处理事件并保存到文件（用于多线程）
        """
        event_id = result['event_info']['id']
        if not event_id:
            return False, {}
        
        # 处理事件
        normalized_tokens, stats = self.process_single_event(result)
        
        # 保存结果
        output_path = os.path.join(normalized_dir, f"{event_id}.txt")
        success = self._save_tokens_to_file(normalized_tokens, output_path)
        
        return success, stats
    
    def _process_events_parallel(self, parsed_results: List[Dict], normalized_dir: str) -> Tuple[int, Dict]:
        """
        使用多进程并行处理事件
        """
        success_count = 0
        total_stats = {
            'original_tokens': 0,
            'final_tokens': 0,
            'events_processed': 0
        }
        
        print(f"使用 {self.max_workers} 个进程并行处理...")
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_event = {
                executor.submit(self._process_and_save_event, result, normalized_dir): result
                for result in parsed_results
            }
            
            for future in as_completed(future_to_event):
                result = future_to_event[future]
                try:
                    success, stats = future.result()
                    if success:
                        success_count += 1
                        total_stats['original_tokens'] += stats['filter_stats']['original_count']
                        total_stats['final_tokens'] += stats['final_token_count']
                        total_stats['events_processed'] += 1
                except Exception as e:
                    event_id = result['event_info'].get('id', 'unknown')
                    print(f"✗ 处理事件失败 {event_id}: {e}")
        
        return success_count, total_stats
    
    def run_pipeline(self) -> Dict:
        """
        运行完整的处理流水线
        
        返回:
            处理结果统计
        """
        print("=" * 80)
        print("事件文本处理流水线")
        print("=" * 80)
        
        start_time = time.time()
        
        # 获取输出目录
        base_dir = self.config["output"]["base_dir"]
        past_events_dir = os.path.join(base_dir, self.config["output"]["past_events_dir"])
        tokenized_dir = os.path.join(base_dir, self.config["output"]["tokenized_events_dir"])
        normalized_dir = os.path.join(base_dir, self.config["output"]["normalized_events_dir"])
        
        # 清空输出目录
        print("清空输出目录...")
        self._clear_directory(past_events_dir)
        self._clear_directory(tokenized_dir)
        self._clear_directory(normalized_dir)
        
        # 步骤1: 解析 XML
        print("\n步骤1: 解析 XML 文件...")
        input_dir = self.config["input"]["xml_dir"]
        parsed_results = self.parse_xml_files(input_dir)
        
        if not parsed_results:
            print("错误: 没有成功解析任何文件")
            return {}
        
        print(f"成功解析 {len(parsed_results)} 个文件")
        
        # 保存解析结果
        print("\n保存解析结果...")
        for result in parsed_results:
            event_id = result['event_info']['id']
            description = result['description']
            
            if event_id:
                output_path = os.path.join(past_events_dir, f"{event_id}.txt")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(description)
        
        # 步骤2: 分词
        print("\n步骤2: 文本分词...")
        if self.config["tokenization"]["use_mwe"]:
            phrase_dict_path = os.path.join(tokenized_dir, self.config["tokenization"]["phrase_dict_file"])
            self._save_phrase_dict(self.sorted_phrases, phrase_dict_path)
        
        # 步骤3: 规范化处理
        print("\n步骤3: Token 规范化...")
        
        # 根据配置决定是否使用多线程
        if self.use_multithread and len(parsed_results) > 1:
            success_count, total_stats = self._process_events_parallel(parsed_results, normalized_dir)
        else:
            # 单线程处理
            success_count = 0
            total_stats = {
                'original_tokens': 0,
                'final_tokens': 0,
                'events_processed': 0
            }
            
            for result in parsed_results:
                event_id = result['event_info']['id']
                if not event_id:
                    continue
                
                # 处理事件
                normalized_tokens, stats = self.process_single_event(result)
                
                # 保存结果
                output_path = os.path.join(normalized_dir, f"{event_id}.txt")
                if self._save_tokens_to_file(normalized_tokens, output_path):
                    success_count += 1
                    total_stats['original_tokens'] += stats['filter_stats']['original_count']
                    total_stats['final_tokens'] += stats['final_token_count']
                    total_stats['events_processed'] += 1
        
        # 计算执行时间
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        # 生成结果统计
        result_stats = {
            'success_count': success_count,
            'total_files': len(parsed_results),
            'elapsed_time': elapsed_time,
            'token_stats': total_stats,
            'output_dirs': {
                'past_events': past_events_dir,
                'tokenized': tokenized_dir,
                'normalized': normalized_dir
            }
        }
        
        # 打印结果
        print("\n" + "=" * 80)
        print("处理完成!")
        print("=" * 80)
        print(f"成功处理: {success_count}/{len(parsed_results)} 个事件")
        print(f"执行时间: {elapsed_time:.2f} 秒")
        print(f"Token 统计: {total_stats['original_tokens']} → {total_stats['final_tokens']}")
        if total_stats['original_tokens'] > 0:
            reduction_rate = (1 - total_stats['final_tokens'] / total_stats['original_tokens']) * 100
            print(f"Token 减少率: {reduction_rate:.1f}%")
        
        return result_stats
    
    def get_config(self) -> Dict:
        """
        获取当前配置
        """
        return self.config
    
    def update_config(self, new_config: Dict):
        """
        更新配置
        """
        self.config.update(new_config)
        self._initialize_components()
