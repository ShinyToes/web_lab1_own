# Web实验一 信息检索部分

###### 小组成员:
###### 秦志强 PB23111621
###### 索子骁 PB23111584
###### 李雪康 PB23111576
<br>
<br>
<br>
<br>
<br>

## 1.文档解析与规范化处理(event_text_processor)

关键设计：

长度优先匹配：将短语按词数降序排序，避免部分匹配导致的错误分割

WordNet集成：从WordNet中提取所有包含下划线的lemma作为多词短语库

正则边界保护：使用 \b 边界匹配符确保完整词匹配，避免误匹配

下划线连接：识别的短语用下划线连接，后续处理中保持不变

关于词性标注与还原的实现:
```python
#### 转换Penn Treebank标签到WordNet标签

def _get_wordnet_pos(self, treebank_tag: str) -> str:
    # 'VBD' → VERB, 'NNS' → NOUN, 'JJ' → ADJ, 'RB' → ADV
 

#### 词形还原时使用词性信息
wordnet_pos = self._get_wordnet_pos(pos)
token = self.lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
```

NLTK的pos_tag返回Penn Treebank标签，WordNetLemmatizer需要WordNet标签，需要转换，并且需要注意同一词在不同词性下还原结果不同（如 "better" 作形容词还原为 "good"，作副词还原为 "well"），对于无法识别的词性，默认按名词处理，因为名词是最常见内容词。

### 核心功能

**XML 解析与文本分词**：使用 `xml.etree.ElementTree` 解析 XML 文件，通过 `nltk.tokenize.word_tokenize()` 分词，支持多词短语识别（用下划线连接）。

**Token 过滤与规范化**：结合 `nltk.corpus.stopwords` 过滤停用词和标点，使用 `nltk.pos_tag()` 进行词性标注，通过 `WordNetLemmatizer` 进行词形还原。

**并行处理**：采用 `ProcessPoolExecutor` 实现多进程并行处理，提升大规模数据处理效率。

<br>
<br>
<br>
<br>

## 2.倒排表的构建(invertd_index_builder)

### 构建流程

采用三步法构建索引：(1) 构建临时索引（提取 `<词项, 文档ID>` 对）→ (2) 排序与合并（按词项排序并合并倒排列表）→ (3) 生成索引文件（支持文本、二进制、压缩格式）。

#### 三步构建法:

```python
  步骤1：构建临时索引（第129-171行）
term_doc_pairs = []  # 内存中的<词项, 文档ID>对
for document in documents:
    for token in document:
        term_doc_pairs.append((token, doc_id))

  步骤2：排序（第173-193行）
term_doc_pairs.sort(key=lambda x: x[0])  # 按词项排序

  步骤3：合并（第195-250行）
  相同词项的文档ID合并成倒排列表
inverted_index = defaultdict(list)
for term, doc_id in term_doc_pairs:
    inverted_index[term].append(doc_id)
```

在合并阶段才去重和排序（第227-231行），避免重复处理,文档ID按数值升序排序而非字符串排序，确保后续Gap编码有效.

#### VB编码算法:

```python
def vb_encode_number(self, n: int) -> bytes:
    """VB编码核心算法"""
    if n == 0:
        return bytes([128])  # 特殊处理0
    
    bytes_list = []
    while n > 0:
        bytes_list.insert(0, n % 128)  # 取低7位
        n //= 128
    
    bytes_list[-1] += 128  # 最后一个字节最高位设为1（标记结束）
    return bytes(bytes_list)
```

文档ID在1-1000范围内时，VB编码平均节省50-70%空间。


#### Gap编码

```python
def encode_gaps(self, numbers: List[int]) -> List[int]:
    """将绝对值转换为差值"""
    if not numbers:
        return []
    
    gaps = [numbers[0]]  # 第一个保持绝对值
    for i in range(1, len(numbers)):
        gaps.append(numbers[i] - numbers[i-1])  # 后续存差值
    
    return gaps
```

Gap编码与VB编码组合，将文档ID序列转换为小差值，进一步提升压缩效果

#### K-间隔指针词典压缩:

```python
# 1. 构建单一字符串（第660-668行）
single_string = ""
for term in dictionary:
    single_string += term  # 所有词项拼接
    # "apple" + "banana" + "cat" → "applebanaracat"

# 2. K-间隔存储策略（第677行）
store_pointer = ((idx + 1) % K == 0) or (idx == last)
# 每K个词项存储完整指针，中间只存长度

# 3. 元数据结构
if store_pointer:
    存储: [8字节前缀][3字节指针][4字节频率][4字节倒排表指针]
else:
    存储: [8字节前缀][1字节长度][4字节频率][4字节倒排表指针]

```
为平衡空间和查找速度，K一般取3-5

#### 二进制存储格式
定长词典格式:使用 `struct.pack('I', doc_id)` 将文档ID打包为4字节无符号整数，支持快速随机访问。

```python
# 词典每项固定大小（第52行计算）
entry_size = term_max_length(20) + doc_freq_size(4) + pointer_size(4) = 28字节

# 写入格式（第342-356行）
term_bytes = term.encode('utf-8').ljust(20, b'\x00')  # 20字节，不足补0
f.write(term_bytes)                    # 词项（定长）
f.write(struct.pack('I', doc_freq))   # 文档频率（4字节无符号整数）
f.write(struct.pack('I', pointer))    # 倒排表指针（4字节无符号整数）
```

可在词典上进行二分查找，时间复杂度O(log n)，且无需顺序读取整个词典,不过有截断风险，超过20字节的词项被截断


<br>
<br>
<br>
<br>

## 3.布尔检索(boolean_search)
操作符优先级：NOT > AND > OR：

```python
def execute_query_expression_with_skip(query, ...):
    # 1. 去除外层括号
    if query.startswith('(') and query.endswith(')'):
        query = query[1:-1].strip()
    
    # 2. 处理OR（最低优先级）
    or_parts = split_by_operator(query, 'OR')
    if len(or_parts) > 1:
        results = [递归处理每个部分]
        return union(results)
    
    # 3. 处理AND（中等优先级）
    and_parts = split_by_operator(query, 'AND')
    if len(and_parts) > 1:
        results = [递归处理每个部分]
        return intersect_with_skip(results)  # 使用跳表指针
    
    # 4. 处理NOT（最高优先级）
    if query.startswith('NOT '):
        return complement(...)
    
    # 5. 基本词项
    return get_term_postings(query)
```

考虑括号嵌套，作了括号感知的操作符分割：

```python
def split_by_operator(query: str, operator: str) -> List[str]:
    """按操作符分割查询字符串（考虑括号嵌套）"""
    paren_depth = 0
    
    for i, char in enumerate(query):
        if char == '(':
            paren_depth += 1
        elif char == ')':
            paren_depth -= 1
        elif paren_depth == 0 and query[i:] startswith(f' {operator} '):
            # 只在括号外层分割
            parts.append(current)
```

此外，作了实现了文件句柄缓存机制的优化，避免重复打开文件，显著提升了多次查询的性能。

<br>
<br>

### 不同处理顺序对时间开支的影响

#### 优化策略

##### 基线策略：按查询表达式从左到右依次执行，不考虑 posting list 长度差异。

##### 优化策略：AND/OR 操作按 posting list 长度升序排序操作数，优先处理短列表，减少中间结果规模。

同一个布尔表达式的不同处理顺序对时间开支的影响结果如下:
![](bool_compare.png)

#### 实验结果

| 查询表达式 | 从左到右 | 优化策略 | 性能提升 | 结果数 |
|-----------|---------|---------|---------|-------|
| (ability AND able) OR accept | 0.000110s | 0.000090s | +18.30% | 134 |
| adventure AND (age OR alert) | 0.000102s | 0.000084s | +18.46% | 3 |
| (alone OR amazing) AND NOT activity | 0.001691s | 0.001079s | +36.19% | 98 |
| (allow AND also) OR (alternative AND although) | 0.000050s | 0.000043s | +15.14% | 0 |

**关键结论**：posting list 长度差异越大，优化效果越明显（最高达 36.19%）。对于包含多个 AND 操作的查询，优化效果最显著。优化策略不改变最终结果，所有查询均在毫秒级完成。

---

<br>
<br>
<br>
<br>

## 4.倒排表的扩展与优化
### 位置信息扩展(build_inverted_index_pos)

```python
# 数据结构：{词项: {文档ID: [位置列表]}}
inverted_index = defaultdict(lambda: defaultdict(list))

# 记录位置信息（位置从1开始）
for position, token in enumerate(tokens, 1):
    inverted_index[token][doc_id].append(position)
```

记录词项在每个文档中的所有位置，支持精确短语检索和邻近检索

### 存储效率分析

| 索引类型 | 倒排表大小 | 相对标准索引 | 特点 |
|---------|----------|------------|------|
| 标准倒排索引 | 1.25 MB | 基准（1.0x） | 仅记录文档ID |
| 位置型索引（未压缩） | 3528 KB | 2.76倍 | 记录所有位置信息 |
| 位置型索引（压缩） | 1635 KB | 1.28倍 | VB+Gap双层压缩 |

采用 VB+Gap 双层压缩（文档ID序列和位置序列分别压缩），位置型倒排表压缩率达 53.65%，压缩后仅比标准索引大 28%，空间开销可控。

<br>
<br>

### 索引压缩方法与压缩效果(compression_search_comparison)
与2相同，使用了VB+Gap编码以及K-间隔指针词典压缩.
比较索引压缩前后在检索效率上的差异的结果如下:

#### 空间节省对比：
<table>
<tr>
<td><img src="inverted.png" alt="未压缩倒排索引" width="400"/></td>
<td><img src="inverted_compress.png" alt="压缩倒排索引" width="400"/></td>
</tr>
<tr>
<td><img src="diction.png" alt="未压缩词典" width="400"/></td>
<td><img src="diction_compress.png" alt="压缩词典" width="400"/></td>
</tr>
</table>


| 对比项 | 未压缩大小 | 压缩后大小 | 压缩率 | 节省空间 |
|-------|----------|----------|-------|---------|
| 倒排表 | 1.25 MB | 803 KB | 35.76% | 447 KB |
| 词典 | 699 KB | 628 KB | 10.16% | 71 KB |
| **总体** | **1.949 MB** | **1.431 MB** | **26.58%** | **518 KB** |

倒排表使用 VB+Gap 编码压缩效果显著，词典使用 K-间隔指针压缩节省约 10.16% 空间。

#### 查询性能对比

##### 单词查询测试（高频词）：

![](compress_compare.png)
| 查询词 | 未压缩时间 | 压缩时间 | 时间差异 | 文档数 |
|-------|----------|---------|---------|-------|
| please | 0.000587s | 0.001108s | +88.74% | 1745 |
| u | 0.000499s | 0.000883s | +76.96% | 1666 |
| see | 0.000488s | 0.000781s | +59.89% | 1406 |

##### AND 查询测试：

| 查询表达式 | 未压缩时间 | 压缩时间 | 时间差异 | 文档数 |
|-----------|----------|---------|---------|-------|
| please AND u | 0.001840s | 0.002508s | +36.28% | 824 |
| see AND meet | 0.001400s | 0.001921s | +37.21% | 542 |
| u AND meet | 0.001445s | 0.002624s | +81.55% | 574 |
| please AND u AND see | 0.002475s | 0.003574s | +44.41% | 391 |
| u AND see AND meet | 0.002313s | 0.003463s | +49.71% | 286 |

##### OR 查询测试：

| 查询表达式 | 未压缩时间 | 压缩时间 | 时间差异 | 文档数 |
|-----------|----------|---------|---------|-------|
| please OR u | 0.001631s | 0.002387s | +46.28% | 2587 |
| see OR meet | 0.001662s | 0.002359s | +42.00% | 2241 |
| please OR u OR see | 0.003208s | 0.004499s | +40.23% | 3037 |

<br>
<br>

### 短语检索系统(phrase_search):
#### 二进制位置索引加载部分:

```python
def load_positional_posting_list(...) -> Dict[int, List[int]]:
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

```
通过使用 f.seek(pointer) 直接定位到倒排表，位置用 unsigned short（2字节）存储，节省空间，可以自动排序位置列表，为后续算法提供便利

#### 精确短语匹配算法：

```python
def phrase_query(...) -> Tuple[Set[int], float]:
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
```
遍历第一个词的所有出现位置,对每个起始位置，验证后续词是否在 start_pos + i 位置,使用 in 运算符快速检查（列表成员测试）
双层循环遍历两个词的位置列表,使用 abs() 计算距离，支持任意顺序,找到一个匹配即可提前退出（双重break）

测试结果如下:
![](phase.png)

<br>
<br>

### 带跳表指针的倒排表的构建(inverted_index_builder_with_skip):
核心算法部分:

```python
def calculate_skip_pointers(self, doc_ids: List[int], file_positions: List[int] = None):
    """计算跳表指针位置（间隔√L均匀放置）"""
    
    L = len(doc_ids)
    if L < self.min_list_length:
        return []  # 太短的列表不需要跳表指针
    
    # 关键：跳表步长 = √L
    skip_step = int(math.sqrt(L))
    
    skip_pointers = []
    i = skip_step - 1  # 从第 skip_step 个位置开始
    
    while i < L - 1:
        target_index = min(i + skip_step, L - 1)
        target_doc_id = doc_ids[target_index]
        target_file_pos = file_positions[target_index] if file_positions else -1
        
        skip_pointers.append((i, target_doc_id, target_file_pos))
        i += skip_step
    
    return skip_pointers
```
以步长√L性能较为平衡稳妥
同时在inverted_index_config_with_skip文件里面可以自选模式:

```python
"skip_pointer": {
    "enabled": true,
    "skip_strategy": "sqrt",
    "skip_step": 10,
    "skip_marker_binary": 4294967295,
    "skip_marker_compressed": 253,
    "min_list_length": 3,
  


"_comment": {
      "skip_strategy": "可选值: 'sqrt', 'log', 'linear', 'cbrt', 'fixed'(使用skip_step)",
      "skip_step": "当skip_strategy='fixed'时使用,null表示固定",
      "示例": {
        "sqrt": "skip_strategy='sqrt', skip_step=null",
        "log": "skip_strategy='log', skip_step=null",
        "fixed_10": "skip_strategy='fixed', skip_step=10",
        "linear": "skip_strategy='linear', skip_step=null"
      }
    }
  }

```

#### 优化实际效果:

##### 存储效率:

选择不同的跳表指针步长后，对存储效率的影响结果如下:
<table>
<tr>
<td><img src="skip_compress_10000files_fixed.png" alt="fixed策略" width="400"/></td>
<td><img src="skip_compress_10000files_sqrt.png" alt="sqrt策略" width="400"/></td>
</tr>
</table>

对比发现，fixed固定步长的比sqrt模式压缩效果更好，原因可能更频繁的跳表指针作为"分隔符"，防止Gap值累积过大，而小Gap值的VB编码效率极高，导致跳表标记开销 < 大Gap值VB编码开销

##### 布尔检索性能:

选择不同的跳表指针步长后，对性能的影响结果如下:
<table>
<tr>
<td><img src="boolean_skip_10000files_fixed.png" alt="fixed策略" width="400"/></td>
<td><img src="boolean_skip_10000files_sqrt.png" alt="sqrt策略" width="400"/></td>
</tr>
</table>

对比发现，fixed固定步长的比sqrt模式节省更多比较次数，原因可能是测试查询里高频词为主，且AND查询比较复杂，密集的跳表指针在交集操作中优势更大.

<br>
<br>
<br>
<br>


## 5.向量空间模型文档检索
### TF-IDF 计算方法(vector_space_retrieval)

系统实现四种 TF-IDF 计算变体，支持程序员通过参数灵活选择：

| 方法 | 公式 | 特点 |
|------|------|------|
| 标准方法 | `tf × log(N/df)` | 默认方法，直接使用词频 |
| 对数频率 | `(1 + log(tf)) × log(N/df)` | 对高频词进行平滑，避免过度权重 |
| 增强频率 | `(0.5 + 0.5×tf/max_tf) × log(N/df)` | 规范化词频至 [0.5, 1.0]，平衡长短文档 |
| 二元频率 | `(1 if tf>0 else 0) × log(N/df)` | 仅考虑词项出现与否，忽略频率差异 |


```python
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
```
<br>
<br>

### 余弦相似度计算

```python
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
```

点积query_vector · doc_vector 用来衡量向量重合程度，取模长：||query_vector|| 和 ||doc_vector|| 作为向量长度，通过除以两个模长的乘积归一化消除长度影响

### 查询模式

**测试模式**：自动执行 10 个预定义查询（music, food, art, technology innovation, education, health, business entrepreneurship, travel adventure, science research discovery, culture tradition heritage），生成详细检索报告和统计信息。

**交互式模式**：用户输入自定义查询，实时显示检索结果，支持动态切换 TF-IDF 方法（命令：`method log`），提供配置查看（`config`）和方法列表（`methods`）等帮助功能。

### 检索结果示例

**查询**：`music`

**Top-3 结果**：

| 排名 | 文档ID | 相似度 | 文档长度 | 匹配词项 | 规范化原文摘要 |
|------|--------|--------|----------|----------|----------------|
| 1 | 11095502 | 0.6346 | 277 词项 | 1/1 (music) | ali amr music festival addition perform group ali amr experiment... |
| 2 | 112001832 | 0.5461 | 29 词项 | 1/1 (music) | bar new_haven live music every wednesday night... |
| 3 | 110826922 | 0.5296 | 48 词项 | 1/1 (music) | monday morning music meetup look meet parent make friend... |

**输出文件结构**：

```text
output_retrieval/
├── query_results/
│   ├── query_1_results.json          # 结构化数据（含 normalized_text）
│   ├── query_1_results.txt           # 可读文本格式（含规范化原文）
│   └── ...                           # 其他查询结果
├── retrieval_report.txt              # 综合报告（系统统计、查询统计、词项分析）
└── retrieval_statistics.json         # 统计信息（相似度分布、查询覆盖率）
```

### 检索性能评估

**查询响应时间**：所有查询均在毫秒级完成，包括加载索引、构建向量、计算相似度、排序等全流程。

**文档覆盖率**：对于单词查询（如 `music`），系统找到 10,514 个相关文档（全部文档），返回 Top-3 最相关结果。相似度范围 0.53-0.63，表明存在明确相关文档。

**词项匹配精度**：系统通过预处理保持查询与文档的一致性（停用词过滤、词形还原），查询词 `music` 精确匹配文档中的 `music` 词项，避免词形变化导致的匹配失败。

### 应用建议

**适用场景**：需要按相关度排序的检索任务（与布尔检索的二值判断不同），支持自然语言查询（多词短语），适合推荐系统、问答系统、文档相似度分析。

**方法选择**：通用场景使用标准方法，长文档场景使用增强频率方法（平衡长度差异），高频词过度权重场景使用对数频率方法，专有名词检索使用二元方法（强调出现与否）。

**性能优化**：预构建文档向量（避免重复计算），使用近似最近邻算法（如 LSH、HNSW）加速大规模检索，对高频词设置权重上限（防止"the"等词主导相似度）。

---


