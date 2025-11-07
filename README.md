# Web Lab 1 - 倒排索引构建与搜索系统

## 项目简介

这是一个基于 Python 的倒排索引构建与搜索系统，用于处理文本数据并实现高效的搜索功能。系统包含完整的文本处理流水线、倒排索引构建和基于向量空间模型的文档检索功能。

## 项目结构

```
web_lab1/
├── src/                           # 源代码目录
│   ├── inverted_index_builder.py              # 倒排索引构建器（基础版）
│   ├── inverted_index_builder_with_skip.py    # 倒排索引构建器（带跳表指针）
│   ├── build_inverted_index_pos.py            # 位置信息倒排索引构建（短语/邻近检索）
│   ├── event_text_processor.py                # 事件文本处理器
│   └── vector_space_retrieval.py              # 向量空间模型检索系统
├── scripts/                       # 运行脚本目录
│   ├── run_processor.py                           # 文本处理脚本
│   ├── run_inverted_index_builder.py              # 倒排索引构建（基础版）
│   ├── run_inverted_index_builder_with_skip.py    # 倒排索引构建（带跳表指针）
│   ├── run_retrieval.py                           # 文档检索脚本（VSM/TF-IDF）
│   ├── boolean_search.py                          # 布尔检索效率实验
│   ├── phrase_search.py                           # 短语/邻近检索实验（位置索引）
│   └── compression_search_comparison.py           # 压缩前后检索效率与空间对比
├── config/                        # 配置文件目录
│   ├── inverted_index_config.json             # 倒排索引配置（基础版）
│   ├── inverted_index_config_with_skip.json   # 倒排索引配置（带跳表指针）
│   ├── config_process.json            # 处理配置
│   └── retrieval_config.json          # 检索配置
└── README.md                      # 项目说明文档
```

## 主要功能

- **文本处理**: 对原始文本数据进行预处理和清洗
- **倒排索引构建**: 构建高效的倒排索引结构
- **向量空间模型检索**: 基于 TF-IDF 和余弦相似度的文档检索
- **布尔检索**: 支持 AND/OR/NOT 的布尔查询与策略优化对比
- **短语/邻近检索**: 基于位置信息索引实现精确短语与邻近搜索
- **跳表指针**: 在倒排表中插入跳表指针加速长列表合并
- **多种 TF-IDF 变体**: 支持标准、对数、增强频率、二元等计算方法
- **交互式查询**: 支持测试模式和交互式查询模式
- **详细报告生成**: 自动生成检索结果报告和统计信息
- **配置管理**: 通过 JSON 配置文件管理系统参数
- **压缩索引对比**: 对比未压缩与压缩（VB+Gap、单一字符串词典）在时空上的差异

## 使用方法

### 1. 文本处理
```bash
python scripts/run_processor.py
```

### 2. 构建倒排索引
```bash
# 基础版倒排索引
python scripts/run_inverted_index_builder.py

# 带跳表指针版本（若存在对应配置文件将优先使用）
python scripts/run_inverted_index_builder_with_skip.py
```

### 3. 文档检索

#### 测试模式（运行预定义查询）
```bash
python scripts/run_retrieval.py --mode test
```

#### 交互式模式（用户输入查询）
```bash
python scripts/run_retrieval.py --mode interactive
```

#### 自定义参数
```bash
# 使用对数频率 TF-IDF 方法
python scripts/run_retrieval.py --method log --top-k 20

# 使用自定义配置文件
python scripts/run_retrieval.py --config my_config.json
```

## 检索系统特性

### TF-IDF 计算方法

系统支持四种 TF-IDF 计算方法：

1. **标准方法** (`standard`): `tf × log(N/df)`
2. **对数频率** (`log`): `(1 + log(tf)) × log(N/df)`
3. **增强频率** (`augmented`): `(0.5 + 0.5×tf/max_tf) × log(N/df)`
4. **二元频率** (`binary`): `(1 if tf>0 else 0) × log(N/df)`

### 查询模式

#### 测试模式
- 自动执行预定义的测试查询
- 生成详细的检索报告
- 保存所有结果到文件

#### 交互式模式
- 支持用户输入自定义查询
- 实时显示检索结果
- 支持动态切换 TF-IDF 方法
- 提供帮助和配置查看功能

### 高级检索与实验

- **布尔检索效率实验**：比较从左到右执行与“优先处理短列表”策略。
  ```bash
  python scripts/boolean_search.py
  # 输出：output/boolean_search_results.txt
  ```

- **短语/邻近检索实验**：使用带位置信息的倒排索引进行精确短语与邻近匹配。
  ```bash
  python scripts/phrase_search.py
  # 输出：part_4/output/phrase_search_results.txt
  ```

- **索引压缩对比实验**：对比未压缩与压缩（VB+Gap、单一字符串词典）的空间与检索时间。
  ```bash
  python scripts/compression_search_comparison.py
  # 输出：part_4/output/compression_comparison_results.txt
  ```

### 输出文件

检索系统会生成以下输出文件：

```
output_retrieval/
├── query_results/                 # 查询结果目录
│   ├── query_1_results.json      # JSON 格式结果
│   ├── query_1_results.txt       # 可读文本格式
│   └── ...
├── retrieval_report.txt           # 详细检索报告
└── retrieval_statistics.json     # 统计信息
```

其他实验与构建输出（示例）：

```
output_inverted_index/
├── dictionary.bin                               # 未压缩词典（定长项）
├── inverted_index.bin                           # 未压缩倒排表
├── compressed_dictionary.bin                    # 压缩词典（单一字符串+锚点）
├── inverted_index_compressed.bin                # 压缩倒排表（VB+Gap）
├── inverted_index_with_skip.bin                 # 带跳表指针的未压缩倒排表（如配置生成）
└── inverted_index_compressed_with_skip.bin      # 带跳表指针的压缩倒排表（如配置生成）

output/
├── boolean_search_results.txt                   # 布尔检索实验结果

part_4/output/
├── phrase_search_results.txt                    # 短语/邻近检索实验结果
└── compression_comparison_results.txt           # 压缩对比实验报告
```

## 配置说明

### 检索配置文件 (`config/retrieval_config.json`)

```json
{
  "inverted_index_dir": "output_inverted_index",
  "documents_dir": "output/normalized_events",
  "output_dir": "output_retrieval",
  "tfidf_method": "standard",
  "top_k": 10,
  "min_similarity": 0.0,
  "test_queries": [
    "music concert performance",
    "food recipe cooking",
    "art exhibition gallery"
  ]
}
```

### 主要配置参数

- `tfidf_method`: TF-IDF 计算方法
- `top_k`: 返回的文档数量
- `min_similarity`: 最小相似度阈值
- `test_queries`: 测试查询列表

## 技术栈

- Python 3.x
- NumPy/SciPy (向量计算)
- NLTK (文本处理)
- JSON 配置文件
- 自定义文本处理算法
- 倒排索引数据结构
- 向量空间模型

## 开发环境

确保您的系统已安装 Python 3.x 环境和必要的依赖包：

```bash
# 使用 conda 环境
conda env create -f environment.yml
conda activate web_lab1_env

# 或使用 pip
pip install nltk numpy scipy
```

## 使用示例

### 基本检索流程

1. **准备数据**：
   ```bash
   python scripts/run_processor.py
   # 可二选一：基础版或带跳表指针版
   python scripts/run_inverted_index_builder.py
   # 或
   python scripts/run_inverted_index_builder_with_skip.py
   ```

2. **执行检索**：
   ```bash
   python scripts/run_retrieval.py --mode test
   ```

3. **查看结果**：
   - 控制台显示检索结果
   - `output_retrieval/` 目录包含所有输出文件

### 交互式查询示例

```bash
python scripts/run_retrieval.py --mode interactive
```

交互式命令：
- 输入查询文本进行检索
- `help` - 显示帮助信息
- `config` - 查看当前配置
- `methods` - 查看 TF-IDF 方法
- `method log` - 切换到对数频率方法
- `quit` - 退出程序

## 许可证

本项目仅供学习和研究使用。
