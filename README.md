# Web-Attack-Detection-Based-on-RoBERTa-TextCNN
本项目实现了一个基于RoBERTa-TextCNN的Web攻击检测模型。项目以HTTP请求文本为输入，完成请求字段提取、编码解码、统一文本表示、RoBERTa领域继续预训练以及多类别攻击检测，支持Normal、SQLi、XSS、SSI、XPath、LDAPi、PathTraversal和OSCommandInjection等类别的识别。
## 项目功能

本项目主要包含以下功能：

1. HTTP请求字段提取与统一文本表示；
2. URL编码、HTML实体编码、Unicode编码、Base64编码等内容的解码与归一化；
3. 基于Web安全语料的RoBERTa领域继续预训练；
4. 基于RoBERTa-TextCNN的多类别Web攻击检测模型训练；
5. BERT、RoBERTa、RoBERTa-DAPT、TextCNN、BiLSTM等对比实验；
6. 使用训练好的模型对单条HTTP请求进行检测。

## 项目目录

```text
RoBERTa-TextCNN-WebAttackDetection/
│
├── README.md
│── requirements.txt
├── classification/
│   ├── config.py
│   ├── dataset.py
│   ├── evaluate.py
│   ├── model.py
│   ├── predict.py
│   ├── train.py
│   │
│   ├── data_process/
│   │   ├── extract_csic2012.py
│   │   ├── extract_pkdd2007.py
│   │   └── merge_datasets.py
│   │
│   ├── common/
│   │   ├── labels.py
│   │   ├── data_utils.py
│   │   ├── char_dataset.py
│   │   ├── transformer_dataset.py
│   │   └── train_utils.py
│   │
│   └── baselines/
│       ├── models.py
│       ├── train_bert_only.py
│       ├── train_roberta_only.py
│       ├── train_roberta_dapt_only.py
│       ├── train_textcnn.py
│       └── train_bilstm.py
│
├── dapt/
│   ├── prepare_owasp_corpus.py
│   ├── prepare_honeypot_corpus.py
│   ├── merge_pretrain_corpus.py
│   └── train_mlm.py
│
└── utils/
    ├── metrics.py
    ├── seed.py
    └── utils.py
```

## 模块说明

### 数据预处理模块

数据预处理代码主要位于：

```text
classification/data_process/
```

其中：

- `extract_csic2012.py`：用于解析 CSIC 2012 数据集中的 HTTP 请求样本；
- `extract_pkdd2007.py`：用于解析 PKDD 2007 数据集中的 HTTP 请求样本；
- `merge_datasets.py`：用于合并不同来源的数据集，并统一标签名称。

处理后的 HTTP 请求统一表示为：

```text
[METHOD] ... [URI] ... [COOKIE] ... [REFERER] ... [BODY] ...
```

该统一表示方式能够将不同来源、不同格式的数据转化为相同的文本输入形式，便于后续模型训练和对比实验。

### 领域继续预训练模块

领域继续预训练代码主要位于：

```text
dapt/
```

其中：

- `prepare_owasp_corpus.py`：用于解析 OWASP / ModSecurity WAF 日志，生成继续预训练语料；
- `prepare_honeypot_corpus.py`：用于解析蜜罐日志，生成继续预训练语料；
- `merge_pretrain_corpus.py`：用于合并多来源预训练语料；
- `train_mlm.py`：用于基于 MLM 任务对 RoBERTa 进行领域继续预训练。

领域继续预训练的目标是使 RoBERTa 更好地适应 HTTP 请求、攻击载荷、特殊符号和 Web 安全语义特征。

### 主模型模块

主模型相关代码主要位于：

```text
classification/
```

其中：

- `config.py`：保存主模型训练相关配置；
- `dataset.py`：定义 RoBERTa 输入数据集；
- `model.py`：定义 RoBERTa-TextCNN 模型结构；
- `train.py`：训练 RoBERTa-TextCNN 主模型；
- `evaluate.py`：模型评估函数；
- `predict.py`：使用训练好的模型进行单条 HTTP 请求检测。

主模型流程如下：

```text
HTTP 请求文本
    ↓
RoBERTa 语义特征提取
    ↓
TextCNN 局部特征提取
    ↓
Dropout + Linear 分类
    ↓
输出检测类别
```

### 对比实验模块

对比实验代码位于：

```text
classification/baselines/
```
这些对比实验用于验证不同模型结构在 Web 攻击检测任务中的表现，并分析领域继续预训练和 TextCNN 局部特征提取模块的作用。

### 公共工具模块

公共工具代码主要位于：

```text
classification/common/
utils/
```

## 环境依赖

建议使用 Python 3.10版本。

主要依赖包括：

```text
torch
transformers
datasets
pandas
numpy
scikit-learn
tqdm
```

安装依赖：

```bash
pip install -r requirements.txt
```

## 数据准备

请将原始数据集按以下结构放置：

```text
data/
├── raw/
│   ├── csic2012/
│   │   ├── attacks/
│   │   └── normals/
│   │
│   ├── pkdd2007/
│   │   ├── xml_train.txt
│   │   └── xml_test.txt
│   │
│   └── dapt/
│       ├── owasp/
│       └── honeypot/
│
└── processed/
```

其中：

- `data/raw/csic2012/`：存放 CSIC 2012 原始数据；
- `data/raw/pkdd2007/`：存放 PKDD 2007 原始数据；
- `data/raw/dapt/`：存放领域继续预训练相关语料；
- `data/processed/`：存放预处理后的训练数据。

## 使用训练好的模型进行检测

使用 `classification/predict.py` 可以对单条 HTTP 请求进行检测。

GET 请求示例：

```bash
python classification/predict.py \
  --method GET \
  --uri "/index.php?id=1' or '1'='1"
```

POST 请求示例：

```bash
python classification/predict.py \
  --method POST \
  --uri "/login.php" \
  --body "username=admin' or '1'='1&password=123456"
```

输出结果包括预测类别、置信度以及各类别概率。

## 项目声明

- 项目名称：基于RoBERTa-TextCNN的Web攻击检测方法
- 项目作者：Yang Xin
- 作者单位：暨南大学网络空间安全学院
- 开发语言：Python
- 核心技术：Web攻击检测、深度学习、RoBERTa模型、卷积神经网络
