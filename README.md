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

~~~text
RoBERTa-TextCNN-WebAttackDetection/
│
├── README.md
├── PROJECT_NOTICE.md
├── requirements.txt
├── .gitignore
│
├── dapt/
│   ├── prepare_owasp_corpus.py
│   ├── prepare_honeypot_corpus.py
│   ├── merge_pretrain_corpus.py
│   └── train_mlm.py
│
├── classification/
│   ├── config.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   │
│   ├── data_process/
│   │   ├── extract_csic2012.py
│   │   ├── extract_pkdd2007.py
│   │   ├── merge_datasets.py
│   │   
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
├── utils/
│   ├── __init__.py
│   └── utils.py
│
├── data/
│   ├── README.md
│   ├── raw/
│   ├── processed/
│   └── samples/
│
└── outputs/
    ├── README.md
    ├── dapt_lr_search/
    ├── classification/
    └── baselines/
~~~

## 4. 模块说明

### 4.1 数据预处理模块

数据预处理代码主要位于：

~~~text
classification/data_process/
utils/decode_utils.py
~~~

其中：

- `extract_csic2012.py`：用于解析 CSIC 2012 数据集中的 HTTP 请求样本；
- `extract_pkdd2007.py`：用于解析 PKDD 2007 数据集中的 HTTP 请求样本；
- `merge_datasets.py`：用于合并不同来源的数据集，并统一标签名称；
- `stat_pkdd_labels.py`：用于统计 PKDD 2007 数据集中的标签分布；
- `decode_utils.py`：用于 URL 解码、HTML 实体解码、Unicode 解码、Base64 解码和文本归一化。

处理后的 HTTP 请求统一表示为：

~~~text
[METHOD] ... [URI] ... [COOKIE] ... [REFERER] ... [BODY] ...
~~~

### 4.2 领域继续预训练模块

领域继续预训练代码主要位于：

~~~text
dapt/
~~~

其中：

- `prepare_owasp_corpus.py`：用于解析 OWASP / ModSecurity WAF 日志，生成继续预训练语料；
- `prepare_honeypot_corpus.py`：用于解析蜜罐日志，生成继续预训练语料；
- `merge_pretrain_corpus.py`：用于合并多来源预训练语料；
- `train_mlm.py`：用于基于 MLM 任务对 RoBERTa 进行领域继续预训练。

### 4.3 主模型训练模块

主模型相关代码主要位于：

~~~text
classification/
~~~

其中：

- `model.py`：定义 RoBERTa-TextCNN 模型结构；
- `dataset.py`：定义 RoBERTa 输入数据集；
- `train.py`：训练 RoBERTa-TextCNN 主模型；
- `evaluate.py`：模型评估函数；
- `predict.py`：使用训练好的模型进行单条 HTTP 请求检测。

### 4.4 对比实验模块

对比实验代码位于：

~~~text
classification/baselines/
~~~

包括：

- `train_bert_only.py`
- `train_roberta_only.py`
- `train_roberta_dapt_only.py`
- `train_textcnn.py`
- `train_bilstm.py`

这些代码用于对比不同模型结构在 Web 攻击检测任务中的表现。

## 5. 环境依赖

建议使用 Python 3.8 及以上版本。

安装依赖：

~~~bash
pip install -r requirements.txt
~~~

主要依赖包括：

~~~text
torch
transformers
datasets
pandas
numpy
scikit-learn
tqdm
~~~

## 6. 数据准备

由于数据集版权、体积及安全原因，本仓库不直接提供完整原始数据集和模型权重文件。

请将原始数据集按以下结构放置：

~~~text
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
~~~

## 7. 数据处理流程

### 7.1 提取 CSIC 2012 数据

~~~bash
python classification/data_process/extract_csic2012.py
~~~

### 7.2 提取 PKDD 2007 数据

~~~bash
python classification/data_process/extract_pkdd2007.py
~~~

### 7.3 合并分类数据集

~~~bash
python classification/data_process/merge_datasets.py
~~~

生成文件：

~~~text
data/processed/merged_two.csv
~~~

## 8. 领域继续预训练

### 8.1 构造继续预训练语料

~~~bash
python dapt/prepare_owasp_corpus.py
python dapt/prepare_honeypot_corpus.py
python dapt/merge_pretrain_corpus.py
~~~

### 8.2 运行 MLM 继续预训练

~~~bash
python dapt/train_mlm.py
~~~

继续预训练后的模型默认保存到：

~~~text
outputs/dapt_lr_search/
~~~

## 9. 训练主模型

~~~bash
python classification/train.py
~~~

训练结果默认保存到：

~~~text
outputs/classification/roberta_textcnn/
~~~

主要输出包括：

~~~text
best_model.pt
label_mapping.json
best_metrics.json
test_metrics.json
classification_report.txt
classification_report.json
confusion_matrix.csv
train_history.csv
model_config.json
~~~

## 10. 运行对比实验

~~~bash
python classification/baselines/train_bert_only.py
python classification/baselines/train_roberta_only.py
python classification/baselines/train_roberta_dapt_only.py
python classification/baselines/train_textcnn.py
python classification/baselines/train_bilstm.py
~~~

## 11. 使用训练好的模型进行检测

示例：

~~~bash
python classification/predict.py \
  --method GET \
  --uri "/index.php?id=1' or '1'='1"
~~~

POST 请求示例：

~~~bash
python classification/predict.py \
  --method POST \
  --uri "/login.php" \
  --body "username=admin' or '1'='1&password=123456"
~~~

输出结果包括预测类别、置信度以及各类别概率。
