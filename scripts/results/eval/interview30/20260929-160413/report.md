# 30 题分层评测

- 运行：20260929-160413
- 评测集：`finance_rag/src/eval/data/research_report_interview30.jsonl`
- 语料：当前 18 篇研报。30 题 = 8 道显式单跳 + 10 道跨文档 + 2 道负样本 + 10 道模糊配对。
- 模糊题与原题共享证据，检索主指标用独立正例（explicit + cross_doc，18 题），不把 30 题加成一个总 Recall。

## 指标定义

- Recall@5：top-5 子块 id 里，命中的证据块占该题证据块的比例。负样本不进入这项。
- MRR@5：第一个证据块在 top-5 中的位次的倒数；没进 top-5 记 0。
- parent_restore_rate：子块已进 top-5 的证据里，父块长度允许替换、且引文出现在 PostgreSQL 父块全文中的比例。父块过长而未替换的次数单独计数。
- context_precision：Ragas 对生成回答所用上下文的精确率，只在「标准检索 vs 三开关全开」这组里计算。

## 检索四级

VectorStoreUnavailableError: 向量数据库客户端初始化失败（uri=http://localhost:19530）：<MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>

## Agentic 对照

VectorStoreUnavailableError: 向量数据库客户端初始化失败（uri=http://localhost:19530）：<MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>


## 三个开关的证据召回

VectorStoreUnavailableError: 向量数据库客户端初始化失败（uri=http://localhost:19530）：<MilvusException: (code=2, message=Fail connecting to server on localhost:19530, illegal connection params or server unavailable)>

## 复现

- `python scripts/build_interview30.py`
- `python scripts/evaluate.py dataset validate --dataset finance_rag/src/eval/data/research_report_interview30.jsonl --online`
- `python scripts/run_interview30.py`

## 说明

数字来自这次运行。简历里的 Recall@5=90%、MRR@5=1.0、上下文精确率 +14.1% 不是这套 18 篇研报、30 题协议的结果。
