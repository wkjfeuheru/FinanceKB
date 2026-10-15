# 研报 Payload 与检索联合改造验收记录

2026-10-02；代码、schema 与数据迁移完成。检索质量目标未通过，不能据此宣称召回改善。

- 18/18 PDF 入库，1504 块；全部 report_type 与类型必填字段合法。新 schema 无 security_code。
- BGE-M3 CPU ONNX INT8、CLS、1024 维；存储抽样范数接近1，两个离线独立进程复用同一缓存。
- 完整 pytest tests/ -v：595 passed，24 条既有警告。前端 PATCH 3 passed，build 成功。
- 原始失败4份均为标题字节超限，UTF-8安全截断修复后重试成功；原始PDF保留。

## 同一新知识库与核心 Gold 数据集

|配置|Recall@5|MRR|正确文档 any@5|粗排证据 any@50|重排输入证据 any@50|事实覆盖|平均耗时 ms|
|---|---:|---:|---:|---:|---:|---:|---:|
|dense-1.0-L3|0.1018|0.1667|17/18|13/18|8/18|0.3796|3446|
|weighted-0.7-L0|0.1389|0.1389|17/18|13/18|N/A|0.4074|61|
|weighted-0.7-L1|0.1389|0.1222|16/18|13/18|N/A|0.3796|73|
|weighted-0.7-L2|0.1389|0.1389|17/18|9/18|N/A|0.4074|64|
|weighted-0.7-L3|0.1574|0.1944|17/18|13/18|8/18|0.4352|2021|
|weighted-0.7-L5|0.1574|0.1944|17/18|13/18|8/18|0.4352|2140|
|weighted-0.8-L1|0.1389|0.1296|16/18|13/18|N/A|0.3796|65|
|weighted-0.8-L2|0.1389|0.1389|17/18|9/18|N/A|0.4074|53|
|weighted-0.8-L3|0.1574|0.1944|17/18|13/18|8/18|0.4352|2193|
|weighted-0.8-L5|0.1574|0.1944|17/18|13/18|8/18|0.4352|2443|
|weighted-0.9-L1|0.1389|0.1389|16/18|13/18|N/A|0.4352|355|
|weighted-0.9-L2|0.1389|0.1389|17/18|9/18|N/A|0.4074|359|
|weighted-0.9-L3|0.1574|0.1944|17/18|13/18|8/18|0.4352|2439|
|weighted-0.9-L5|0.1574|0.1944|17/18|13/18|8/18|0.4352|2464|

## 结论与未通过项

- 0.7/0.8/0.9 的 L3 Recall/MRR 相同；Weighted vs Dense 的10000次Bootstrap为 inconclusive。cliff没有指标收益，保持默认关闭。
- 候选构建阶段从全局13/18题证据变为实际重排输入8/18，涉及文档路由、局部检索与配额截断；尚不能将损失独立归因于某一项。减少路由source或增加k至100未同时改善Recall与MRR；见 candidate-diagnosis/。
- 正确文档any@5为17/18，未达到18/18；evidence-any@50≥15/18、L3 Recall/MRR>0.2407 尚未达到；cross_doc Recall为0.0833，未达0.1333，详见 metrics_overview.json。
- 权重保留现有实验配置0.8/0.2，未用18个正例宣称稳定优越性。至少100条人工核验仍需人工工作，未自动伪造批准标注。
- 历史KB指纹不同，比较器明确返回 no_decision；历史0.2407只作门槛参考，不能直接当作可比增益或退化。
- quote覆盖使用严格数值边界；旧引文有从2026内部截为26的片段，会保守漏计。需人工完善引文边界，required-fact覆盖同时报告。
- 核心gold20全部引文唯一命中，chunk ID无变化；额外interview30的3条跨块引文已按同一事实定位真实原文并重新验证。
- 首组dense含冷加载且烟测曾并行运行，延迟是描述性运行耗时；其余组在同进程暖缓存顺序执行，未宣称严格延迟A/B。

## 可复现材料

audit.json、embedding-smoke-1/2.json、pytest-final.log、ingest.log、ingest-retry.log。
sweep/* 下含 run_config.json、records.jsonl、per_query.csv、summary.json、report.md、dataset_snapshot.jsonl。
run_config保存dataset/KB指纹、embedding/pooling/维度、chunk参数、权重、HEAD与dirty状态。
weighted-vs-dense.json/、cliff-comparison.json/、historical-comparison/含Bootstrap区间与可比性判定。
