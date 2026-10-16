# 数据来源与复现

截至2026-10-16，本机有47,558份来源候选：LeCaRD v1 5,200份、LeCaRDv2 triplet 衍生全文8,590份、MTEB LeCaRDv2新增2,311份、MNBVC新增31,457份。候选数量不能视作独立且线索充分的案件数量。机器处理、分组、费用与验收状态见[统计](../data/dataset_report.json)。

原文、证据、隐藏答案、模型检查点及研究者身份只保存本地。公开仓库发布代码、来源版本、校验和与统计，不重新分发案件内容。上游许可不能替代对原始文书使用、隐私和再分发条件的判断。

## 已取得的来源

| 来源 | 固定版本 | 本机结果 | 审计 |
|---|---|---|---|
| [LeCaRD v1](https://github.com/myx666/LeCaRD) | `b1a4b72b2050a4aa643e6360d5ff3a6b1fd221e3` | 两个ZIP共10,718条记录，9,088个不同writId；筛选5,200份候选 | [source_audit.json](../data/source_audit.json) |
| [sentence-transformers/lecard-v2](https://huggingface.co/datasets/sentence-transformers/lecard-v2) | `84ebcf30a342b044e4c99c3510e34bb8830356da` | 两个Parquet，重复三元组去重、分段后新增8,590份 | [lecardv2_source_audit.json](../data/lecardv2_source_audit.json) |
| [MTEB LeCaRDv2](https://huggingface.co/datasets/mteb/LeCaRDv2) | `566865fc50a566db1635307496e46c5a1a118cb2` | 3,795篇全文；与已有来源及自身去重后新增2,311份 | [mteb_source_audit.json](../data/mteb_source_audit.json) |
| [MNBVC-judgment](https://huggingface.co/datasets/wormtooth/MNBVC-judgment) | `dcbbb633b313eb9270e32cee763bb1dfc0702718` | 11个分片868,929篇各类文书，60,196篇含刑事判决标记；筛掉乱码标记、缺段及重复后新增31,457份 | [mnbvc_source_audit.json](../data/mnbvc_source_audit.json) |

LeCaRDv2官方55,192篇全文的Google Drive地址在本机命令行及浏览器均超时。上表使用可访问的固定衍生版本，没有将无法下载的全文计入本机数量。镜像仅用于传输，下载后必须匹配固定版本SHA-256。

MNBVC作者[披露](https://github.com/wormtooth/MNBVC-judgment)来源存在随机字符乱码，并通过重复文书清洗还原。当前拒绝作者标记待查/低质量以及可疑字符的材料，但正则不能证明每个汉字正确，也不能证明每案已对照法院原件。这一来源只能作为待核验候选。11个分片约1.4GB，没有下载完整约125GB语料。

## 英国历史案件独立分组

按用户确认，历史案件单独保存和报告，不用这一分组的成绩代表现代案件分析能力。[Old Bailey官方数据说明](https://www.dhi.ac.uk/blogs/old-bailey/data/)记载原始庭审报告覆盖1674—1913年；[原始XML数据存档](https://doi.org/10.15131/shef.data.4775434.v2)与可访问的[BigLAM衍生数据](https://huggingface.co/datasets/biglam/old_bailey_proceedings)分别记录。

本机已取得全部6个庭审Parquet，固定BigLAM版本`8caa27eb4445098598d38ef8df04a79dd9d88903`，共197,751份报告。下载大小和SHA-256见[下载清单](../data/oldbailey_download_manifest.json)。只从1834—1913年、单一被告、单一原始类别、无减等裁决及完整标签连接的报告开始筛查；必须至少两名标注证人，保留质询、交叉询问和辩解，起诉段、裁决、量刑和程序概述不进入线索。含指控罪名或结局的证言整案排除。当前9,411份结构候选、7,252个按规范化当事人姓名形成的保守分组，尚非线索充分认证；见[来源审计](../data/oldbailey_source_audit.json)。

Flash从英文原始证言选择经过句子ID，Max审计可见线索对原始历史类别与参与者范围的支持。历史v1.5用Max执行来源审核、翻译和双语质检，Flash负责选择句子与明确参与者。译者仅看到允许的证言和中性人物对应，不接收起诉、裁决或隐藏答案。中文每句保留英文ID，隐藏经过同时绑定中英文偏移。分句或来源变化会使旧标注、审计和翻译失效，格式重试保留历史检查点。导出重新读取并校验全部固定Parquet、逐案原始XML段、连接标签与双语句子。

```bash
.venv/bin/python scripts/fetch_oldbailey.py
.venv/bin/python scripts/prepare_oldbailey.py
# 有费用；按预先固定哈希顺序、每姓名组最多一例
.venv/bin/python scripts/build_oldbailey.py --limit 500 --selection admission-priority --concurrency 24 --budget-ceiling 400
.venv/bin/python scripts/export_oldbailey.py
.venv/bin/python scripts/refresh_dataset_status.py
```

私有文件为`data/processed/review_bundle/oldbailey_v1/visible/{train,dev,test}.jsonl`和`hidden/{train,dev,test}.jsonl`。**这些是机器整理的研究材料，尚未通过独立验收。** 历史类别使用来源的`category/subcategory`，不伪装成中国现行罪名。全是单一被告有罪裁决，因此固定猜被告人的作案者基线很高，不能据此证明破案能力。姓名分组可能合并同名者或漏掉不同姓名的关联案，仍需核验。原始印刷报告经过历史编辑，原始有罪记录也不是客观真实有罪的保证。

来源及衍生译文遵循CC-BY-NC-4.0，需要署名且限非商业使用；MIT仅适用于项目代码。公共仓库只发布方法、版本与统计，不发布这些庭审原文或中文译文。

## 不纳入的来源

- [CLEG](https://github.com/NowYouSeeMeNot/CLEG)：本机核查后发现证据计划有“包括但不限于”等生成内容，不是案件的完整原始证据，不计入合格数。
- [CAIL2019阅读理解](https://github.com/china-ai-law-challenge/CAIL2019)：官方大训练集已下载并匹配Git blob SHA1。8,000段中4,041段含“被告人”，均没有完整“判决如下”，且缺少来源案号及独立完整结局。不用罪名标题或问答替代完整案件。见[cail_source_audit.json](../data/cail_source_audit.json)。
- LeCaRD-Elem：候选仍为LeCaRD的9,088篇，不重复计作新增案源。
- 用户样例JSONL：没有逐案来源证明，不计入真实案例数量。

## 本机复现

先下载和导入，再检查跨来源关联，最后执行有费用的标注。新增分片时保留已有MNBVC候选的来源和划分。不要删除处理文件来绕过冻结记录。

```bash
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/prepare_candidates.py --limit 5200 --preserve-splits
.venv/bin/python scripts/fetch_lecardv2.py --include-mteb
.venv/bin/python scripts/prepare_lecardv2.py
.venv/bin/python scripts/prepare_mteb.py
.venv/bin/python scripts/fetch_mnbvc.py --shards 000,050,150,250,350,450,500,650,750,850,950 --max-download-gb 2
.venv/bin/python scripts/prepare_mnbvc.py
.venv/bin/python scripts/audit_case_groups.py
.venv/bin/python scripts/build_v3.py --preflight
# 以下调用付费模型，仍不生成正式合格基准
.venv/bin/python scripts/build_v3.py --limit 1000 --selection hash --concurrency 8
.venv/bin/python scripts/export_review_bundle.py --source-version v3_8
.venv/bin/python scripts/refresh_dataset_status.py
```

## v3.7与v3.8材料边界

结构筛查只接受含具体内容的编号证言、供述、鉴定、勘验等条目，保留连续原始引文与字符偏移。后续未重复写“证人”的证言可识别；长条目不在1500字处静默截断，超过处理上限则整条拒绝。法院查明、判决、裁判理由和罪名标题不能作为可见证据。唯一文字转换是将“被告人/犯罪嫌疑人”中性化为“当事人”，保留转换日志。

Flash标注员只选择可见证据的句子ID，隐藏事实段只供标注时核对冲突。Max审计员检查来源区段、泄漏、全部参与者、罪名要件、经过覆盖及一致性、替代解释与身份映射。隐藏事实不会进入被测agent。固定分组哈希决定审查顺序，没有使用答题对错来选案。同系列基础模型的错误可能相关，机器通过仍须独立复核。

`audit_case_groups.py`连接已有组、全文/事实精确哈希、现代法院编码案号和法院范围内的旧案号引用。跨冻结集合的已知冲突先隔离，其他同组最多保留一条，不偷偷重新分配划分。未引用的二审、再审及近似改写仍可能漏识别；“未报警”不等于独立案件已验收。

现有v3.7批次保留为历史版本。v3.8另加指控罪名、法院概括和证据句子ID检查，审计结果不从旧版本继承；运行新版本须显式使用`--source-version v3_8`导出。`export_review_bundle.py`再次校验引文及gold句子的偏移，输出本地 `data/processed/review_bundle/v3_7/visible/{train,dev,test}.jsonl` 与 `hidden/{train,dev,test}.jsonl`。可见文件只含案ID、分组、划分及证据ID/文本/类型，隐藏文件另存结局、来源和审计。文件哈希见[curation_report.json](../data/curation_report.json)。这些文件仍未验收，不作为正式benchmark。

正式验收与冻结见[审核规程](human_review_protocol.md)。逐案独立审核和完整隐藏测试完成前，`qualified=0`与90%未验证保持不变。公开旧案可能进入基础模型预训练，下游未参与训练不等于排除预训练污染。

完整[数据集卡](dataset_card.md)说明字段、开发样本隔离、选择偏差、用途与许可。

## 当前账户阻塞与断点恢复（2026-10-16）

历史v1.5预定3000组处理中，百炼返回HTTP 400 `Arrearage`。当前导出234组历史材料；3000组内234组机器暂通过、403组拒绝、31组格式/调用错误、2332组因账户拒绝待恢复。旧失败检查点已归档，并单独标记为`provider_deferred`，不计入通过或有效来源审查数。具体状态见[历史处理报告](../data/oldbailey_curation_report.json)和[导出报告](../data/oldbailey_export_report.json)。

本地账本按供应商用量与已核对单价记账约255.70元，另90.29元为未对账预占；这不是阿里云官方账单。旧失败未保留可逐笔对应的供应商Request ID，不能直接删除预占。账户恢复后须先核对云端账单，仍遵守500元总上限。新代码保留账户错误的请求/预占标识，并在欠费、授权或限流拒绝后停止同批次排队请求。真正超时仍保留预占。

恢复账户后，同一条构建命令会继续`provider_deferred`记录，跳过已有通过和拒绝项；`--retry-errors`仅用于一次有归档的格式错误重试。无需删除数据或账本，不应循环重复尝试欠费请求。现代评测和历史评测仍需独立验收。
