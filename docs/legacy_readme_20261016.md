# 数字神探 · Digital Detective

面向研究人员的中文案件分析工作台。输入或上传案件文字，agent 提取原文证据、选择核查工具、提出竞争假设、询问关键缺口，并在独立复核后给出暂定分析或“证据不足”。

**研究状态：系统实现不等于准确率达标。** 90% 是独立测试目标；没有完成且通过盲测时，不宣称达成。数据的下载数、候选数、机器审核数与人工审核数分别记录。

## 本地运行

需要 Python 3.12+。Mac Apple Silicon 可直接运行；推理由阿里云提供，不需要本地 GPU。

```bash
git clone https://github.com/Momingzzhh/digital-detective.git
cd digital-detective
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.lock
cp .env.example .env
chmod 600 .env
# 使用文本编辑器在 .env 填写 DASHSCOPE_API_KEY
sh run.sh
```

打开 http://127.0.0.1:8765 ，首次注册一个本机账户。密钥只在服务器读取，不发给浏览器，不写入 Git。保存 `.env` 后下一次模型请求自动生效。

功能：

- 支持 TXT、Markdown、DOCX、有文字层的 PDF；单文件 10MB、提取 4 万字上限，单案累计 10 万字。
- 账户登录、独立案件列表、持久化对话与补充材料、JSON 导出和删除。
- 每条证据必须是来源材料的连续引文，记录来源与字符偏移；对不上原文的提取会被丢弃。
- 有界工具循环：搜索证据、查看证据、选择冲突核对材料，然后形成分析并独立复核。
- 展示证据、人物实体、结论之间的可点击关系图；展示竞争假设、反证和缺失信息。
- 置信度仅为未校准的模型估计，不能代替实测准确率。复核仍可能犯错。

## 模型与费用

默认 `qwen3.8-flash`，北京百炼 OpenAI 兼容接口。2026-10-15 核对官方价格为输入 ¥0.8 / 百万 tokens，输出 ¥2.7 / 百万 tokens；见[官方模型说明](https://help.aliyun.com/zh/model-studio/qwen3-8-flash)。接口与价格可随供应商变动，变更前更新配置和核验。

本项目没有训练基础模型。agent 的能力来自外部模型、工具和验证流程；不会把提示词编排称为“自主训练”。模型官方当前不支持该型号微调。

SQLite 账本在请求前按 UTF-8 字节上界和输出上限预占预算，返回后以用量结算。异常请求保留预占，避免超时重复收费后漏记。默认和硬上限为 500 元。该控制只覆盖本项目账本，不覆盖同一云账号的其他程序；还需在百炼控制台设置账户预算。不要删除账本来重置额度。

## 数据

数据源与核验规则见 [docs/data_sources.md](docs/data_sources.md)，下载审计见 [data/source_audit.json](data/source_audit.json)。

LeCaRD、LeCaRDv2衍生全文与MNBVC的裁判文书为候选案源，**不是现成的破案测试集**。文书事实段、标题、判决和“经审理查明”会直接泄露答案，不能直接传入被测 agent。

```bash
.venv/bin/python scripts/fetch_data.py
.venv/bin/python scripts/prepare_candidates.py
```

原始文书、加工后的证据、隐藏标签只在本地 `data/raw/`、`data/processed/`、`data/review/` 保存。公开仓库只包含下载和处理工具、审计及统计。用户提供的样例 JSONL 没有逐案来源证明，未计入真实案例数量。

## 测试与部署

```bash
.venv/bin/python -m pytest -q
# 真实 API 小额联通测试，虚构场景不计入 benchmark：
.venv/bin/python scripts/smoke_model.py
# 可选 Docker 本机部署：
docker compose up --build -d
```

Docker 默认只映射本机端口。当前交付为本地运行及部署配置，不包含公网托管。其他研究人员可以 clone 后用自己的密钥与本机账户运行。若自行部署公网，应使用 HTTPS 反向代理，将 `COOKIE_SECURE=true`，完成账户创建后关闭 `ALLOW_REGISTRATION`，并配置网关访问控制和速率限制。请保持单个应用进程，后台任务和限流状态保存在内存中；SQLite 数据落盘，重启中断的任务可由用户重试。

## 研究边界

第一版处理文字证据，不鉴定原始指纹、DNA 或监控视频，不连接警务数据库。用户输入和模型输出均需复核，系统允许拒答。展示的是可核验的证据摘要与工具记录，不是模型私密思维链。

独立测试须按同案分组、去重并隐藏罪名/作案者/真实结局。罪名与作案者采用集合完全匹配；经过还原报告关键事实覆盖以及整案全部正确率。拒答与失败留在分母中。公开旧案可能已存在于基础模型训练数据，不能宣称已经排除预训练污染。

## 代码结构

`app/` 后端、模型适配与 agent；`static/` 中文网页；`scripts/` 数据获取与实验；`tests/` 机制和隔离测试；`data/` 来源审计；`runtime/` 私有账户、案件、调用账本。

代码采用 MIT 许可。第三方数据遵循各自来源条款，代码许可不授予案件材料的重新分发权。

## 数据工作状态（2026-10-16）

本机来源候选已扩展到 **47,558份**，v3.7结构筛查通过5,056个初步分组。新增可访问的MTEB LeCaRDv2全文及MNBVC裁判文书分片，下载均固定版本并校验SHA-256。候选、结构筛查、机器审查与正式验收分别统计；最新数量和费用见[dataset_report.json](data/dataset_report.json)。

**1000个线索充分的独立案件尚未完成正式验收，90%准确率尚未验证。** MNBVC作者披露过乱码清洗还原，原文一致性仍须核查。跨来源关联检查会先隔离已知跨集合冲突，其他同组最多保留一例。未检出的关联和预训练污染仍可能存在。

v3.7从带具体内容的证言、供述、勘验、鉴定等原始证据条目提取连续引文；不静默截断长供述。Flash只选择原文句子ID作经过标签，Max独立检查证据、结局泄漏、作案者范围、罪名、经过与替代解释。同系列模型审查可能有相关错误，机器通过不等于已验收。

```bash
# 下载、导入、跨来源检查的完整顺序见 docs/data_sources.md
.venv/bin/python scripts/audit_case_groups.py
.venv/bin/python scripts/build_v3.py --preflight
# 有费用；按固定分组哈希选1000个待审样本，不按agent答题选案
.venv/bin/python scripts/build_v3.py --limit 1000 --selection hash --concurrency 8
.venv/bin/python scripts/export_review_bundle.py --source-version v3_8
.venv/bin/python scripts/refresh_dataset_status.py
```

可见证据和隐藏结局分别导出到本地 `data/processed/review_bundle/v3_7/visible/` 与 `hidden/`，并按train/dev/test分文件。再次校验来源与句子偏移；公开仅发布[导出统计和哈希](data/curation_report.json)。文件明确未验收，不自动生成正式benchmark。

完整[案源与复现说明](docs/data_sources.md)记录了使用与拒绝的来源。旧v2机器通过98例，但固定抽查10例原样通过0例，已隔离；详见[质量审计](docs/quality_audit.md)。一次开发期真实案件软件示范的整案判定未通过，只有1例，不能推算准确率；见[示范报告](reports/pilot-full.json)。

正式冻结采用[独立审核规程](docs/human_review_protocol.md)：双人逐项核验，审核单绑定来源、证据、隐藏标签及分组划分；程序再次验证偏移和关联冲突，满1000个独立组才生成私有正式基准。双人机制是项目质量控制要求，不代表已有人员完成审核。当前人工审核为0，不以模型审查冒充人工。

正式评测要求至少1000个已核验独立组、完整运行不少于150个隐藏测试案件，分别报告完整证据与渐进交互结果；错误、超时和拒答计入分母。只有整案全部正确率实际达到90%才触发达标声明。当前没有训练或微调产物，仍需完成数据核验和独立效果验收。

英国历史庭审已作为独立分组加入：全部197,751份记录已下载，1834—1913年初筛留下9,411份候选、7,252个姓名组。中英证言、起诉裁决及翻译审计分开保存，详见[历史案源说明](docs/data_sources.md#英国历史案件独立分组)。单一被告有罪案件存在明显基线偏差，不以历史结果代替现代案件验证。

[数据集卡](docs/dataset_card.md)说明两类研究材料的字段、划分、许可和局限。历史构建v1.5由Max进行翻译及质检，Flash选择句子；含承认表述的候选优先级不代表已通过审核。

当前已导出170组现代中文、234组英国历史中文机器整理材料，分别保存；历史分组另提供可直接上传网页的TXT。百炼账户出现`Arrearage`，扩充任务已保存断点，等待账户恢复及预占对账。详见[当前处理说明](docs/data_sources.md#当前账户阻塞与断点恢复2026-10-16)。这些数量不等于正式验收数量，1000案及90%目标仍未达成。

按用户要求暂停新增付费调用后，现有404案已整理为本地研究包：两组均提供逐案中文TXT、按原划分保存的可见JSONL，以及隔离的原始记录、隐藏答案和机器审核记录。公开[研究包汇总](data/research_bundle_report.json)仅含数量及哈希；本地包和案件正文不推送GitHub。复现命令及使用边界见[现有材料整理说明](docs/existing_materials.md)。
