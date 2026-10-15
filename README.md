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

LeCaRD 的真实裁判文书为候选案源，**不是现成的破案测试集**。文书事实段、标题、判决和“经审理查明”会直接泄露答案，不能直接传入被测 agent。

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

## 当前验收结果（2026-10-15）

已取得 9,088 份去重后的真实来源文书，并筛出 3,000 份候选文书；这些数字不代表独立且线索充分的案件数量。329 份机器处理记录中有 98 份机器通过，但固定抽查的 10 份均存在至少一项不满足完整基准要求的问题，已全部隔离。**当前严格验收合格案件数为 0；1000+ 要求尚未完成，90% 准确率尚未验证。** 详见 [质量审计](docs/quality_audit.md) 与 [状态统计](data/dataset_report.json)。

机器处理脚本只输出 `benchmark.v2.unverified.jsonl`，不会生成可宣称合格的基准。正式评测要求逐例独立核验后的 `independently_verified` 状态；设置这个字段本身不能替代实际审查。

```bash
# 有费用：机器筛选，不代表验收
.venv/bin/python scripts/build_benchmark.py --limit 20
# 仅本地已有修复示范时使用；结果不属于独立盲测
.venv/bin/python scripts/evaluate.py --dataset data/processed/audited_pilot.jsonl --pilot --limit 1
```

当前没有训练或微调产物。完整研究验收仍需要完成逐案来源、结局泄漏、同案分组、人物范围与证据充分性复核，冻结数据划分，再分别开展完整证据与渐进交互盲测。现有软件可以用于这一研究流程，但不能替代尚未完成的数据与效果验收。

一次真实案件软件示范中，罪名和所列经营主体匹配参考，经过还原未完全匹配，整案判定未通过；这只有1例且经过开发期检查，不能推算准确率。见 [示范报告](reports/pilot-full.json)。
