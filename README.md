# 数字神探 Digital Detective

面向研究人员的中文刑事案件研究工作台。输入或上传文字材料，Agent从原文提取线索、核查证据、提出关键问题，并形成可引用依据的暂定结论或“证据不足”。

**2026年10月21日起采用本地离线方案。外部模型API已停用。** 本轮目标为500+复杂刑案、五维标注、实际权重微调，首轮完整验证与独立测试的整案准确率门槛为70%。尚未达到的指标不会因代码完成或训练损失下降而宣称通过。

## Mac离线运行

当前实现优先支持Apple Silicon Mac，按24GB统一内存设计。Win11兼容尚未实现。MLX在普通Mac终端访问Metal GPU；容器内或受限沙箱可能不能访问GPU。

一次联网准备：

```bash
git clone https://github.com/Momingzzhh/digital-detective.git
cd digital-detective
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-mac.lock
cp .env.example .env
chmod 600 .env
.venv/bin/python scripts/download_local_model.py
```

此后断网也可启动，不自动安装或下载文件：

```bash
sh run_offline.sh
```

打开 http://127.0.0.1:8765 ，首次注册本机账户。无需任何模型API Key。准备阶段只下载公开文件；标注、训练、推理均使用本地权重，禁止云端回退。断网检查见[data/offline_smoke_report.json](data/offline_smoke_report.json)，它是软件机制测试，不计入真实案件数量或准确率。

基座采用Apache-2.0许可的[Qwen3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)及固定修订的[MLX四位转换](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)。模型文件逐一校验Git blob或LFS SHA-256，见[本地模型清单](data/local_model_manifest.json)。网页明确显示基座或已配置适配器，不能把基座推理冒充微调成果。

## 五维复杂案件与实际训练

完整要求、字段含义和复现顺序见[离线训练及验收规程](docs/local_training_protocol.md)，字段结构见[五维Schema](data/five_dimensions_schema.json)，实时阶段统计见[data/five_dimensions_report.json](data/five_dimensions_report.json)。

计划冻结600个不同案件组，训练420、验证90、测试90，严格70/15/15。每案至少6条有效线索、3类证据，具有真实矛盾或多个待排查对象，且至少需要3层证据关联。标注证据层级、逻辑链、矛盾点、推理深度和对话轮次。来源、语义充分性与人工验收状态分别记录。

训练集的多轮对话只能从原始材料释放证据，轮次标记为模拟；验证、测试和新案记录真实研究人员的交互，未发生时为null。训练变体包含证据顺序扰动、缺失证据时拒绝定论、合成提示注入及模拟证据追问；变体不增加真实案件数量，验证和测试案件禁止生成训练变体。

```bash
# 须先有来源材料和跨来源关联审计，案源获取见docs/data_sources.md
.venv/bin/python scripts/prepare_five_dimensions.py
.venv/bin/python scripts/annotate_five_dimensions_batch.py --limit 1980 --batch-size 4
.venv/bin/python scripts/assemble_five_dimensions.py
# 完整数据未准备好时，正式训练会被阻止
.venv/bin/python scripts/train_local.py
```

QLoRA会实际更新适配器权重，记录参数、数据及权重哈希，检查学习后的LoRA B矩阵。超长训练样例不会静默截断；所有训练组必须有可用样例。小样本pilot只验证训练工程，不能进入正式验收。token级验证损失不等于整案准确率。

2026-10-21已完成一次8个微批次、2次优化器更新的工程试训：56个LoRA B张量发生非零更新，记录峰值内存7.093GB。见[真实训练记录](data/latest_training_report.json)及[训练后断网检查](data/pilot_offline_smoke_report.json)。这是小样本pilot，尚无500+合格数据集、正式训练完成或70%整案准确率的证据。

长任务可脱离当前终端续跑；Mac运行期间暂时阻止自动休眠，任务退出后解除。默认4案批次，实测8案批次发生Metal内存不足，见[资源边界记录](data/local_resource_probe.json)。

```bash
.venv/bin/python scripts/run_local_pipeline.py --launch
.venv/bin/python scripts/run_local_pipeline.py --status
# 如需停止，等当前标注批次保存后退出
.venv/bin/python scripts/run_local_pipeline.py --stop
```

流水线先续标、导出并检查600案配额；不足时保留失败原因，不降低门槛补数。完整材料导出后，须先由研究人员核验全部验证/测试来源，再次续跑才会进行正式训练及断网检查。最终真人验证和独立测试仍由网页完成。日志保存在本机runtime/local_pipeline.log。

## 人工复核与独立验收

在.env设置RESEARCH_REVIEW_ENABLED=true，登录后打开 http://127.0.0.1:8765/research 。该页面向复核人员显示原文、隐藏答案及五维标注，至少验证与测试集合必须逐案人工核验；acting Agent只收到研究人员选择释放的原始证据。

正式训练后将.env中的LOCAL_ADAPTER_PATH设置为run.json记录的适配器目录。完成真人验证集交互、评分后运行：

```bash
.venv/bin/python scripts/summarize_acceptance.py
```

完整验证集达到70%后才开启测试集。首次测试冻结模型和数据版本；同一测试案不选择多次运行中的最好结果。测试集打开后禁止继续同周期训练调参。拒答、失败、超时及未完成案件保留在分母，报告Wilson 95%区间。只有数据人工核验、实际权重训练、完整验证与独立测试均通过，才能标记训练完成。

## 功能与使用边界

- TXT、Markdown、DOCX及有文字层的PDF上传；本机账户和案件隔离、持续对话、导出与删除。
- 原文连续引句及字符偏移检查；证据工具、矛盾核对、竞争解释、关系图及待补线索。
- 仅处理文字证据，不鉴定原始指纹、DNA或视频，不连接警务数据库。
- 模型自检可能产生相关错误，主观置信度不是已校准概率；70%是有限测试指标，不是对任意新案的保证。
- 公开旧案预训练污染和漏检的关联案件仍可能存在，需要在研究报告中披露。

原始文书、证据、答案、人工审核身份和权重保留本地忽略目录。公开仓库包含代码、配置、版本清单与汇总；代码MIT许可不授予第三方材料再分发权。案源、许可及历史审计见[数据来源](docs/data_sources.md)、[数据集卡](docs/dataset_card.md)和[质量审计](docs/quality_audit.md)。旧版云端1000案90%流程及404案包保留为[历史记录](docs/legacy_readme_20261016.md)，不并入本轮达标统计。旧账本保留待对账，不新增云端预占。

## 工程验证

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/smoke_offline.py
```

Linux CI验证与数据/账户隔离有关的机制，不验证Mac GPU性能或真实案件准确率。旧Docker配置不提供本轮Mac Metal推理环境。当前交付为本地系统与GitHub源码，未提供公网托管服务。
