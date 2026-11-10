# 数字神探 Digital Detective

**11月10日最新续跑：**继续筛选与标注后，Codex 5小时额度剩余9%时已停止。保存2份新案草稿及失败／中断记录，修复一处角色误拦截，313项软件测试通过；新增合格案例及权重更新均为0，原五案权重继续可用。后续先修复审核引用输出超长问题，见[续跑汇总](data/local_training_continuation_report.json)与[进度](docs/progress.md)。

**后续独立评测汇总：**另一个数据集上完成150案训练；150个与训练案不重叠的测试案中112案按整案标准正确，准确率为**74.7%**，达到70%阶段门槛。该结果与下述本机五案小批量运行分开统计。当前仓库保存[汇总记录](data/follow_up_aggregate_report.json)和[结果说明](docs/follow_up_evaluation.md)，尚未包含逐案评分与对应训练日志。

面向研究人员的中文刑事案件研究工作台。输入或上传文字材料，Agent从原文提取线索、核查证据、提出关键问题，并形成可引用依据的暂定结论或“证据不足”。

**当前交付：本机可用的离线 Agent（2026年11月10日）。** 已核验并启用现有五案微调权重：86条训练示例、88个微迭代、22次参数更新。支持材料上传、证据提取、工具核查、第二轮复核、补充线索、图谱与记录导出。无需模型 API Key。

本轮沿用已完成的真实训练，没有重复训练相同五案，也没有新增合格案件数。修复了真实流程中结构化提取的解码约束膨胀问题，并补齐权重激活、校验、中断恢复和运行版本记录。运行和验证方式见[本机交付说明](docs/local_delivery.md)，当前状态见[进度](docs/progress.md)。

600案复杂基准仍是后续研究目标；本机五案小批量按“先完整跑通、后续提高准确率”交付，**单就该五案运行尚无独立准确率结论**。上方74.7%来自分开的150案训练与独立测试汇总。软件检查和合成流程测试均不等于研究验收通过。

这台已准备好的 Mac 可直接启动：

```bash
sh run_offline.sh
```

打开 http://127.0.0.1:8765 ，注册本机账户后可先点击“填入合成示例”。下方“一次联网准备”供新机器使用；私有案源和微调权重不在 GitHub 中。

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

打开 http://127.0.0.1:8765 ，首次注册本机账户。无需任何模型API Key。准备阶段只下载公开文件；标注、训练、推理均使用本地权重，禁止云端回退。断网检查见[data/five_dimensions_v2_offline_smoke.json](data/five_dimensions_v2_offline_smoke.json)，它是软件机制测试，不计入真实案件数量或准确率。

基座采用Apache-2.0许可的[Qwen3 4B Instruct 2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)及固定修订的[MLX四位转换](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit)。模型文件逐一校验Git blob或LFS SHA-256，见[本地模型清单](data/local_model_manifest.json)。网页显示基座、未核验自定义适配器或已核验的小批量微调版本；已激活权重及配置按 SHA-256 校验。

## 五维复杂案件与实际训练

完整要求、字段含义和复现顺序见[离线训练及验收规程](docs/local_training_protocol.md)，字段结构见[五维Schema](data/five_dimensions_schema.json)，实时阶段统计见[data/five_dimensions_report.json](data/five_dimensions_report.json)。

长期计划冻结600个不同案件组，训练420、验证90、测试90，严格70/15/15。当前新版结构候选池为1,999组；候选不等于合格案例。每案至少6条有效线索、3类证据；完整三层基准还要求真实矛盾或多个待排查对象。正式小批量可用实际两层，用户另允许单行为人且无矛盾，单列统计。标注证据层级、逻辑链、矛盾点、推理深度和对话轮次。来源、语义充分性与人工验收状态分别记录。新版五维v2覆盖全案有行为证据支持的参与人（含另案处理者）及其角色；当前判决定罪名单单独保留，不能作为全案人物全集。关键经过采用摘要与逐句原文支持映射。

### 正式小批量训练

11月7日的新确认取消“必须凑齐600案才启动”的限制，优先三层、必要时实际两层。具体已被用户接受的修订稿重新核验后可按新政策派生；新标注须完成来源绑定的语义质检，可标为自动审核通过、待人工复核。原文、答案依据、集合隔离和完整长度检查仍为硬条件，没有放宽到一层或创建真人签署。

```bash
# 冻结新的来源绑定小批量政策；实际授权文件保留本机
.venv/bin/python scripts/prepare_small_batch_training.py --policy /本机私有路径/authorization.private.json
.venv/bin/python scripts/train_local.py --small-batch-manifest /本机私有路径/manifest.private.json --prepare-only --epochs 2 --max-seq-length 12288
# 使用准备输出中的不可变目录执行真实权重训练
.venv/bin/python scripts/train_local.py --small-batch-run artifacts/small_batch_formal/实际运行编号
.venv/bin/python scripts/smoke_offline.py --adapter artifacts/small_batch_formal/实际运行编号/adapters --output data/small_batch_offline_smoke_report.json
```

当前五案均保持实际两层，包含45条单条证据训练变体，从固定4B基座新建适配器，峰值内存8.863GB；训练重放token损失2.221→1.669不代表独立准确率。见[小批量数据](data/small_batch_dataset_report.json)、[真实训练](data/latest_small_batch_training_report.json)和[同权重离线检查](data/small_batch_offline_smoke_report.json)。先前两案及三案的训练、引用修正和实际重训报告另存；5次运行只涉及5个不同训练组。旧辅助试训没有被改名，不同阶段复用的相同案件不重复累计。完整研究验收仍未完成；该训练运行当时未自动激活，11月10日已通过显式核验启用。

新增`scripts/annotate_small_batch.py --create-cohort 4`从未处理的训练来源按分组哈希冻结诊断组；随后用`--cohort`逐案生成标注和单独语义质检。材料、原始输出、失败和代码快照留本机，全部通过后才能加入新的训练政策版本。没有使用留出案语义或待测Agent成绩选案。完整规则见[训练规程](docs/local_training_protocol.md#2026年11月7日确认的正式小批量训练规则)。下列旧完整基准流程仍用于600案规模化准备。

纠正后两项固定局部证据检查为2项通过、0项未通过；[原失败及相同输入比较](data/small_batch_probe_comparison.json)完整保留，不能作为独立准确率。

新阶段另记录[历次运行及去重统计](data/small_batch_run_history.json)、[实际训练分词与库版本](data/small_batch_runtime_audit.json)及[待人工复核阅读包状态](data/small_batch_pending_review_report.json)。原始阅读包只留在本机，不产生签署。训练结论的引用包含所有必要前提，修改目标后须实际重训；旧运行保留并标明是否已被新目标替代。

4B初标抽查发现人物、角色和伪矛盾错误，四案均隔离。已下载并逐文件校验固定修订的官方[Qwen3 14B四位本地标注模型](https://huggingface.co/Qwen/Qwen3-14B-MLX-4bit)，约7.86GB。单案批次已实际运行，资源记录见[本机内存实测](data/local_resource_probe.json)；首20案仍全部被复杂度或语义检查拦截，尚不能证明标注质量改善。Agent训练基座仍为4B。

标注先做原文边界、偏移及结构校验，再由另一轮本地生成核对人物角色、事件覆盖、真实矛盾和依赖关系。语义失败只允许一次依据原文与复核意见的修订，修订后重新经过所有门槛；结构复杂度不足不会被强制改成通过。不通过或质检未完成均隔离。该机器质检与初标使用相同基座，可能共享错误，不能替代人工复核。

训练集的多轮对话只能从原始材料释放证据，轮次标记为模拟；验证、测试和新案记录真实研究人员的交互，未发生时为null。训练变体包含证据顺序扰动、缺失证据时拒绝定论、合成提示注入及模拟证据追问；变体不增加真实案件数量，验证和测试案件禁止生成训练变体。

```bash
# 须先有来源材料和跨来源关联审计，案源获取见docs/data_sources.md
.venv/bin/python scripts/prepare_five_dimensions.py
# 数据构建使用单独的本地14B模型，不更改4B Agent训练基座
.venv/bin/python scripts/download_local_model.py --profile curator
.venv/bin/python scripts/annotate_five_dimensions_batch.py --limit 2000 --batch-size 1 --model-profile curator
.venv/bin/python scripts/assemble_five_dimensions.py
# 完整数据未准备好时，原600案基准训练会被阻止；小批量使用上面的显式入口
.venv/bin/python scripts/train_local.py
```

QLoRA会实际更新适配器权重，记录参数、数据及权重哈希，检查学习后的LoRA B矩阵。超长训练样例不会静默截断；所有训练组必须有可用样例。小样本pilot只验证训练工程，不能进入正式验收。token级验证损失不等于整案准确率。

2026-10-21已完成一次8个微批次、2次优化器更新的工程试训：56个LoRA B张量发生非零更新，记录峰值内存7.093GB。见[真实训练记录](data/latest_training_report.json)及[训练后断网检查](data/pilot_offline_smoke_report.json)。这是旧版口径的小样本pilot，不能用于新版全案角色验收；尚无500+合格数据集、正式训练完成或70%整案准确率的证据。旧版55案机器初标已[归档](data/five_dimensions_v1_legacy_report.json)，不计入本轮数量。

长任务可脱离当前终端续跑；14B标注默认每批1案，每2案退出并重启工作进程，释放GPU与约束编译资源。8案批次曾发生Metal内存不足，见[资源边界记录](data/local_resource_probe.json)。任务暂时阻止空闲休眠，但合盖仍会暂停计算；持续运行请保持开盖并接通电源。进度文件记录约束编译、预填充、生成与保存阶段。

```bash
.venv/bin/python scripts/run_local_pipeline.py --launch
.venv/bin/python scripts/run_local_pipeline.py --status
# 如需停止，等当前标注批次保存后退出
.venv/bin/python scripts/run_local_pipeline.py --stop
```

原完整基准流水线先续标、导出并检查600案配额；不足时保留失败原因，不降低门槛补数。它不替代新的显式小批量入口。完整材料导出后，须先由研究人员核验全部验证/测试来源，再次续跑才会进行完整基准训练及断网检查。最终真人验证和独立测试仍由网页完成。日志保存在本机runtime/local_pipeline.log。

## 人工复核与独立验收

当前先进行训练集标注校准：`scripts/prepare_calibration_review.py --limit 10`生成原文与候选标注对照页。研究人员可先核对3案并导出反馈，操作见[校准与复核规程](docs/local_training_protocol.md#人工复核与真人验收)。校准反馈不自动批准正式案例，包内原文、身份与意见均不提交GitHub。长度检查已修复为显式统计token ID，[候选池长度审计](data/local_context_audit.json)仅判断可容纳性，不代表数据已合格。

截至2026-11-05，该批已确认训练集校准反馈为5份，涉及4个不同案例；该批最新意见为3案需修订、1案排除，见[回收统计](data/calibration_feedback_report.json)。11月7日处理的新选案反馈另见[独立统计](data/returned_selection_report.json)，不混入已签署校准计数。程序据此加强引用与来源类型检查，并另建3案的来源修订，原冻结池不覆盖。人工意见、Word和具体修订引文均保留本机，汇总与[结构规则审计](data/calibration_source_policy_audit.json)不计作合格数据或准确率证据。

用户已同意将修订正例用于辅助训练、保留实际层数，并对第二份新稿的具体三个核对点作出确认。当前绑定到完整修订稿的2案均保持实际两层，生成18条监督示例，最长4778 tokens，见[辅助数据统计](data/auxiliary_calibration_report.json)。辅助数据和适配器使用独立目录、入口和报告；不能计入600案正式复杂基准，也不自动激活为网页当前模型。准备和试训方法见[辅助训练规程](docs/local_training_protocol.md#用户接受修订稿的辅助训练)。

2026-11-05已完成两案联合辅助试训：20个微迭代、5次优化器更新，56个LoRA B张量非零，峰值内存7.158GB，见[独立辅助试训报告](data/latest_auxiliary_training_report.json)。同一权重哈希的新适配器通过[断网及空证据注入拒答检查](data/auxiliary_offline_smoke_report.json)，联网尝试0、云端账本未变。此前单案试训及其检查报告保留在原本机运行目录。训练重放损失和合成检查不能证明独立准确率提升；三层基准可导出数仍为0，11月7日的新正式小批量结果单独见上文，完整验收未完成。

后续来源规则v4.4补齐司法评判边界、鉴定与财务记录标题。新的2案来源诊断已完成，均因支持或依赖问题隔离；[确定性复查](data/training_source_screen_recheck_report.json)另拦截将原句直接转述算成依赖层级的情形，原模型输出保留。来源补全只追加连续原文、保留旧线索与集合，不绕过已确认的人工暂停或答案泄漏检查。正式复杂案例仍未增加。

继续修复一份被内部编号截断的报告，恢复287个连续原始字符并保留旧池，见[来源修复统计](data/training_source_extension_report.json)。另一份已有人工反馈的材料已整理成完整两层辅助修订稿；用户已确认具体版本的三个核对点，见[核对状态](data/auxiliary_correction_review_report.json)。新确认另存并绑定原稿，标注正文不改写、不继承旧签署。两案联合试训及离线机制检查已完成，适配器未激活；完整软件测试218项通过。

在.env设置RESEARCH_REVIEW_ENABLED=true，登录后打开 http://127.0.0.1:8765/research 。该页面向复核人员显示原文、隐藏答案及五维标注，至少验证与测试集合必须逐案人工核验；acting Agent只收到研究人员选择释放的原始证据。

小批量权重请按[交付说明](docs/local_delivery.md)使用激活脚本；它同时绑定权重及配置哈希。后续完整研究验收中，完成真人验证集交互、评分后运行：

```bash
.venv/bin/python scripts/summarize_acceptance.py
```

整案正确要求罪名、全案行为参与人、角色和关键经过同时正确；仅在场或仅被提及者不算作案者。完整验证集达到70%后才开启测试集。首次测试冻结模型和数据版本；同一测试案不选择多次运行中的最好结果。测试集打开后禁止继续同周期训练调参。拒答、失败、超时及未完成案件保留在分母，报告Wilson 95%区间。只有数据人工核验、实际权重训练、完整验证与独立测试均通过，才能标记训练完成。

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
