# 本机可用版本 · 2026年11月10日

本轮交付目标是：使用现有训练成果，在这台 Apple Silicon Mac 上完成可操作的案件分析流程。训练准确率留待后续完善。代码、运行说明和脱敏汇总同步 GitHub；原始案源、账户、案件、配置备份和权重仍保留本机。

## 立即使用

在项目目录运行：

```bash
sh run_offline.sh
```

打开 <http://127.0.0.1:8765>。首次使用注册本机账户；已有账户直接登录。无需模型 API Key。

1. 点击“填入合成示例”，或输入／导入自己的文字材料。
2. 创建案件，点击“开始分析现有材料”。
3. 等待证据提取、工具核查、分析和第二轮复核完成；最终代码的短合成材料实测每轮约64秒，长材料会更慢。
4. 在“分析结论”查看依据，点击引用转到原文证据；“证据图谱”可点击节点。
5. 在右侧补充线索并发送，自动开始下一轮；“导出记录”保存完整 JSON。

终端按 Ctrl+C 停止服务。下次重新运行同一命令即可；账户和案件保存在 `runtime/detective.sqlite`。不要删除该文件来重启服务。

## 训练完成到哪一步

当前启用运行 `20261107T173318311149Z`：5个训练案件、86条示例、88个微迭代、22次优化器更新；56个 LoRA B 张量存在非零学习结果。11月10日重新检查全部86条训练输入，最长6978 tokens，没有截断，训练输入与已安装训练库的分词结果一致。

本轮**复用已完成的小批量真实训练**，没有为增加运行次数而重复训练同样五案，也没有把合成测试或重复运行计入案件数。当前权重 SHA-256：

```text
ee4e9344b957114cedcbe57c22fb2e3acc6e9656f5a8b95b7e3078de5d1b1282
```

本轮启用记录：[local_release_report.json](../data/local_release_report.json)。历史训练报告中的 `adapter_activated:false` 表示训练结束当时的状态；激活是11月10日完成的后续操作。历史 `training_complete:false` 指完整研究验收未通过，不否认该小批量权重训练已经完成。

600案、完整真人验证／测试及70%整案准确率继续作为研究目标。当前没有独立准确率结论；五案中3案仍是自动审核后待人工复核。第二轮复核使用同一本地模型，不是独立专家，也可能重复第一轮的错误。

## 本轮修复和补齐

- 首次真实完整流程在提取阶段被系统终止（exit 137）；代码检查发现证据列表上限80与引文长度上限5000会进入解码器的有限重复状态构建。现在解码器约束 JSON 结构，原 Pydantic Schema 继续校验所有长度限制，token预算仍限制生成；修复后完整流程通过。没有截断原始材料或放松输出验收。
- 显式激活脚本检查训练完成状态、基座清单、实际学习张量、同权重离线检查；更新本机配置并备份旧配置。权重或适配器配置被替换后拒绝推理，避免静默使用错误版本。
- 状态页面显示当前微调运行、训练数量和权重哈希。每轮分析把模型版本写入导出记录。
- 已提取证据随分析阶段保存；失败或重启后保留材料和已完成提取的证据，可重新分析。相同工具动作不反复执行。
- 前端使用串行轮询并核对当前案件；切换页面后旧请求不会把页面拉回旧案。补齐合成示例、启动检查和离线帮助，移除过时云端密钥提示。

## 已验证内容与复现

软件测试包含账户隔离、CSRF、上传、引用、研究集合隔离、激活校验和失败恢复。本轮共312项通过；另外使用真实MLX模型跑完整流程，不用模拟模型替代。

```bash
.venv/bin/python -m pytest -q
.venv/bin/python scripts/check_local_setup.py --verify
.venv/bin/python scripts/smoke_offline.py \
  --adapter artifacts/small_batch_formal/20261107T173318311149Z/adapters \
  --output data/local_delivery_offline_report.json
.venv/bin/python scripts/smoke_agent.py
```

`smoke_agent.py` 在独立本机数据库中使用合成材料和临时测试账户，阻止联网，执行两轮完整HTTP接口流程，并把原始结果存入忽略目录 `artifacts/agent_smoke/`。公开报告只包含检查结果、耗时和代码／权重哈希。测试覆盖上传、创建、提取、工具核查、复核、追加证据、引用定位、图谱、导出、模型版本、删除与退出登录；没有独立案件准确率含义。

- [完整 Agent 检查](../data/local_agent_smoke_report.json)：两轮通过，提取5条／7条证据，联网尝试0。
- [离线与空证据注入检查](../data/local_delivery_offline_report.json)：通过，云端账本未变。
- [网页界面检查](../data/local_ui_smoke_report.json)：独立测试账户登录、真实分析、切页、原文引用和图谱通过；测试服务已关闭。
- [实际训练输入复核](../data/small_batch_runtime_audit.json)：86条输入未截断。

初次失败运行保留在 `artifacts/agent_smoke/20261110T042632374263Z`；修复前后的运行没有当成新增训练或验证案例。

## 后续扩充和替换模型

继续沿用既有来源绑定的小批量入口，先添加经来源核验的新训练案例，保留验证／测试集合隔离，再训练新运行。完整命令和准入规则见[训练规程](local_training_protocol.md)。

```bash
.venv/bin/python scripts/train_local.py \
  --small-batch-manifest data/processed/你的新批次/manifest.private.json \
  --prepare-only --epochs 2 --max-seq-length 12288
.venv/bin/python scripts/train_local.py \
  --small-batch-run artifacts/small_batch_formal/新运行编号
.venv/bin/python scripts/smoke_offline.py \
  --adapter artifacts/small_batch_formal/新运行编号/adapters \
  --output data/new_adapter_offline_report.json
.venv/bin/python scripts/activate_local_adapter.py \
  --run artifacts/small_batch_formal/新运行编号 \
  --smoke-report data/new_adapter_offline_report.json
```

激活前先停止网页服务和其他模型任务；激活后重启网页，再运行 `scripts/smoke_agent.py`。每个训练运行保留原始记录，不覆盖旧权重。旧 `.env` 备份保存在 `runtime/activation_backups/`，如需回退可重新核验并激活旧运行。

新机器从 GitHub 克隆后，需要按 README 安装依赖并下载基座；私有五案训练材料和适配器不会随仓库提供。在未迁移合法持有的本地权重前，新机器运行的是基座模型，页面会明确显示。普通案件功能可使用基座，不能据此宣称已完成微调。

## 当前局限

当前训练规模很小，细节引用、假设和角色判断仍可能有语义错误；逐字引用及编号有效不等于推断正确。过长材料会明确报上下文超限，不静默删减证据。扫描PDF需先转成文字；当前只支持Apple Silicon Mac的本地推理。研究复核页面另由 `RESEARCH_REVIEW_ENABLED` 开关控制，不影响普通案件使用。
