# T01 基线核验（2026-11-12）

- 工作区：`digital-detective`；分支 `main`。用户明确选择直接提交 main，因此不执行任务书默认的新建分支。
- 起始 HEAD：`76574d3419b07498179d08f7da8a40aa6727bc6b`，与任务书相同；开始前 `git status --short` 为空。远端同步另在交付时检查，不把 origin/main 缓存当网络证明。
- 未发现仓库或祖先目录的适用 AGENTS.md。
- 授权范围 T01—T05；不启动训练、标注推理或正式测试；不更换网页适配器。五小时窗口剩余 ≤15% 时停止新增工作并保存、同步。起始剩余85%。
- 私有清单：`artifacts/research_v2/baseline_manifest.json`；SHA-256 `e7015bd974ea04e0d00566554918f6f69daf00b60b8fc9cb13e58e815e8178f9`。
- 核验83个文件可读，缺失0，模型/适配器对照哈希不匹配0。包括4B与14B基座文件、已激活五案权重和未激活七案权重、来源和历史数据清单。实际 `.env` 由 `scripts/check_local_setup.py --verify` 只读检查确认五案运行启用；未打印配置秘密。
- 五案激活运行 `20261107T173318311149Z`，权重 SHA-256 `ee4e9344b957114cedcbe57c22fb2e3acc6e9656f5a8b95b7e3078de5d1b1282`。
- 七案最新完整运行 `20261111T113749687528Z`，权重 SHA-256 `3c2db2873018aace1b1ca2a0e7f9eaf6ac26254bd91d2cef49178cb021749e4c`，未激活。
- 4B清单 `data/local_model_manifest.json`，14B清单 `data/local_curator_manifest.json`；具体清单文件哈希和逐权重哈希见私有清单 `files`。
- 历史训练输入/候选身份保守索引38,198个group ID，每个绑定case ID及出现文件。包含冻结候选身份及跨版本别名，不是已训练案例数或独立案件数。来源与语义相关性完整审计属于T06；本轮没有重新划分或阅读测试案语义。

复核命令：`.venv/bin/python scripts/audit_research_baseline.py`。脚本仅读取模型、来源和训练身份；输出为本机私有文件，重复运行会更新核验时刻和工作区状态。原始首次快照另保存在 `baseline_manifest.initial.json`。
