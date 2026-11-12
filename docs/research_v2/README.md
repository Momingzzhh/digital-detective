# Research v2 工作入口

T01—T05 已完成。后续用户确认推进 T06—T13，仅使用训练案例并允许本机14B整理/审查；开发集和测试集正文继续封存。直接同步main，五小时额度剩余≤15%时保存停止；不启动模型训练或正式研究测试。

- [训练材料与逐项人工复核入口](curation.md)

- [基线与私有数据入口](baseline_audit.md)
- [历史结果撤回与归档索引](historical_results.md)
- [CPU测试与Mac离线运行](setup.md)
- [实验身份和缓存契约](experiment_identity.md)
- [已核对原文的相关工作矩阵](related_work_matrix.md)
- [冻结的开发协议v1](protocol_draft.md)
- [任务状态账本](task_status.json)
- [验收记录](validation.json)

开发配置在 `configs/research_v2/development.json`。私有基线清单、训练组索引、测试原始日志仅在 `artifacts/research_v2/`，不发布案件内容。旧正式评测入口尚未接入新实验缓存模块，这项后续迁移属于T40，不因模块测试通过就视为旧评分系统已修复。
