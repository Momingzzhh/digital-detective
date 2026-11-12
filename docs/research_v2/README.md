# Research v2：T01—T05 工作入口

本轮范围由用户确认：只完成T01—T05，直接同步main，五小时额度剩余≤15%时保存停止；不启动训练或正式研究测试。后续T06—T55需另行启动。

- [基线与私有数据入口](baseline_audit.md)
- [历史结果撤回与归档索引](historical_results.md)
- [CPU测试与Mac离线运行](setup.md)
- [实验身份和缓存契约](experiment_identity.md)
- [已核对原文的相关工作矩阵](related_work_matrix.md)
- [冻结的开发协议v1](protocol_draft.md)
- [任务状态账本](task_status.json)
- [验收记录](validation.json)

开发配置在 `configs/research_v2/development.json`。私有基线清单、训练组索引、测试原始日志仅在 `artifacts/research_v2/`，不发布案件内容。旧正式评测入口尚未接入新实验缓存模块，这项后续迁移属于T40，不因模块测试通过就视为旧评分系统已修复。
