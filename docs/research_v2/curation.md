# 训练材料整理与人工复核入口

当前授权范围为 T06—T13：先使用已有七案，仅训练集；允许本机 Qwen3-14B 生成与审查；开发集和测试集正文封存。不启动训练、正式评分或更换网页模型。新增候选须等待首批质量检查，不能按模型正确率挑选。

## 本机文件

私有材料均在仓库的 `data/processed/research_v2/`，不上传 GitHub：

- `group_registry.private.json`：现代案件关联组、冻结集合、历史训练暴露、隔离标志，以及单独保留的 Old Bailey 历史身份。
- `current.json`：当前材料版本指针；其 `directory` 对应下方版本目录。
- `<版本>/cases.private.json`：训练用交互案件候选，包括证据、查询映射与未核验的状态支持规则。
- `<版本>/sources.private.json`：七案原文、来源、旧五维标注；用于复核，不能交给行动 Agent。
- `<版本>/evaluation_private/state_support_rules.json`：条件支持候选，尚不是正式金标。
- `<版本>/solvability.private.json`：可达性检查，不能替代语义可解性判断。
- `<版本>/local_audits/`：逐次机器审查、错误、请求/模型/代码指纹与修订建议；失败保留，不覆盖。
- `data/review/research_v2_human/`：只有实际登录用户提交后才产生人工回执。本轮代理不代签。

公开的统计与限制见 `group_registry_report.json`、`train_curation_report.json`、`curation_validation.json`。原始案件正文、人物、案例标识和机器建议不放入公开汇总。

## 查看和核对

在本机仓库根目录启动原有服务，临时启用复核页面：

```sh
RESEARCH_REVIEW_ENABLED=true ./run_offline.sh
```

先在 `http://127.0.0.1:8765/` 登录，再打开 `http://127.0.0.1:8765/research/v2`。
页面只列当前七个训练案例，不含开发集/测试集选择器。逐项核对原文、证据偏移、主体、事件、来源限定、反证与状态规则，填写说明和实际人工分钟数。所有勾选默认关闭。记录绑定来源、标注、支持规则和当前机器建议；这些内容变动后，旧记录失效。

“复核通过”仅代表单人非专家整理意见，不会放开旧系统正式评测门槛，不代表法律专家审查或独立测试通过。人工耗时单独记录，不并入模型运行时间。

## 可复现命令

```sh
# 首次从冻结七案构建（已有修订时拒绝回退）
.venv/bin/python scripts/build_interactive_cases.py
# 当前首案修订版的重建命令
.venv/bin/python scripts/build_interactive_cases.py --revision data/processed/research_v2/first_case_revision.private.json
.venv/bin/python scripts/audit_interactive_cases.py --limit 1 --attempt 2
/private/tmp/dd-research-v2-cpu/bin/python scripts/test_cpu.py
```

构建器绑定现有七案、历史清单及分组审计的 SHA-256；任何更改会拒绝复用旧来源。读取混合集合时仅扫描 JSON 分隔结构，反序列化身份字段，不反序列化留出正文。分组审计复用此前的案号、全文与事实哈希结果，仍可能漏掉改写重复或过度合并，不证明各案独立。

机器审查仅本地 14B、无适配器、温度 0、上下文上限 16,384、默认输出上限 2,048，不截断超长请求。请求、schema、模型清单、关键代码与输出均保留。同一绑定的已结束尝试不会自动重跑。创建 `runtime/research_v2.stop` 可在下一案例前停止；运行中发送 SIGINT 会保存中断状态。配额由工作代理检查，脚本自身没有账户配额接口。

## 当前能力与尚缺工作

严格 schema 将私有案件、可见观察、动作、结果和状态支持分开，禁止额外字段。初始观察不释放证据 ID、答案或私有行动目录。通用动作类型对所有案件一致；未匹配、错人错时和缺少前置条件不会返回证据，失败查询同样扣减次数。合成机制例子在 `schemas/research_v2/synthetic_example.json`，不混入真实数据。

当前真实案件仅生成按材料类型检索的原型，人物/时间定向调查以及来源支持的初始报案摘要尚待逐案设计；不能宣称完整调查环境已完成。引用证据保留原始文本，仅沿用已有的“被告人/犯罪嫌疑人→当事人”转换，机械核对不等于无语义泄漏。

旧五维事件和角色作为待复核候选，原子事件建议单独保存，不自动采用。首版 14B 即使自报“未发现问题”，也曾遗漏来源限定和身份引用，因此不得直接据其结论扩充数据。支持规则可表达多个替代充分集、反证与未知；真实七案暂沿用单个保守合取支持集，替代路径尚未逐案审定。法律要件与法律结论保持未知，不把裁判标签用作部分观察金标。

T12 真实人工签署必须由用户产生。T13 独立开发集及测试候选文本本轮按授权封存；训练材料尚不具备正式研究验收资格。

本次首案助手修订已保存17项事件/行为候选，另有角色与法律未知项，版本见current.json。最终14B建议仍遗漏身份上下文，已在该版本的assistant_audit_disposition.private.json中标记不采纳；模型原始输出与助手处置同时出现在复核页面。全部7案均无真实人工签署，尚未扩充数据。
