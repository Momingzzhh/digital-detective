# T04 实验身份与缓存契约

`app/experiment_manifest.py` 用严格JSON规范化（键排序、UTF-8、有限数值、保留数组顺序）生成完整SHA-256 run ID。输入包括 data/environment/actions/model/adapter/prompts/decoding/scorer/code/seeds/runtime；缺项、空对象、非字符串键、NaN和非JSON类型拒绝。adapter无权重时也必须明确写mode。种子需显式非负整数。

`build_manifest(config, files)` 对调用者提供的每个文件读字节计算哈希，路径搬迁不改变身份，内容更改改变身份。调用者必须绑定所有实际依赖：来源/划分/环境清单及其引用文件、动作定义、模型分片和tokenizer、适配器、提示词、评分代码、运行代码及lock。只绑定文件名、Git SHA或清单而不绑定对应内容不充分。此模块不自动发现遗漏依赖，不宣称单独保证GPU逐位确定性。

`verify_resume` 重新计算当前文件和配置，拒绝与保存清单不一致的恢复。`cache_read/cache_write` 每次先核验清单与当前输入，按run ID与案件/episode ID隔离缓存，并校验缓存内容摘要；禁止覆盖已有结果。部分写入/损坏JSON拒绝复用，保留后显式恢复。缓存目录只应由可信本地流程管理，摘要不是数字签名。

CLI（仅制作清单，不运行实验）：

```sh
python scripts/prepare_experiment_manifest.py --config CONFIG.json --files INPUTS.json --output artifacts/research_v2/RUN/manifest.json
```

INPUTS.json是逻辑名称到文件路径的字典，路径相对该JSON文件。输出已存在时核验而不覆盖。配置和绑定都相同才能复用；将一个真实运行的所有文件放入私有输入映射后使用，不把开发草案当实测配置。

`configs/research_v2/development.json` 是冻结的开发设计起点，含待T06等任务实现的占位语义，不可据此声称实验已运行。修改开发设计需版本化及说明；最终测试冻结在T42。旧 `scripts/evaluate.py` 暂未接入新缓存协议，迁移属于T40；本轮不宣称旧评测缓存问题已全部修复。
