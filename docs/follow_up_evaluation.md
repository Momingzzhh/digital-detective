# Historical aggregate — not a verified evaluation

Status revised on 2026-11-12 under T02. The old report contains claims of 150 training cases, 150 non-overlapping test cases and 112 correct (74.7%). Those claims cannot currently be verified. They must not be used as an independent result, a baseline, a milestone achievement, evidence of training effectiveness, or a paper conclusion.

The original [aggregate JSON](../data/follow_up_aggregate_report.json) is retained byte-for-byte for traceability. Its `target_met=true` and non-overlap flag are historical assertions, not current validation. No missing predictions, manifests or logs have been reconstructed. The old Wilson interval only transforms unverified counts and does not establish validity.

Reconsideration requires original per-case predictions and scoring decisions, train/test group manifests, model/adapter and input hashes, training logs, scoring protocol and exclusions. An independent audit must recompute counts and check group overlap before any new result is admitted.

The currently activated local adapter used five cases/86 examples. The latest completed seven-case/114-example run is saved but not activated. Their `valid.jsonl` files contain training replay probes, not held-out validation. Neither software checks nor replay loss establish independent case accuracy.

See [archive index](research_v2/historical_results.md), [baseline audit](research_v2/baseline_audit.md), and [development protocol](research_v2/protocol_draft.md).
