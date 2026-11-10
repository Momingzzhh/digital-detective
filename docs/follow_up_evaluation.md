# Follow-up training and independent evaluation

The project owner reports a separate follow-up experiment using an additional dataset: **150 training cases**, **150 non-overlapping test cases**, and **112 whole-case correct**. The resulting independent case-level point estimate is **112 / 150 = 74.7%**, exceeding the 70% milestone. A case counts as correct only when the charge, all evidence-supported behavior participants and roles, and the key event reconstruction are correct together. A descriptive Wilson 95% interval from these counts is **67.2%–81.0%**; the point estimate reaches the milestone, while the interval does not establish performance above 70% at 95% confidence.

The machine-readable aggregate is in [follow_up_aggregate_report.json](../data/follow_up_aggregate_report.json). It is distinct from the five-case local adapter run and its synthetic smoke scenario. The counts are not added together, and the local five-case report remains an accurate record of that earlier run.

The individual predictions, scorer decisions, training log, model hash, and train/test manifests have not yet been added to this repository. The aggregate file is a record of the supplied result, not a reconstruction of those missing artifacts. When available, the research archive should bind the model and dataset revisions, publish the scoring rule and exclusions, and include one result per test case so the aggregate and error categories can be independently recomputed.

The existing local pool of 1,999 structurally screened case groups and pending source revisions remains future training material, not additional qualified cases. Expansion should retain source verification, human review, and case-group separation from the independent test set.
