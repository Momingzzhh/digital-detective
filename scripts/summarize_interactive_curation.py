#!/usr/bin/env python3
"""Public counts only, preserving all prior local audit attempts privately."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.research_v2_review import bundle, material, current_receipt
from scripts.build_interactive_cases import save, OUT
from app.experiment_manifest import file_digest


def main():
    folder, cases = bundle()
    attempts = Counter(); current_attempts = Counter(); decisions = Counter()
    for path in OUT.glob('*/local_audits/*/*.json'):
        row = json.loads(path.read_text()); attempts[row['status']] += 1
        if folder in path.parents:
            current_attempts[row['status']] += 1
            if row.get('audit'): decisions[row['audit']['decision']] += 1
    receipt_states = Counter(current_receipt(c.case_id, material(c.case_id))['status'] for c in cases)
    holds = sum(bool(material(c.case_id)['assistant_audit_disposition']) for c in cases)
    report = {'generated_at': datetime.now(timezone.utc).isoformat(),
        'bundle_binding': json.loads((folder/'manifest.private.json').read_text())['binding'],
        'train_cases': len(cases), 'train_groups': len({c.group_id for c in cases}),
        'evidence_items': sum(len(c.evidence) for c in cases),
        'claim_candidates': sum(len(c.claims) for c in cases),
        'claims_with_alternative_support': sum(len(claim.support_sets)>1 for c in cases for claim in c.claims),
        'all_version_attempt_statuses': dict(attempts), 'current_version_attempt_statuses': dict(current_attempts),
        'current_machine_decisions': dict(decisions), 'human_receipt_statuses': dict(receipt_states),
        'cases_with_assistant_audit_holds': holds,
        'formal_benchmark_eligible': False, 'heldout_text_accessed': False, 'new_model_training': False,
        'new_candidates_added': 0, 'expansion': 'held_until_first_batch_semantic_and_human_review',
        'source_snapshot_sha256': file_digest(folder/'sources.private.json'),
        'limits': ['Machine no_issue_found is not acceptance; prior proposals failed attribution/atomicity review.',
                   'One case has assistant source-bound refinement; remaining cases retain legacy event granularity.',
                   'Legal elements unresolved; concrete person/time query design incomplete; dev/test deferred.']}
    save(ROOT/'docs/research_v2/curation_validation.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
