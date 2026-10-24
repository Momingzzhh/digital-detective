#!/usr/bin/env python3
"""Read-only structural audit from training calibration; never edits the frozen pool."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,sys,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from app.source_policy import RULES,CURRENT_CASE_FILE_LABEL
from scripts.build_v3 import evidence_kind,VERSION
rows=list(map(json.loads,(ROOT/'data/processed/five_dimensions_v2/candidates.private.jsonl').open()))
reasons=Counter();changes=Counter();private=[];clean=0;structural=0
for row in rows:
    problems=[];new_kinds=[]
    for e in row['evidence']:
        for name,pattern in RULES.items():
            if pattern.search(e['text']):problems.append(name)
        if CURRENT_CASE_FILE_LABEL.search(e['text']):problems.append('current_offence_file_label')
        kind=evidence_kind(e['quote'])
        if kind and kind!=e['kind']:
            changes[e['kind']+' -> '+kind]+=1
        new_kinds.append(kind or e['kind'])
    for name in set(problems):reasons[name]+=1
    if not problems:clean+=1
    if not problems and len(row['evidence'])>=6 and len(set(new_kinds))>=3:structural+=1
    if problems or new_kinds!=[e['kind'] for e in row['evidence']]:private.append({'case_id':row['case_id'],'split':row['split'],'issues':sorted(set(problems)),'kinds_after':new_kinds})
(ROOT/'runtime').mkdir(exist_ok=True)
(ROOT/'runtime/calibration_source_policy_audit.private.json').write_text(json.dumps(private,ensure_ascii=False,indent=2)+'\n')
report={'generated_at':datetime.now(timezone.utc).isoformat(),'rule_version':VERSION,'audit_kind':'Automatic structural checks derived from training calibration; no agent predictions',
    'frozen_candidate_pool_cases':len(rows),'cases_without_new_boundary_or_filename_flags':clean,
    'cases_with_new_flags':len(rows)-clean,'case_flag_counts_nonexclusive':dict(reasons),'evidence_kind_changes_nonexclusive':dict(changes),
    'cases_meeting_count_type_and_new_text_checks':structural,'frozen_pool_modified':False,'heldout_case_semantics_manually_read':False,
    'qualified_cases_added':0,'note':'No source sufficiency, necessary depth, human approval, or task accuracy is established by these counts.'}
(ROOT/'data/calibration_source_policy_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');print(json.dumps(report,ensure_ascii=False,indent=2))
