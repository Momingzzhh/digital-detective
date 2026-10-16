#!/usr/bin/env python3
"""Export a private historical research stratum, never a certified benchmark.

Re-read pinned Parquet records rather than trusting generated checkpoints.
English transcript, Chinese translation and hidden labels remain separate.
"""
from collections import Counter
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_oldbailey import extract,sha
from build_oldbailey import Annotation,SourceAudit,TranslationAudit,Translation,chinese_evidence,sentence_spans,span_fingerprint,actors_match
from fetch_lecardv2 import digest
from export_review_bundle import write_rows


def visible(row):
    return {k:row[k] for k in ('case_id','group_id','split','stratum','jurisdiction_context','participant_references')} | {
        'evidence':[{'id':e['id'],'kind':e['kind'],'text':e['text']} for e in row['evidence']]}


def visible_text(row):
    view=visible(row)
    lines=['英国历史庭审研究材料','中文内容为原始英文记录的译文。',
        '法域：England；庭审日期：'+str(view['jurisdiction_context']['trial_date']),
        '中性称谓对应（不代表有罪）：']
    for p in view['participant_references']:
        lines.append('/'.join(p['source_reference_terms'])+' = '+p['name_en'])
    lines.append('可用材料：')
    lines.extend(e['id']+'\n'+e['text'] for e in view['evidence'])
    return '\n\n'.join(lines)+'\n'


def verify(row,source):
    Annotation.model_validate(row['annotation'])
    SourceAudit.model_validate(row['english_source_audit'])
    TranslationAudit.model_validate(row['translation_audit'])
    if row['source']!=source['source']:raise ValueError('Source metadata changed')
    for key in ('case_id','group_id','split','stratum','participant_references','jurisdiction_context'):
        if row[key]!=source[key]:raise ValueError('Identity, grouping or context changed')
    spans=sentence_spans(source['evidence'])
    if row.get('sentence_spans')!=spans or row.get('span_fingerprint')!=span_fingerprint(source,spans):
        raise ValueError('Missing or stale source/segmentation binding')
    translated,cn=chinese_evidence(source,spans,Translation.model_validate(row['translation']))
    if len(translated)!=len(row['evidence']):raise ValueError('Evidence count changed')
    for actual,expected in zip(row['evidence'],translated):
        for key in expected:
            if key!='translation_version' and actual.get(key)!=expected[key]:
                raise ValueError('English evidence or Chinese translation changed')
    for key in ('english_source_audit','translation_audit'):
        if not row.get(key) or not all(v is True for k,v in row[key].items() if k not in ('reason','identified_perpetrators','unresolved_actors')):
            raise ValueError('Audit missing or failed')
    if (row['annotation'].get('unresolved_actors') or row['english_source_audit'].get('unresolved_actors') or
        not actors_match(row['annotation'].get('perpetrators',[]),source) or
        not actors_match(row['english_source_audit'].get('identified_perpetrators',[]),source)):
        raise ValueError('Visible actor set not complete source gold')
    gold=row['hidden_gold']
    for key,value in source['hidden_gold'].items():
        if gold[key]!=value:raise ValueError('Source historical verdict/charge joins changed')
    offence=source['hidden_gold']['offences'][0]
    if gold['charges']!=[offence['category']+'/'+offence['subcategory']]:raise ValueError('Historical class changed')
    index={s['id']:s for s in spans}
    ids=row['annotation']['event_sentence_ids']
    if not 2<=len(ids)<=12 or len(set(ids))!=len(ids):raise ValueError('Invalid chronology selection')
    if row['english_gold_events']!=[index[x] for x in ids]:raise ValueError('English event source changed')
    if len(gold['chronology'])!=len(ids):raise ValueError('Chronology coverage changed')
    for event,sid in zip(gold['chronology'],ids):
        unit=cn[sid];english=index[sid]
        if (event['sentence_id']!=sid or event['english_source_sentence']!=english or
            event['evidence_ids']!=[english['evidence_id']] or
            event['sentence_start']!=unit['start'] or event['sentence_end']!=unit['end'] or
            event['summary']!=unit['text']):raise ValueError('Chinese/English supporting sentence changed')


def main():
    directory=ROOT/'data/review/oldbailey_v1'
    checkpoints=[json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
    checkpoints=[r for r in checkpoints if 'case_id' in r and 'status' in r]
    passes={r['case_id']:r for r in checkpoints if r['status']=='machine_pass_translated_unverified'}
    manifest=json.loads((ROOT/'data/oldbailey_download_manifest.json').read_text())
    selected=[];excluded=Counter();groups=set();classes=Counter()
    seen=set()
    quarantine_path=ROOT/'data/review/agent_audit/oldbailey_v1/quarantine.json'
    quarantine=json.loads(quarantine_path.read_text())['case_ids'] if quarantine_path.exists() else []
    development_path=directory/'development_cohort.json'
    development_groups={r['group_id'] for r in json.loads(development_path.read_text())['cases']} if development_path.exists() else set()
    for key,expected in manifest['files'].items():
        path=ROOT/'data/raw/oldbailey'/Path(key).name
        if path.stat().st_size!=expected['bytes'] or digest(path)!=expected['sha256']:
            raise ValueError('Pinned source Parquet is missing or changed')
        for batch_index,batch in enumerate(pq.ParquetFile(path).iter_batches(batch_size=128)):
            for index,raw in enumerate(batch.to_pylist()):
                cid='oldbailey-'+raw['trial_id']
                if cid not in passes:continue
                seen.add(cid);row=passes[cid]
                source,reason=extract(raw)
                if not source:excluded['current_source_filter:'+reason]+=1;continue
                source['source'].update(dataset=manifest['dataset'],revision=manifest['revision'],
                    parquet=path.name,row_index=batch_index*128+index,license=manifest['license'])
                try:verify(row,source)
                except (ValueError,KeyError,TypeError):excluded['integrity_or_stale_binding']+=1;continue
                if cid in quarantine:excluded['independent_agent_audit']+=1;continue
                if row['split']=='test' and row['group_id'] in development_groups:
                    excluded['development_group_not_heldout']+=1;continue
                if row['group_id'] in groups:excluded['same_defendant_name_group']+=1;continue
                groups.add(row['group_id']);selected.append(row);classes.update(row['hidden_gold']['charges'])
    if set(passes)!=seen:raise ValueError('Checkpoint trial absent from pinned source')
    selected.sort(key=lambda r:sha('oldbailey-curation-order-v1:'+r['case_id']))
    out=ROOT/'data/processed/review_bundle/oldbailey_v1'
    write_rows(out/'machine_curated.unverified.jsonl',selected)
    for split in ('train','dev','test'):
        subset=[r for r in selected if r['split']==split]
        write_rows(out/'visible'/(split+'.jsonl'),[visible(r) for r in subset])
        write_rows(out/'hidden'/(split+'.jsonl'),[{
            'case_id':r['case_id'],'hidden_gold':r['hidden_gold'],'source':r['source'],
            'english_evidence':[{k:v for k,v in e.items() if k not in ('text','translation_version')} for e in r['evidence']],
            'translation':r['translation'],'english_source_audit':r['english_source_audit'],
            'translation_audit':r['translation_audit']} for r in subset])
    export_id=sha(''.join(digest(out/'visible'/(s+'.jsonl')) for s in ('train','dev','test')))[:16]
    text_root=out/'visible_text'/export_id
    text_hashes={};text_bytes=0
    for row in selected:
        path=text_root/row['split']/(row['case_id']+'.txt');path.parent.mkdir(parents=True,exist_ok=True)
        payload=visible_text(row).encode();path.write_bytes(payload)
        text_hashes[str(path.relative_to(text_root))]=hashlib.sha256(payload).hexdigest();text_bytes+=len(payload)
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'stratum':'england_historical_1834_1913',
        'dataset':manifest['dataset'],'revision':manifest['revision'],'license':manifest['license'],
        'checkpoints':len(checkpoints),'status_counts':dict(Counter(r['status'] for r in checkpoints)),
        'checkpoint_versions':dict(Counter(r['pipeline_version'] for r in checkpoints)),
        'exported_pipeline_versions':dict(Counter(r['pipeline_version'] for r in selected)),
        'machine_curated_cases':len(selected),'machine_curated_groups':len(groups),
        'excluded_after_export_audit':dict(excluded),'split_counts':dict(Counter(r['split'] for r in selected)),
        'historical_class_counts':dict(classes),'qualified':0,'human_reviewed':0,
        'target_1000_complete':False,'accuracy_90_verified':False,
        'uploadable_text_bundle':{'directory':'visible_text/'+export_id,'cases':len(selected),
            'bytes':text_bytes,'tree_sha256':sha(json.dumps(text_hashes,sort_keys=True))},
        'artifacts':{str(p.relative_to(out)):{'sha256':digest(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*.jsonl'))},
        'limitations':['Machine curation and translation require independent source review.',
            'Single-defendant guilty-only selection makes a fixed accused-person culprit baseline trivial.',
            'Historical verdicts and classes are recorded source labels, not objective guilt or modern Chinese law.',
            'Report this stratum separately; never combine its score into modern criminal investigation accuracy.',
            'Name grouping cannot guarantee all related cases are separated; pretrained-model memorization is unknown.',
            'CC-BY-NC-4.0 source and derived translations are for attributed noncommercial use; MIT applies only to code.']}
    (out/'manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (ROOT/'data/oldbailey_export_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ('machine_curated_cases','split_counts','excluded_after_export_audit','qualified')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
