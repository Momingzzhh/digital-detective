"""Isolated re-annotation of source revisions; no formal dataset export."""
from collections import Counter
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import time
from pydantic import Field,create_model,ValidationError

from app.config import ROOT,settings
from app.five_dimensions import fingerprint,validate_annotation
from app.source_policy import require_clean_evidence
from app.annotation_quality import SemanticAudit,AUDIT_PROMPT,AUDIT_VERSION,audit_binding,quality_passed,obvious_quality_errors
from app.local_model import model_path
from scripts.annotation_lock import acquire_annotation_lock
from scripts.compact_annotation import INSTRUCTIONS,schema_for,payload,canonical,Strict
from scripts.annotate_five_dimensions import normalize_references
from scripts.export_review_bundle import verify

PROTOCOL='calibrated-feedback-bound-annotation-v2'
RULES='''本轮按已收到的训练组人工校准改进标注，仍只能引用本轮source_evidence。
裁判认定、公诉指控、文书身份部分以及人工意见都不替代可见证据。别名只收可见原文明示的对应，未显示就留空。
不能因某人出现在整段另一句就认定当前引用句支持其动作；先核对谁做什么及付款、物品的流转方向，不能把收付款写成同一人自我转交。
证人、普通交易者、配合侦查者、家属不自动是共犯；确有本案参与嫌疑且支持不足才用无法确认，不能收纳所有人。
同一交易不同议价阶段、后续退缴与早期未退、不同交易的金额都不自动形成矛盾。没有真实矛盾可留空，不必编造。
逻辑链从原始观察出发，推论的内容须实质依赖先前判断；独立的原始观察不能为凑深度而串联。
t_notes逐条说明本条证据层级的理由。整条含亲历、推测、听说等混合内容时，注明对应句子及限度，不把推测或转述说成直接观察；层级不等于真实性。
无法覆盖所有必要人物或经过时，如实填完整性false，不能为了通过检查补造支持。'''


def check_revision_rows(rows,parents):
    keys=('case_id','group_id','split','source','evidence','sentence_spans','source_fact_sha256')
    for row in rows:
        parent=parents.get(row['case_id'])
        if not parent or row['split']!='train' or parent['split']!='train' or row['group_id']!=parent['group_id']:
            raise ValueError('Only existing training groups can enter calibration re-annotation')
        if row.get('source_revision',{}).get('parent_source_binding')!=parent['source_binding']:
            raise ValueError('Source revision parent differs')
        if row['source_binding']!=fingerprint({k:row[k] for k in keys}):raise ValueError('Source revision binding differs')
        require_clean_evidence(row['evidence'])
        verify(row,{'raw_record':row['raw_record']})


def confirmed_feedback(row):
    wanted=set(row['source_revision']['feedback_bindings']);found=[]
    folder=ROOT/'data/review/five_dimensions_v2_calibration'/row['case_id']
    for path in folder.glob('*.json'):
        value=json.loads(path.read_text())
        if fingerprint(value) not in wanted:continue
        if (value.get('human_attestation') is not True or value['split']!='train'
            or value['source_binding']!=row['source_revision']['parent_source_binding']):
            raise ValueError('Feedback differs from the confirmed source revision')
        found.append(value)
    if len(found)!=len(wanted) or not found:raise ValueError('Confirmed feedback snapshot is missing')
    return found


def main(args):
    from scripts.annotate_five_dimensions_batch import bounded_generation,context_metadata,save
    dataset=Path(args.calibration_revision).resolve()
    base=(ROOT/'data/processed/five_dimensions_v2_calibration_revision').resolve()
    if not dataset.is_relative_to(base):raise ValueError('Revision must use the private calibration directory')
    rows=list(map(json.loads,dataset.open()))
    parents={r['case_id']:r for r in map(json.loads,(ROOT/'data/processed/five_dimensions_v2/candidates.private.jsonl').open())}
    check_revision_rows(rows,parents)
    if args.case_id:rows=[r for r in rows if r['case_id']==args.case_id]
    if not rows:raise ValueError('No requested training revision')
    if getattr(args,'audit_draft',None):return audit_draft(args,rows)
    if args.model_profile!='curator':raise ValueError('Calibration uses the fixed curator, never the trained agent')
    acquire_annotation_lock()
    os.environ.update(LOCAL_MODEL_PATH=str(ROOT/'models/qwen3-14b-mlx-4bit'),LOCAL_ADAPTER_PATH='',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_path()),local_files_only=True,trust_remote_code=False)
    context_limit=int(settings().get('LOCAL_CONTEXT_TOKENS','16384'))
    model_binding=fingerprint(json.loads((model_path()/'local_manifest.json').read_text()))
    output=ROOT/'data/review/five_dimensions_v2_calibration_revision'/dataset.parent.name;output.mkdir(parents=True,exist_ok=True)
    progress_path=ROOT/'runtime/calibration_revision_progress.json';counts=Counter();processed=0
    for row in rows[:args.limit]:
        feedback=confirmed_feedback(row)
        notes=create_model('TierNotes',__base__=Strict,**{e['id']:(str,Field(min_length=4,max_length=160)) for e in row['evidence']})
        schema=create_model('CalibratedAnnotation',__base__=schema_for(row),t_notes=(notes,...))
        code_bindings={name:__import__('hashlib').sha256((ROOT/name).read_bytes()).hexdigest() for name in
            ('scripts/annotate_calibration_revision.py','app/five_dimensions.py','app/source_policy.py','scripts/compact_annotation.py')}
        annotation_binding=fingerprint({'protocol':PROTOCOL,'rules':RULES,'instructions':INSTRUCTIONS,
            'source':row['source_binding'],'schema':schema.model_json_schema(),'model':model_binding,
            'feedback_bindings':row['source_revision']['feedback_bindings'],'code':code_bindings,
            'audit_prompt':AUDIT_PROMPT,'audit_version':AUDIT_VERSION})
        path=output/(row['case_id']+'.json')
        if path.exists() and json.loads(path.read_text()).get('annotation_binding')==annotation_binding:continue
        if path.exists():raise ValueError('Different calibration result exists; archive it explicitly before another protocol')
        begin=time.monotonic()
        def progress(event):save(progress_path,{'case_id':row['case_id'],'pid':os.getpid(),
            'updated_at':datetime.now(timezone.utc).isoformat(),**event})
        material={'source_evidence':payload(row)['source_evidence'],
            'human_calibration_feedback':[{'answers':f['answers'],'decision_reason':f['decision_reason']} for f in feedback],
            'feedback_scope':'反馈针对旧版。请逐条对照本轮可见证据处理；原文未展示的建议不能当作事实，反馈不是新增证据。'}
        messages=[{'role':'system','content':INSTRUCTIONS+'\n'+RULES+'\nSchema：'+json.dumps(schema.model_json_schema(),ensure_ascii=False)},
                  {'role':'user','content':json.dumps(material,ensure_ascii=False)}]
        result={'case_id':row['case_id'],'group_id':row['group_id'],'split':'train','generation_protocol':PROTOCOL,
            'source_binding':row['source_binding'],'parent_source_binding':row['source_revision']['parent_source_binding'],
            'annotation_binding':annotation_binding,'model_manifest_binding':model_binding,
            'human_reviewed':False,'formal_approval':False,'annotation_input_includes_direct_hidden_reference':False,
            'human_feedback_may_reference_hidden_source':True,'grounding_scope':'Current visible evidence only',
            'annotation_input_includes_human_feedback':True,'feedback_bindings':row['source_revision']['feedback_bindings'],
            'code_file_sha256':code_bindings,
            'actual_prompt_sha256':fingerprint(messages),'generated_at':datetime.now(timezone.utc).isoformat()}
        try:
            generated,stats,reserved=bounded_generation(tokenizer,[(messages,schema)],args.max_tokens,context_limit,progress)
            if reserved:
                result.update(status='local_context_reserved_for_later',**context_metadata(reserved[0]))
            else:
                _,text,tokens=generated[0];raw=schema.model_validate_json(text).model_dump()
                result.update(raw_compact_annotation=raw,prompt_tokens=tokens,generation_stats=stats)
                ann=canonical(raw,row)
                for tier in ann['evidence_hierarchy']:tier['reason']=raw['t_notes'][tier['evidence_id']]
                result['raw_annotation']=ann
                normalized,repairs=normalize_references(ann,row['sentence_spans'])
                result['identifier_repairs']=repairs
                ann=validate_annotation(row,normalized,row['sentence_spans'])
                result.update(status='machine_annotated_unverified',five_dimensions=ann)
                errors=obvious_quality_errors(ann)
                if errors:result.update(status='semantic_quality_rejected',error='; '.join(errors))
                else:
                    audit_request=[{'role':'system','content':AUDIT_PROMPT+'\nSchema：'+json.dumps(SemanticAudit.model_json_schema(),ensure_ascii=False)},
                        {'role':'user','content':json.dumps({'source':payload(row),'candidate':ann},ensure_ascii=False)}]
                    audited,audit_stats,audit_reserved=bounded_generation(tokenizer,[(audit_request,SemanticAudit)],1400,context_limit,
                        lambda event:progress(event|{'phase':'semantic_audit'}))
                    if audit_reserved:result.update(status='local_context_reserved_for_later',semantic_audit_context=context_metadata(audit_reserved[0]))
                    else:
                        result['semantic_audit_raw']=audited[0][1]
                        result['semantic_audit']=SemanticAudit.model_validate_json(audited[0][1]).model_dump()
                        result['semantic_audit_binding']=audit_binding(row,ann)
                        result['semantic_audit_stats']=audit_stats
                        if not quality_passed(result,row):result.update(status='semantic_quality_rejected',error='Local semantic screen failed')
        except ValidationError as exc:result.update(status='local_annotation_error',error=str(exc)[:500])
        except ValueError as exc:result.update(status='complexity_or_reference_rejected',error=str(exc)[:500])
        except Exception as exc:result.update(status='local_annotation_error',error=str(exc)[:500])
        result['seconds']=round(time.monotonic()-begin,2);save(path,result)
        counts[result['status']]+=1;processed+=1
        progress({'stage':'saved','status':result['status']})
        print(json.dumps({'processed':processed,'status':result['status'],'seconds':result['seconds']},ensure_ascii=False),flush=True)
    report={'generated_at':datetime.now(timezone.utc).isoformat(),'protocol':PROTOCOL,
        'attempted_training_revisions':processed,'status_counts':dict(counts),
        'formal_dataset_changed':False,'heldout_materials_used':False,'external_inference_api_calls':0,
        'qualified_cases_added':0,'formal_training_complete':False}
    save(ROOT/'data/calibration_revision_report.json',report);print(json.dumps(report),flush=True)


def audit_draft(args,rows):
    from scripts.annotate_five_dimensions_batch import bounded_generation,save
    draft_path=Path(args.audit_draft).resolve()
    private=(ROOT/'data/review/five_dimensions_v2_calibration_revision').resolve()
    if not draft_path.is_relative_to(private):raise ValueError('Draft must remain in the private revision directory')
    draft=json.loads(draft_path.read_text());row=next((r for r in rows if r['case_id']==draft['case_id']),None)
    if not row or row['source_binding']!=draft['source_binding']:raise ValueError('Draft source differs')
    if args.model_profile!='curator':raise ValueError('Use the fixed curator for semantic screening')
    ann=validate_annotation(row,draft['raw_annotation'],row['sentence_spans'])
    errors=obvious_quality_errors(ann)
    if errors:raise ValueError('; '.join(errors))
    acquire_annotation_lock()
    os.environ.update(LOCAL_MODEL_PATH=str(ROOT/'models/qwen3-14b-mlx-4bit'),LOCAL_ADAPTER_PATH='',
                      HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_HUB_DISABLE_TELEMETRY='1')
    from transformers import AutoTokenizer
    tokenizer=AutoTokenizer.from_pretrained(str(model_path()),local_files_only=True,trust_remote_code=False)
    request=[{'role':'system','content':AUDIT_PROMPT+'\nSchema：'+json.dumps(SemanticAudit.model_json_schema(),ensure_ascii=False)},
        {'role':'user','content':json.dumps({'source':payload(row),'candidate':ann},ensure_ascii=False)}]
    result=draft|{'five_dimensions':ann,'status':'machine_annotated_unverified','human_reviewed':False,
        'formal_approval':False,'semantic_audit_prompt_sha256':fingerprint(request),
        'semantic_audit_model_binding':fingerprint(json.loads((model_path()/'local_manifest.json').read_text()))}
    def progress(event):save(ROOT/'runtime/calibration_revision_progress.json',{
        'case_id':row['case_id'],'phase':'assistant_draft_semantic_audit','updated_at':datetime.now(timezone.utc).isoformat(),**event})
    generated,stats,reserved=bounded_generation(tokenizer,[(request,SemanticAudit)],1400,
        int(settings().get('LOCAL_CONTEXT_TOKENS','16384')),progress)
    if reserved:result.update(status='local_context_reserved_for_later')
    else:
        result['semantic_audit_raw']=generated[0][1]
        result['semantic_audit']=SemanticAudit.model_validate_json(generated[0][1]).model_dump()
        result['semantic_audit_binding']=audit_binding(row,ann);result['semantic_audit_stats']=stats
        if not quality_passed(result,row):result.update(status='semantic_quality_rejected')
    result['semantic_audit_completed_at']=datetime.now(timezone.utc).isoformat()
    destination=draft_path.with_name(draft_path.stem+'.screened.json');save(destination,result)
    progress({'stage':'saved','status':result['status']})
    print(json.dumps({'status':result['status'],'qualified_cases_added':0,'output':str(destination)},ensure_ascii=False),flush=True)
