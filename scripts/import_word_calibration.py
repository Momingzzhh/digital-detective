#!/usr/bin/env python3
"""Archive returned Word feedback against a private training-only snapshot.

No model calls, source edits, dataset approvals, or inferred signatures.
"""
import argparse
from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
RUBRIC='train-only-calibration-v1'
PROTOCOL='word-calibration-feedback-v1'
NS={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    'cp':'http://schemas.openxmlformats.org/package/2006/metadata/core-properties',
    'dc':'http://purl.org/dc/elements/1.1/'}
SECTIONS={'一 人物与角色':'actors','二 关键经过':'events','三 证据层级与材料质量':'evidence',
          '四 逻辑链与必要推理深度':'depth','五 真实矛盾':'conflicts','六 对话轮次记录':'dialogue'}
DECISIONS={'需要修正':'needs_revision','需要修正标注':'needs_revision','不满足要求应排除':'exclude',
           '本次未发现需修改项':'no_correction_found','暂无法判定':'uncertain'}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def read_docx(path):
    with ZipFile(path) as archive:
        if sum(i.file_size for i in archive.infolist())>20*1024*1024:raise ValueError('Word package is too large')
        root=ET.fromstring(archive.read('word/document.xml'))
        if root.findall('.//w:ins',NS) or root.findall('.//w:del',NS):
            raise ValueError('Unresolved tracked changes require review before import')
        paragraphs=[]
        for para in root.findall('.//w:body//w:p',NS):
            text=''.join(t.text or '' for t in para.findall('.//w:t',NS)).strip()
            if text:paragraphs.append(text)
        core=ET.fromstring(archive.read('docProps/core.xml'))
        comments=[]
        if 'word/comments.xml' in archive.namelist():
            croot=ET.fromstring(archive.read('word/comments.xml'))
            comments=[''.join(t.text or '' for t in c.findall('.//w:t',NS)) for c in croot]
    return paragraphs,core.findtext('cp:keywords','',NS),core.findtext('dc:description','',NS),comments


def parse_feedback(path, manifest, template_paragraphs, confirmation=None):
    paragraphs,_,description,comments=read_docx(path)
    text='\n'.join(paragraphs)
    cid_match=re.search(r'案例编号\s*([A-Za-z0-9_-]+)',text)
    version_match=re.search(r'材料版本\s*([a-f0-9]{16,64})',text)
    if not cid_match or not version_match:raise ValueError('Missing visible case or material version')
    cid=cid_match.group(1);version=version_match.group(1)
    expected=fingerprint({'rubric':manifest['rubric'],'cases':manifest['cases']})
    if manifest['rubric']!=RUBRIC or manifest['bundle_binding']!=expected or not expected.startswith(version):
        raise ValueError('Material snapshot binding differs')
    embedded=re.search(r'bundle_binding=([a-f0-9]{64})',description)
    if embedded and embedded.group(1)!=expected:raise ValueError('Embedded material binding differs')
    case=next((c for c in manifest['cases'] if c['case_id']==cid),None)
    if not case or case['split']!='train':raise ValueError('Calibration only accepts known training cases')
    reviewer_match=re.search(r'核对人或研究编号\s+([^\n]+)',text)
    reviewer=re.split(r'单位或研究组',reviewer_match.group(1))[0].strip() if reviewer_match else ''
    if not reviewer or any(s in reviewer for s in ('待填写','【','】')):raise ValueError('Missing actual reviewer identity')
    day_match=re.search(r'核对日期\s+(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',text)
    if not day_match:raise ValueError('Missing actual review date')
    reviewed_date=date(*map(int,day_match.groups())).isoformat()
    # Match the immutable template text, rather than assuming answers have a
    # particular number of paragraphs or remained in the same Word styles.
    template_text=set(template_paragraphs);answers={};active=None
    for para in paragraphs:
        if para in SECTIONS:
            active=SECTIONS[para];answers[active]=[];continue
        if para=='处理建议及本人确认':active=None
        if active and para not in template_text:
            if '待填写' in para:raise ValueError('Unfilled feedback section')
            answers[active].append(para)
    answers={k:'\n'.join(v) for k,v in answers.items()}
    if set(answers)!=set(SECTIONS.values()) or not all(v.strip() for v in answers.values()):
        raise ValueError('Each Word feedback section needs an answer')
    decision_line=next((p for p in paragraphs if p.startswith('处理建议 ') and '请选择' not in p),'')
    decision_text=re.sub(r'^处理建议\s*','',decision_line).strip(' 【】[]')
    if decision_text not in DECISIONS:raise ValueError('Missing explicit calibration decision')
    reason=next((re.sub(r'^主要理由或待协调分歧[：:\s]*','',p) for p in paragraphs
                 if p.startswith('主要理由或待协调分歧') and '待填写' not in p),'')
    attested=False;basis='awaiting_confirmation'
    attestation_line=next((p for p in paragraphs if p.startswith('亲自核对确认')),'')
    document_signed=(bool(re.search(r'亲自核对确认\s*[【\[]?确认[】\]]?\s+确认人及日期',attestation_line))
        and reviewer in attestation_line and all(s not in attestation_line for s in ('待填写','请填写','不确认')))
    if document_signed:
        signed_day=re.search(r'(\d{4})[.\-/](\d{1,2})[.\-/](\d{1,2})',attestation_line)
        document_signed=bool(signed_day and date(*map(int,signed_day.groups())).isoformat()==reviewed_date)
    if document_signed:attested=True;basis='completed_word_confirmation'
    # The template's boilerplate declaration is never treated as a signature.
    if confirmation is not None:
        if (confirmation.get('reviewer_id')!=reviewer or confirmation.get('reviewed_date')!=reviewed_date
            or confirmation.get('human_attestation') is not True or cid not in confirmation.get('case_ids',[])
            or confirmation.get('source')!='explicit_user_message' or not confirmation.get('user_answer')):
            raise ValueError('Human confirmation does not match this returned document')
        attested=True;basis='explicit_user_message'
    read_full=re.search(r'本次是否通读完整来源\s*([^\n]+)',text)
    read_full_value=read_full.group(1).strip() if read_full else ''
    return {'feedback_protocol':PROTOCOL,'rubric':RUBRIC,'bundle_binding':expected,
        'case_id':cid,'split':'train','source_binding':case['source_binding'],
        'annotation_result_binding':case['annotation_result_binding'],
        'reviewer_id':reviewer,'reviewed_date':reviewed_date,
        'reviewed_at':None,'review_time_precision':'date_only',
        'read_full_source':True if read_full_value=='是' else (False if read_full_value=='否' else None),
        'decision':DECISIONS[decision_text],'answers':answers,'decision_reason':reason,
        'human_attestation':attested,'attestation_basis':basis,
        'document_attestation_completed':document_signed,'word_comments':comments,
        'original_paragraphs':paragraphs,'input_file_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'confirmation_binding':fingerprint(confirmation) if confirmation else None,
        'formal_approval':False}


def import_files(files,manifest_path,template_paths,confirmation_path=None,write=False):
    manifest=json.loads(manifest_path.read_text())
    confirmation=json.loads(confirmation_path.read_text()) if confirmation_path else None
    parsed=[]
    for file in files:
        paragraphs,_,_,_=read_docx(file)
        match=re.search(r'案例编号\s*([A-Za-z0-9_-]+)','\n'.join(paragraphs))
        if not match or match.group(1) not in template_paths:raise ValueError('Missing immutable template')
        template,_,_,_=read_docx(template_paths[match.group(1)])
        parsed.append((file,parse_feedback(file,manifest,template,confirmation)))
    if len({r['input_file_sha256'] for _,r in parsed})!=len(parsed):raise ValueError('Duplicate returned file')
    if write:
        base=ROOT/'data/review/five_dimensions_v2_calibration'
        for file,review in parsed:
            digest=fingerprint(review);folder=base/review['case_id'];folder.mkdir(parents=True,exist_ok=True)
            archived=base/'documents'/(review['input_file_sha256']+'.docx');archived.parent.mkdir(exist_ok=True)
            if archived.exists() and archived.read_bytes()!=file.read_bytes():raise ValueError('Archived Word bytes differ')
            if not archived.exists():shutil.copyfile(file,archived)
            target=folder/(digest+'.json')
            if not target.exists():
                review=review|{'received_at':datetime.now(timezone.utc).isoformat(),'original_document':str(archived.relative_to(ROOT))}
                target.write_text(json.dumps(review,ensure_ascii=False,indent=2)+'\n')
        if confirmation_path:
            dest=base/'confirmations'/(fingerprint(confirmation)+'.json');dest.parent.mkdir(exist_ok=True)
            if not dest.exists():dest.write_bytes(confirmation_path.read_bytes())
    return {'written':write,'returned_documents':len(parsed),'training_cases':len({r['case_id'] for _,r in parsed}),
        'attested_feedback_cases':len({r['case_id'] for _,r in parsed if r['human_attestation']}),
        'decision_counts':dict(Counter(r['decision'] for _,r in parsed)),
        'formal_approvals_created':0,'qualified_cases_added':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('files',nargs='+',type=Path)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--templates',type=Path,required=True)
    parser.add_argument('--confirmation-file',type=Path)
    parser.add_argument('--write',action='store_true')
    args=parser.parse_args();templates={}
    for path in args.templates.rglob('*_人工校准表.docx'):
        paragraphs,_,_,_=read_docx(path)
        match=re.search(r'案例编号\s*([A-Za-z0-9_-]+)','\n'.join(paragraphs))
        if match:templates[match.group(1)]=path
    result=import_files(args.files,args.manifest,templates,args.confirmation_file,args.write)
    if args.write:
        report=ROOT/'data/calibration_feedback_report.json'
        report.write_text(json.dumps({'generated_at':datetime.now(timezone.utc).isoformat(),
            'protocol':PROTOCOL,'scope':'Training-only feedback, not formal heldout source approvals',**result},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
