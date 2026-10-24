import json
from pathlib import Path
from zipfile import ZipFile
import pytest
from scripts import import_word_calibration as word


def write_docx(path,paragraphs,description=''):
    from xml.etree.ElementTree import Element,SubElement,tostring
    w=word.NS['w'];root=Element('{'+w+'}document');body=SubElement(root,'{'+w+'}body')
    for text in paragraphs:
        p=SubElement(body,'{'+w+'}p');r=SubElement(p,'{'+w+'}r');SubElement(r,'{'+w+'}t').text=text
    core=Element('{'+word.NS['cp']+'}coreProperties');SubElement(core,'{'+word.NS['dc']+'}description').text=description
    with ZipFile(path,'w') as archive:
        archive.writestr('word/document.xml',tostring(root));archive.writestr('docProps/core.xml',tostring(core))


def fixture(tmp_path):
    case={'case_id':'synthetic','split':'train','source_binding':'source','annotation_result_binding':'annotation'}
    manifest={'rubric':word.RUBRIC,'cases':[case]};manifest['bundle_binding']=word.fingerprint(manifest)
    template=['样本01人工校准表','案例编号 synthetic材料版本 '+manifest['bundle_binding'][:16],
              '核对人或研究编号 【待填写】','核对日期 【待填写】 本次是否通读完整来源 【是／否／部分】']
    returned=template[:2]+['核对人或研究编号 合成机制核对人','核对日期 2026.10.24 本次是否通读完整来源 是']
    for heading in word.SECTIONS:
        template += [heading,'这是模板说明','【待填写】']
        returned += [heading,'这是模板说明','仅为机制测试的核对意见']
    template += ['处理建议及本人确认','处理建议 【请选择】','本人确认已亲自对照原文。','亲自核对确认 【请填写 确认／不确认】 确认人及日期 【待填写】']
    returned += ['处理建议及本人确认','处理建议 【需要修正】',*template[-2:]]
    source=tmp_path/'returned.docx';base=tmp_path/'template.docx'
    write_docx(source,returned,'bundle_binding='+manifest['bundle_binding']);write_docx(base,template)
    confirmation={'source':'explicit_user_message','reviewer_id':'合成机制核对人','reviewed_date':'2026-10-24',
        'human_attestation':True,'case_ids':['synthetic'],'user_answer':'仅为测试的显式确认'}
    return source,base,manifest,template,confirmation


def test_template_boilerplate_cannot_become_human_attestation(tmp_path):
    source,_,manifest,template,_=fixture(tmp_path)
    result=word.parse_feedback(source,manifest,template)
    assert result['human_attestation'] is False and result['formal_approval'] is False
    assert result['reviewed_at'] is None and result['review_time_precision']=='date_only'
    assert result['decision']=='needs_revision' and len(result['answers'])==6


@pytest.mark.parametrize('defect',['identity','date','case','unsigned'])
def test_human_confirmation_must_match_actual_document(tmp_path,defect):
    source,_,manifest,template,confirmation=fixture(tmp_path)
    if defect=='identity':confirmation['reviewer_id']='different'
    elif defect=='date':confirmation['reviewed_date']='2026-10-23'
    elif defect=='case':confirmation['case_ids']=[]
    else:confirmation['human_attestation']=False
    with pytest.raises(ValueError,match='confirmation'):
        word.parse_feedback(source,manifest,template,confirmation)


def test_changed_snapshot_and_heldout_material_cannot_be_imported(tmp_path):
    source,_,manifest,template,confirmation=fixture(tmp_path)
    manifest['cases'][0]['split']='test'
    with pytest.raises(ValueError,match='binding'):
        word.parse_feedback(source,manifest,template,confirmation)


def test_archive_is_idempotent_and_never_creates_formal_approvals(tmp_path,monkeypatch):
    source,base,manifest,_,confirmation=fixture(tmp_path)
    m=tmp_path/'manifest.json';m.write_text(json.dumps(manifest))
    c=tmp_path/'confirmation.json';c.write_text(json.dumps(confirmation))
    monkeypatch.setattr(word,'ROOT',tmp_path)
    for _ in range(2):
        result=word.import_files([source],m,{'synthetic':base},c,write=True)
        assert result['attested_feedback_cases']==1 and result['formal_approvals_created']==0
    folder=tmp_path/'data/review/five_dimensions_v2_calibration'
    assert len(list((folder/'synthetic').glob('*.json')))==1
    assert len(list((folder/'documents').glob('*.docx')))==1
    assert not (tmp_path/'data/review/five_dimensions_v2_human').exists()


def test_actual_word_confirmation_needs_matching_identity_and_date(tmp_path):
    source,_,manifest,template,_=fixture(tmp_path)
    paragraphs,_,description,_=word.read_docx(source)
    paragraphs[-1]='亲自核对确认 确认 确认人及日期 合成机制核对人 2026.10.24'
    write_docx(source,paragraphs,description)
    result=word.parse_feedback(source,manifest,template)
    assert result['human_attestation'] is True and result['document_attestation_completed'] is True
    paragraphs[-1]='亲自核对确认 确认 确认人及日期 合成机制核对人 2026.10.23'
    write_docx(source,paragraphs,description)
    assert word.parse_feedback(source,manifest,template)['human_attestation'] is False
