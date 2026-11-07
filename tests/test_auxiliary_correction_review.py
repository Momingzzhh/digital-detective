import copy
import hashlib
import json

import pytest

from app.five_dimensions import validate_auxiliary_annotation
from scripts.prepare_auxiliary_correction_review import load_draft,render
from test_auxiliary_training import inputs


def package(root):
    row,original=inputs()
    source=root/'source.jsonl';source.write_text(json.dumps(row)+'\n')
    ann=original['raw_annotation'];checked=validate_auxiliary_annotation(row,ann,row['sentence_spans'])
    draft={k:row[k] for k in ('case_id','split','source_binding')}
    draft.update(status='auxiliary_annotation_pending_human_review',human_reviewed=False,
                 formal_approval=False,qualified_cases_added=0,raw_annotation=ann,five_dimensions=checked,
                 computed_graph_depth=2,source_dataset={'path':source.name,'sha256':hashlib.sha256(source.read_bytes()).hexdigest()},
                 review_checks=[{'question':'合成核对问题','sentence_ids':['S1']}])
    path=root/'draft.json';path.write_text(json.dumps(draft))
    return path


def test_complete_new_correction_is_still_pending_and_separate_from_formal_data(tmp_path):
    draft,row=load_draft(tmp_path,package(tmp_path))
    assert not draft['human_reviewed'] and not draft['formal_approval']
    assert draft['five_dimensions']['reasoning_depth']==2 and row['split']=='train'


@pytest.mark.parametrize('defect',['heldout','source_changed','annotation_changed','unknown_question_ref'])
def test_review_packet_rejects_changed_or_unbound_material(tmp_path,defect):
    path=package(tmp_path);draft=json.loads(path.read_text())
    if defect=='heldout':draft['split']='test'
    elif defect=='source_changed':(tmp_path/'source.jsonl').write_text('{}\n')
    elif defect=='annotation_changed':draft['five_dimensions']['reasoning_depth']=3
    else:draft['review_checks'][0]['sentence_ids']=['NOT-PUBLISHED']
    path.write_text(json.dumps(draft))
    with pytest.raises(ValueError):load_draft(tmp_path,path)


def test_source_and_review_strings_remain_escaped_data_in_the_offline_page(tmp_path):
    draft,row=load_draft(tmp_path,package(tmp_path))
    payload='</pre><script>upload_private_source()</script>'
    draft['review_checks'][0]['question']=payload
    row['raw_record']['qw']=payload
    page=render(draft,row,'synthetic-version')
    assert '<script>' not in page and '&lt;script&gt;' in page
    assert 'http://' not in page and 'https://' not in page
