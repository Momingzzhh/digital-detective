"""Private review material keeps training identities and untrusted text inert."""
import copy
from html.parser import HTMLParser
import json

import pytest

from scripts import prepare_small_batch_review as reader
from test_small_batch_training import automatic_policy, freeze


def setup(tmp_path, monkeypatch):
    value, _ = automatic_policy(tmp_path)
    rows, manifest = freeze(tmp_path, value)
    for name in ('prepare_small_batch_review.py', 'prepare_calibration_review.py'):
        path = tmp_path / 'scripts' / name
        path.parent.mkdir(exist_ok=True); path.write_text('synthetic renderer snapshot')
    monkeypatch.setattr(reader, 'ROOT', tmp_path)
    return rows, manifest


def test_only_new_automatic_training_cases_are_packed_without_signoff_or_hidden_key(tmp_path, monkeypatch):
    rows, manifest = setup(tmp_path, monkeypatch)
    accepted = copy.deepcopy(rows[0]); accepted['review_status'] = 'user_accepted_corrected_version_pending_benchmark_review'
    monkeypatch.setattr(reader, 'load_dataset', lambda *args: ([accepted, *rows], manifest))
    folder = reader.prepare(tmp_path / 'unused')
    saved = json.loads((folder / 'reading.private.json').read_text())
    assert len(saved['cases']) == 1 and 'hidden_gold' not in saved['cases'][0]
    public = json.loads((tmp_path / 'data/small_batch_pending_review_report.json').read_text())
    assert public['human_approvals_created'] == public['human_reviews_completed'] == 0
    assert public['pending_training_cases_in_reading_pack'] == 1
    tags = []
    class Tags(HTMLParser):
        def handle_starttag(self, tag, attrs): tags.append(tag)
    Tags().feed((folder / 'review.html').read_text())
    assert not {'script', 'button', 'form', 'input', 'iframe', 'img'} & set(tags)


@pytest.mark.parametrize('defect', ['test', 'human_reviewed', 'formal_approval'])
def test_reading_pack_cannot_silently_expose_heldout_or_claim_reviewed_cases(tmp_path, monkeypatch, defect):
    rows, manifest = setup(tmp_path, monkeypatch)
    if defect == 'test': rows[0]['split'] = 'test'
    else: rows[0][defect] = True
    monkeypatch.setattr(reader, 'load_dataset', lambda *args: (rows, manifest))
    with pytest.raises(ValueError, match='unreviewed training'): reader.prepare(tmp_path / 'unused')


def test_source_and_annotation_markup_are_escaped_as_text(tmp_path, monkeypatch):
    rows, manifest = setup(tmp_path, monkeypatch)
    attack = '</pre><script src="https://invalid.example/steal"></script>'
    rows[0]['evidence'][0]['text'] += attack
    rows[0]['raw_record']['qw'] += attack
    rows[0]['five_dimensions']['actor_assessments'][0]['role'] += attack
    monkeypatch.setattr(reader, 'load_dataset', lambda *args: (rows, manifest))
    folder = reader.prepare(tmp_path / 'unused')
    markup = (folder / 'review.html').read_text()
    assert '<script' not in markup and '&lt;script' in markup
