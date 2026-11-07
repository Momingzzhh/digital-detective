"""Conservative, source-bound exclusions from signed training calibration feedback."""
import json
import hashlib
from .config import ROOT
from .five_dimensions import fingerprint


def complexity_exclusion(row):
    path=ROOT/'data/review/five_dimensions_v2_complexity_exclusions'/(row['case_id']+'.json')
    if not path.exists():return None
    value=json.loads(path.read_text())
    if row['source_binding'] not in value['excluded_source_bindings']:return None
    if (value.get('case_id')!=row['case_id'] or value.get('group_id')!=row['group_id']
        or row['split']!='train' or value.get('split')!='train'
        or type(value.get('maximum_reviewed_depth')) is not int
        or not 1<=value['maximum_reviewed_depth']<3):
        raise ValueError('Invalid source-bound complexity exclusion')
    feedback_path=(ROOT/value['feedback_path']).resolve()
    private=(ROOT/'data/review/five_dimensions_v2_calibration').resolve()
    if not feedback_path.is_relative_to(private):raise ValueError('Complexity feedback must remain private')
    review=json.loads(feedback_path.read_text())
    if (fingerprint(review)!=value['feedback_binding'] or review.get('human_attestation') is not True
        or review.get('case_id')!=row['case_id'] or review.get('split')!='train'
        or not review.get('reviewer_id') or not review.get('reviewed_date')
        or review.get('source_binding')!=value['reviewed_source_binding']
        or review.get('decision') not in ('needs_revision','exclude')
        or not review.get('answers',{}).get('depth')):
        raise ValueError('Complexity exclusion lacks matching signed feedback')
    return value


def source_review_hold(row):
    """Keep a confirmed request for revision out of the frozen source version."""
    if row['split']!='train':return None
    folder=ROOT/'data/review/five_dimensions_v2_calibration'/row['case_id']
    confirmed=[]
    for path in folder.glob('*.json'):
        value=json.loads(path.read_text())
        if (value.get('source_binding')!=row['source_binding'] or value.get('split')!='train'
            or value.get('human_attestation') is not True):continue
        if not value.get('reviewer_id') or not value.get('reviewed_date'):
            raise ValueError('Incomplete signed calibration feedback')
        if value.get('feedback_protocol')=='word-calibration-feedback-v1':
            if not value.get('original_document') or not value.get('input_file_sha256'):
                raise ValueError('Word calibration archive binding is missing')
            document=(ROOT/value['original_document']).resolve()
            private=(ROOT/'data/review/five_dimensions_v2_calibration/documents').resolve()
            if (not document.is_relative_to(private) or
                hashlib.sha256(document.read_bytes()).hexdigest()!=value['input_file_sha256']):
                raise ValueError('Calibration feedback document hash differs')
        else:
            if not value.get('bundle_binding') or not value.get('annotation_result_binding'):
                raise ValueError('Calibration feedback snapshot binding is missing')
            if value.get('feedback_protocol')=='html-calibration-feedback-v1':
                submission=(ROOT/value['original_submission']).resolve()
                private=(ROOT/'data/review/five_dimensions_v2_calibration/submissions').resolve()
                if (not submission.is_relative_to(private) or
                    hashlib.sha256(submission.read_bytes()).hexdigest()!=value['input_file_sha256']):
                    raise ValueError('Calibration feedback submission hash differs')
        confirmed.append(value)
    if not confirmed:return None
    latest=max(confirmed,key=lambda v:(v['reviewed_date'],v.get('received_at',''),v.get('input_file_sha256',fingerprint(v))))
    return latest if latest['decision'] in ('needs_revision','exclude','uncertain') else None
