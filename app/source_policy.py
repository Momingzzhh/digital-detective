"""Conservative boundaries between quoted case evidence and judicial narration."""
import re

RULES = {
    'court_fact_or_outcome': re.compile(r'经审理查明|原审法院认定|原判认定|判决如下|判处'),
    'judicial_reasoning': re.compile(r'本院(?:认为|确认|认定|予以|不予|依法|经审理|查明|采信|采纳)|本院对[^。！？\n]{0,120}(?:采信|采纳|确认|认定|不予)'),
    'defence_assessment': re.compile(r'关于(?:被告人|当事人|上诉人)[^。！？\n]{0,180}(?:辩解|辩护|诉辩|上诉|意见)[^。！？\n]{0,100}(?:经查|本院|认为)'),
    'court_evidence_confirmation': re.compile(r'(?:以上|上述|前述)证据[，、\s]*(?:均|已经|均已)?(?:经过|经)?(?:庭审|举证|质证|审查|本院|相互印证)'),
    'generic_court_inventory': re.compile(r'(?:此外|另外)[，,]?在案(?:还有|有)[^。！？\n]{0,150}(?:刑事判决书|刑事裁定书|相关事实)'),
    'generic_conclusive_confession': re.compile(r'对上述认定事实所作的一致供述|供述的内容与其他证据相互印证'),
}
CURRENT_CASE_FILE_LABEL=re.compile(r'涉嫌[^。！？\n]{1,40}?案[.．](?:rar|zip|7z|pdf|docx?)(?![A-Za-z0-9])',re.I)


def court_boundary(text):
    matches = [(m.start(), name) for name,pattern in RULES.items() if (m := pattern.search(text))]
    if not matches: return None
    offset,rule = min(matches)
    return {'offset':offset,'rule':rule}


def require_clean_evidence(evidence):
    if any(court_boundary(e['text']) for e in evidence):
        raise ValueError('Visible evidence contains judicial reasoning or court confirmation')
    if any(CURRENT_CASE_FILE_LABEL.search(e['text']) for e in evidence):
        raise ValueError('Visible evidence contains a current-offence label in a source file name')
