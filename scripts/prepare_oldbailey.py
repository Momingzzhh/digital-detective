#!/usr/bin/env python3
"""Extract source-backed English witness turns, with verdicts kept hidden.

Historical source verdicts are not assertions of objective guilt or modern law.
This script performs structural screening only, never evidence certification.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET
import pyarrow.parquet as pq
from fetch_lecardv2 import digest as file_digest

ROOT=Path(__file__).resolve().parents[1]
LEAK=re.compile(r'\b(?:guilty|convicted|acquitted|sentenced|indicted|indictment)\b|\bcharg(?:ed|e|ing)\b.{0,45}\bwith\b|jury.{0,25}(?:found|verdict)',re.I)
CROSS_CASE=re.compile(r'\b(?:like|same) offence\b|\b(?:last|previous) case\b|\balready (?:been )?tried\b',re.I)
MULTI_ACTOR=re.compile(r'\b(?:the )?(?:prisoner|defendant)\s+and\s+(?:several|some|other|a man|a woman|[A-Z][a-z]+)\b|\b(?:the )?(?:prisoner|defendant).{0,25}\bwith (?:a man|a woman|two men|three men|several|others)\b')
PROCEDURE=re.compile(r'conducted the (?:prosecution|defence)|appeared for the|(?:the court|jury) (?:held|found|considered|directed)|summing.up',re.I)
DEFENCE=re.compile(r"^(?:prisoner(?:['’]s)?(?: defence| statement)|the prisoner in (?:his|her) defence|defence[. :])",re.I)


def sha(value):return hashlib.sha256(value.encode()).hexdigest()
def plain(element):return ' '.join(''.join(element.itertext()).split())


def extract(row):
    if row['part']!=2 or row['n_defendants']!=1:return None,'outside_initial_scope'
    if len(row['offences'])!=1 or len(row['verdicts'])!=1:return None,'ambiguous_offence_or_verdict'
    verdict=row['verdicts'][0]
    if verdict['category']!='guilty' or verdict['subcategory']!='no_subcategory':
        return None,'not_unqualified_guilty_verdict'
    if not row['charges'] or any(c['unresolved_ids'] for c in row['charges']):return None,'unresolved_join'
    defendants=[p for p in row['persons'] if p['type']=='defendant']
    if len(defendants)!=1 or not defendants[0]['name']:return None,'ambiguous_defendant'
    defendant=defendants[0]
    # Every source criminalCharge link must resolve to this exact defendant,
    # offence and verdict. No guessed ids or lesser-offence remapping.
    for c in row['charges']:
        if (set(c['defendant_ids'])!={defendant['id']} or
            set(c['offence_ids'])!={row['offences'][0]['id']} or
            set(c['verdict_ids'])!={verdict['id']}):return None,'charge_join_mismatch'
    xml=row['xml']
    if '<!DOCTYPE' in xml.upper() or '<!ENTITY' in xml.upper():return None,'xml_declaration_rejected'
    try:root=ET.fromstring(xml)
    except ET.ParseError:return None,'invalid_xml'
    paragraphs=[p for p in root if p.tag=='p']
    spans=list(re.finditer(r'<p(?:\s[^>]*)?>[\s\S]*?</p>',xml))
    if len(spans)!=len(paragraphs):return None,'paragraph_span_ambiguity'
    evidence=[];speaker=None;role=None;witnesses=set();omitted=[]
    for number,(p,span) in enumerate(zip(paragraphs,spans),1):
        if any(x.get('type') in ('verdictDescription','punishmentDescription') for x in p.iter()):break
        text=plain(p)
        if not text:continue
        if any(x.get('type')=='offenceDescription' for x in p.iter()):
            if CROSS_CASE.search(text):return None,'cross_case_charge_reference'
            speaker=None;role=None;omitted.append(number);continue
        named=p.findall('.//persName[@type="witnessName"]')
        tagged=[plain(x) for x in named if plain(x)]
        header=next((x for x in named if plain(x) and text.find(plain(x)) in range(0,6)),None)
        if header is not None:
            speaker=plain(header);role='witness_statement';witnesses.add(speaker)
        elif re.match(r'^COURT\s+to\b',text,re.I):
            switch=re.match(r'^COURT\s+to\s+(.+?)\.\s*Q\.',text,re.I)
            if not switch or switch.group(1).strip() not in witnesses:
                return None,'unresolved_speaker_switch'
            speaker=switch.group(1).strip();role='witness_statement'
        elif DEFENCE.search(text):speaker=defendant['id'];role='defence_statement'
        elif PROCEDURE.search(text):
            speaker=None;role=None;omitted.append(number);continue
        elif text.startswith('(') and re.search(r'\b(?:gave|called|proved|deposed|examined)\b',text,re.I):
            # Printed reporter narrative is not the previous witness's speech.
            speaker=None;role=None;omitted.append(number);continue
        if not speaker:
            omitted.append(number);continue
        if MULTI_ACTOR.search(text):return None,'possible_multiple_actors'
        if LEAK.search(text) or CROSS_CASE.search(text):return None,'outcome_or_cross_case_inside_testimony'
        evidence_role=role;evidence_speaker=speaker
        if len(tagged)>1 or (tagged and header is None):
            # Multiple turns can share one printed/XML paragraph. Preserve it
            # verbatim, but never attribute all of it to the previous speaker.
            evidence_role='mixed_speech_transcript';evidence_speaker='multiple_or_unattributed'
            witnesses.update(tagged);speaker=tagged[-1];role='witness_statement'
        untagged=re.match(r'^(?:MR|MRS|MISS|MS)\.\s+([A-Z][A-Z \x27’-]+)\.',text)
        if untagged and not tagged:
            matching=[n for n in witnesses if n.split()[-1].rstrip('.')==untagged.group(1).strip()]
            if len(matching)==1:speaker=matching[0];role='witness_statement';evidence_speaker=speaker;evidence_role=role
            else:evidence_speaker='multiple_or_unattributed';evidence_role='mixed_speech_transcript'
        if re.match(r'^Prisoner\.\s+(?!Q\.)',text,re.I):
            evidence_speaker=defendant['id'];evidence_role='defence_statement'
        raw_xml=xml[span.start():span.end()]
        if plain(ET.fromstring(raw_xml))!=text:raise ValueError('XML paragraph mismatch')
        evidence.append({'id':f'E{len(evidence)+1}','text_en':text,'kind':evidence_role,'speaker_en':evidence_speaker,
                 'tagged_speakers_en':tagged,
                 'paragraph_number':number,'xml_start':span.start(),'xml_end':span.end(),
                 'source_xml_sha256':sha(raw_xml),
                 'transform':'XML text nodes concatenated; Unicode whitespace normalized; no words invented'})
    size=sum(len(e['text_en']) for e in evidence)
    if len(witnesses)<2 or not 1500<=size<=16000:return None,'insufficient_or_oversized_transcript'
    name=' '.join(x for x in (defendant.get('given'),defendant.get('surname')) if x).strip()
    if not name:name=re.sub(r'\s*\(\d+\)\s*','',defendant['name']).strip()
    if not name:return None,'missing_name_reference'
    offence=row['offences'][0]
    for code in (offence['category'],offence['subcategory']):
        if code=='no_subcategory':continue
        label=re.sub(r'([a-z])([A-Z])',r'\1 \2',code).casefold()
        if any(re.search(r'\b'+re.escape(label)+r'\b',e['text_en'],re.I) for e in evidence):
            return None,'exact_historical_label_inside_evidence'
    group='oldbailey-person-'+sha(' '.join(name.casefold().split()))[:24]
    ratio=int(sha(group)[:16],16)/2**64
    return {'case_id':'oldbailey-'+row['trial_id'],'group_id':group,
             'split':'train' if ratio<.7 else ('dev' if ratio<.85 else 'test'),
             'stratum':'england_historical_1834_1913','status':'source_transcript_candidate_unverified',
             'jurisdiction_context':{'country':'England','source_period':'1834–1913','trial_date':row['date'],
                    'interpretation':'Predict the recorded historical outcome, never modern Chinese legal liability'},
             'participant_references':[{'name_en':name,'source_person_id':defendant['id'],
                    'source_reference_terms':['the prisoner','prisoner'],
                    'meaning':'Neutral identity reference in the transcript, not a finding of guilt'}],
             'evidence':evidence,'hidden_gold':{'verdict':verdict,'perpetrators':[name],
                  'offences':row['offences'],'charges_source_links':row['charges']},
             'source':{'trial_id':row['trial_id'],'session_id':row['session_id'],
                       'record_url':'https://www.oldbaileyonline.org/record/'+row['trial_id'],
                       'sha256_xml':sha(xml),'sha256_text':sha(row['text']),
                       'omitted_pre_evidence_paragraphs':omitted},
             'raw_record':row},None


def main():
    manifest=json.loads((ROOT/'data/oldbailey_download_manifest.json').read_text())
    counts=Counter();labels=Counter();parts={};rows=[];ids=set()
    for key,expected in manifest['files'].items():
        file=ROOT/'data/raw/oldbailey'/Path(key).name
        if file.stat().st_size!=expected['bytes'] or file_digest(file)!=expected['sha256']:
            raise ValueError('Pinned transcript file missing or changed')
        local=Counter()
        for batch_index,batch in enumerate(pq.ParquetFile(file).iter_batches(batch_size=128)):
            for index,source in enumerate(batch.to_pylist()):
                local['source_trials']+=1
                row,reason=extract(source)
                if not row:local[reason]+=1;continue
                if row['case_id'] in ids:raise ValueError('Duplicate trial id')
                ids.add(row['case_id']);local['structural_candidates']+=1
                row['source'].update(dataset=manifest['dataset'],revision=manifest['revision'],
                     parquet=file.name,row_index=batch_index*128+index,license=manifest['license'])
                rows.append(row);labels[row['hidden_gold']['offences'][0]['subcategory']]+=1
        counts.update(local);parts[file.name]=dict(local)
        print(json.dumps({'file':file.name,'counts':dict(local)},ensure_ascii=False),flush=True)
    rows.sort(key=lambda r:sha('oldbailey-curation-order-v1:'+r['case_id']))
    out=ROOT/'data/processed/oldbailey_candidates.jsonl';temp=out.with_suffix('.tmp')
    with temp.open('w') as f:
        for row in rows:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    temp.replace(out)
    report={'dataset':manifest['dataset'],'revision':manifest['revision'],'counts':dict(counts),
            'parser_sha256':file_digest(Path(__file__)),
            'per_file':parts,'candidate_groups':len({r['group_id'] for r in rows}),
            'offence_subcategory_counts':dict(labels),'qualified':0,'translated':0,
            'limitations':['Historical printed proceedings are edited reports, not complete modern police files.',
                           'Only later-period single-defendant, single-offence, unqualified guilty trials are initially screened.',
                           'All charge/verdict ids must resolve exactly; no guessed lesser-offence mapping.',
                           'Witness turns are actual source text; witness accuracy and evidential sufficiency are unverified.',
                           'Grouping by normalized defendant name conservatively merges homonyms; unmentioned accomplices and related trials may escape.',
                           'No Chinese translations or formal benchmark have yet been generated.',
                           'Historical labels and performance must be reported separately from modern Chinese cases.']}
    (ROOT/'data/oldbailey_source_audit.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'counts':dict(counts),'candidate_groups':report['candidate_groups'],'qualified':0},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
