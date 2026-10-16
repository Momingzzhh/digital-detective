import copy
import asyncio
import pytest
from test_v3_source import module


def example():
    # Synthetic mechanism fixture only; never included in real-case counts.
    a='I saw the person take the computer from the warehouse and carry it to the car. '*12
    b='I checked the computer serial number against the warehouse register and they matched. '*12
    xml=('<div1><p><persName type="defendantName">JOHN DOE</persName>'
         '<rs type="offenceDescription">was indicted for theft</rs></p>'
         '<p><persName type="witnessName">WITNESS ONE</persName>. '+a+'</p>'
         '<p>Cross-examined. I was standing near the door and had a clear view.</p>'
         '<p><persName type="witnessName">WITNESS TWO</persName>. '+b+'</p>'
         '<p><rs type="verdictDescription">GUILTY</rs></p>'
         '<p><rs type="punishmentDescription">Imprisonment</rs></p></div1>')
    return {'part':2,'n_defendants':1,'trial_id':'fixture','session_id':'18400101',
            'date':'18400101','xml':xml,'text':'fixture text',
            'persons':[{'id':'D1','type':'defendant','name':'JOHN DOE','given':'JOHN','surname':'DOE'}],
            'offences':[{'id':'O1','category':'theft','subcategory':'simpleLarceny'}],
            'verdicts':[{'id':'V1','category':'guilty','subcategory':'no_subcategory'}],
            'charges':[{'defendant_ids':['D1'],'offence_ids':['O1'],'verdict_ids':['V1'],'unresolved_ids':[]}]}


def test_parser_retains_cross_examination_and_hides_charge_verdict():
    p=module('prepare_oldbailey');source=example();row,error=p.extract(source)
    assert error is None and row['status']=='source_transcript_candidate_unverified'
    assert len(row['evidence'])==3
    assert 'Cross-examined' in row['evidence'][1]['text_en']
    for e in row['evidence']:
        xml=source['xml'][e['xml_start']:e['xml_end']]
        assert p.plain(p.ET.fromstring(xml))==e['text_en']
        assert 'GUILTY' not in e['text_en'] and 'indicted' not in e['text_en']


@pytest.mark.parametrize('defect',['join','lesser','leak','charge','label','entity'])
def test_ambiguous_or_contaminated_records_are_rejected(defect):
    p=module('prepare_oldbailey');source=example()
    if defect=='join':source['charges'][0]['verdict_ids']=['unknown']
    elif defect=='lesser':source['verdicts'][0]['subcategory']='lesserOffence'
    elif defect=='leak':source['xml']=source['xml'].replace('I checked','He was convicted. I checked')
    elif defect=='charge':source['xml']=source['xml'].replace('I checked','I charged him with theft. I checked')
    elif defect=='label':source['xml']=source['xml'].replace('I checked','It was theft. I checked')
    else:source['xml']='<!DOCTYPE test [<!ENTITY x "test">]>'+source['xml']
    row,error=p.extract(source)
    assert row is None and error


def test_multiple_tagged_witnesses_in_one_paragraph_preserve_scope():
    p=module('prepare_oldbailey');source=example()
    source['xml']=source['xml'].replace('Cross-examined. I was',
        'Cross-examined. The answer was given. <persName type="witnessName">ANOTHER WITNESS</persName>. I was')
    row,error=p.extract(source)
    assert error is None
    assert row['evidence'][1]['kind']=='mixed_speech_transcript'
    assert row['evidence'][1]['speaker_en']=='multiple_or_unattributed'
    assert 'ANOTHER WITNESS' in row['evidence'][1]['text_en']


def test_sentence_coverage_and_translation_input_exclude_hidden_labels():
    p=module('prepare_oldbailey');b=module('build_oldbailey')
    row,_=p.extract(example());spans=b.sentence_spans(row['evidence'])
    for e in row['evidence']:
        subset=[s for s in spans if s['evidence_id']==e['id']]
        assert ''.join(s['text_en'] for s in subset)==e['text_en']
        assert all(e['text_en'][s['start']:s['end']]==s['text_en'] for s in subset)
    payload=b.translation_payload(row,spans)
    assert set(payload)=={'source_reference','paragraphs','sentences'}
    assert 'GUILTY' not in str(payload) and 'simpleLarceny' not in str(payload)
    assert 'raw_record' not in payload and 'hidden_gold' not in payload
    audit=b.source_audit_payload(row,spans,[])
    assert 'GUILTY' in str(audit['hidden_original_context'])
    assert ''.join(s['text_en'] for s in audit['visible_sentences'])==''.join(e['text_en'] for e in row['evidence'])
    assert 'GUILTY' not in str(audit['visible_sentences'])


def test_translation_rejects_invented_sentences():
    b=module('build_oldbailey')
    row={'evidence':[{'id':'E1','text_en':'A witness saw the car.','kind':'witness_statement','speaker_en':'WITNESS'}]}
    spans=b.sentence_spans(row['evidence'])
    translation=b.Translation(sentences=[b.Line(id='invented',text_zh='虚构线索')])
    with pytest.raises(ValueError):b.chinese_evidence(row,spans,translation)


def test_changed_segmentation_invalidates_cached_audit_and_translation():
    p=module('prepare_oldbailey');b=module('build_oldbailey')
    source,_=p.extract(example());spans=b.sentence_spans(source['evidence'])
    saved=copy.deepcopy(source)
    b.bind_spans(saved,source,spans)
    saved.update(annotation={'event_sentence_ids':['E1:S1']},english_source_audit={'ok':True},translation={'sentences':[]})
    b.bind_spans(saved,source,spans)
    assert 'annotation' in saved
    altered=copy.deepcopy(spans);altered[0]['end']-=1
    b.bind_spans(saved,source,altered)
    assert not any(k in saved for k in ('annotation','english_source_audit','translation'))


def test_historical_export_checks_both_languages_and_gold():
    p=module('prepare_oldbailey');b=module('build_oldbailey');export=module('export_oldbailey')
    source,_=p.extract(example());row=copy.deepcopy(source)
    spans=b.sentence_spans(source['evidence']);b.bind_spans(row,source,spans)
    tr=b.Translation(sentences=[b.Line(id=s['id'],text_zh=s['text_en']) for s in spans])
    translated,cn=b.chinese_evidence(source,spans,tr)
    ids=[s['id'] for s in spans[:2]];index={s['id']:s for s in spans}
    audit={k:True for k in b.SourceAudit.model_fields if k not in ('reason','identified_perpetrators','unresolved_actors')}
    audit.update(reason='',identified_perpetrators=['JOHN DOE'],unresolved_actors=[])
    qc={k:True for k in b.TranslationAudit.model_fields if k!='reason'};qc['reason']=''
    row.update(evidence=translated,translation=tr.model_dump(),annotation={'event_sentence_ids':ids,'perpetrators':['JOHN DOE'],'unresolved_actors':[]},
        english_gold_events=[index[x] for x in ids],english_source_audit=audit,translation_audit=qc)
    row['hidden_gold'].update(charges=['theft/simpleLarceny'],chronology=[{
        'summary':cn[x]['text'],'evidence_ids':[index[x]['evidence_id']],'sentence_id':x,
        'sentence_start':cn[x]['start'],'sentence_end':cn[x]['end'],'english_source_sentence':index[x]} for x in ids])
    export.verify(row,source)
    view=export.visible(row)
    assert not any(k in view for k in ('hidden_gold','source','translation','english_source_audit'))
    text=export.visible_text(row)
    assert 'GUILTY' not in text and 'simpleLarceny' not in text and 'JOHN DOE' in text
    assert all(e['text'] in text for e in row['evidence'])
    for defect in ('evidence','verdict','event','audit','actor'):
        bad=copy.deepcopy(row)
        if defect=='evidence':bad['evidence'][0]['text']+='虚构'
        elif defect=='verdict':bad['hidden_gold']['verdict']['category']='notGuilty'
        elif defect=='event':bad['hidden_gold']['chronology'][0]['summary']+='虚构'
        elif defect=='audit':bad['english_source_audit'].pop('alternatives_resolved')
        else:bad['annotation']['perpetrators'].append('UNLISTED COACTOR')
        with pytest.raises(ValueError):export.verify(bad,source)


def test_provider_rejection_stops_queued_cases(tmp_path,monkeypatch):
    p=module('prepare_oldbailey');b=module('build_oldbailey')
    from app.store import initialize
    from app.llm import ProviderAccessError
    monkeypatch.setenv('RUNTIME_DIR',str(tmp_path/'runtime'));initialize()
    monkeypatch.setattr(b,'OUT',tmp_path)
    source,_=p.extract(example());second=copy.deepcopy(source);second['case_id']='second-fixture'
    calls=[]
    async def unavailable(*args,**kwargs):
        calls.append(1);raise ProviderAccessError('provider denied','Arrearage',400)
    monkeypatch.setattr(b,'structured',unavailable)
    async def run():
        stop=asyncio.Event();sem=asyncio.Semaphore(1)
        return await asyncio.gather(b.one(source,sem,400,provider_stop=stop),b.one(second,sem,400,provider_stop=stop))
    results=asyncio.run(run())
    assert len(calls)==1
    assert all(r['status']=='provider_deferred' for r in results)
