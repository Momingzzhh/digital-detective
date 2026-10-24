import importlib.util
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]


def module(name):
    sys.path.insert(0,str(ROOT/'scripts'))
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def test_v3_preserves_source_offsets_and_rejects_court_section():
    b=module('build_v3')
    fact=('经审理查明。上述事实有下列证据证实： '
          '1、证人甲的证言：我看见乙拿走了登记在册的电脑，随后把它放进车里并驾车离开仓库。 '
          '2、鉴定意见：车内提取的电脑序列号与仓库登记表上记录的序列号一致，检测过程有照片。 '
          '3、被告人乙的供述：我把电脑从仓库搬到车上，后来驾车离开并把它存放在住所。 '
          '二、被告人乙盗窃犯罪事实：法院确认乙构成盗窃罪。')
    items=b.extract_items(fact,'盗窃罪')
    assert len(items)==2
    assert all(fact[e['start']:e['end']]==e['quote'] for e in items)
    assert all('法院确认' not in e['text'] for e in items)


def test_source_kind_distinguishes_accused_statement_victim_and_lab_report():
    b=module('build_v3')
    assert b.evidence_kind('2.被告人甲陈述，我申请后收到款项，用于归还借款。')=='当事人供述'
    assert b.evidence_kind('3.被害人乙的陈述，自己的物品被拿走，之后向警方反映。')=='被害人陈述'
    assert b.evidence_kind('10、物证检验报告书证实，两份样本的化验结果为阴性。')=='检验报告'
    assert b.evidence_kind('5、现场勘验检查笔录，记录了现场实际观察及提取物。')=='勘验笔录'


def test_file_label_item_is_excluded_without_stopping_other_evidence():
    b=module('build_v3')
    fact=('下列证据证实：1、电子数据，提取结果存放在甲涉嫌测试案.rar文件中，文件标题载明了目前的案件分类与人员信息。 '
          '2、证人乙的证言：我看到有人将登记在册的物品搬进车里，随后驾车离开，天黑后还曾返回现场。 '
          '3、鉴定意见：从现场提取的痕迹与车内物品上的痕迹相符，检验过程保留照片及相关编号记录。')
    items=b.extract_items(fact,'测试罪')
    assert len(items)==2 and all('涉嫌测试案.rar' not in e['text'] for e in items)


def test_same_primary_case_number_groups_across_title_formats():
    p=module('prepare_lecardv2')
    a='某某一审判决书甲省乙市人民法院（2020）甲0101刑初42号：案情'
    b='甲省乙市人民法院刑事判决书（2020）甲0101刑初42号。'
    assert p.case_key(a)==p.case_key(b)
    full='首页。经审理查明：甲拿走物品。证人乙陈述。 本院认为：符合要件。判决如下：被告人甲犯盗窃罪。'
    fact,judgment,offset=p.split_document(full)
    assert full[offset:offset+len(fact)]==fact
    assert judgment.startswith('判决如下')


def test_verdict_names_keep_digits_and_do_not_match_confiscation_clause():
    b=module('build_v3')
    judgment=('被告人刘某1犯盗窃罪，判处有期徒刑一年。'
              '被告李某犯盗窃罪，判处罚金。扣押被告人刘某1供犯罪使用的手机。')
    assert b.VERDICT_ACTOR.findall(judgment)==['刘某1','李某']


def test_testimony_without_word_zhengyan_and_court_tail_are_handled():
    b=module('build_v3')
    fact=('下列证据证实：1、证人王某1证实，乙将铲车开向围墙，把院墙和西侧大门推倒。'
          '后来我们在倒塌大门下发现甲，甲的头部流血，已经不能回答问题。'
          '2、被告人乙供述，我开着铲车推倒了门，我想把房子推倒吓唬他，并没有看到院内有人。'
          '对于控辩双方争议焦点，本院评判如下：此案不属于故意杀人。')
    items=b.extract_items(fact,'过失致人死亡罪')
    assert len(items)==2
    assert all('本院评判' not in e['quote'] for e in items)
    for span in b.sentences(items):
        item=next(e for e in items if e['id']==span['evidence_id'])
        assert item['text'][span['start']:span['end']]==span['text']


def test_continued_witness_and_long_final_confession_are_not_lost():
    b=module('build_v3')
    long='我把电脑带到住所，随后联系买家出售，收到的钱存入自己的银行账户。'*70
    fact=('下列证据证实：1、证人甲证言：我在仓库门口看到乙把电脑放进车里，随后驾车离开，我当时记下了车牌号码并报了警。'
          '2、丙的证言证明：乙在当天晚上把这台电脑送到我家，并拿出了仓库登记表要求我购买。'
          '3、被告人乙供述：'+long+'上述证据经庭审确认，本院认为其构成盗窃罪。')
    items=b.extract_items(fact,'盗窃罪')
    assert len(items)==3
    assert items[1]['kind']=='证人证言'
    assert long in items[2]['quote']
    assert '本院认为' not in items[2]['text']
    assert all(fact[e['start']:e['end']]==e['quote'] for e in items)


def test_reference_repair_resolves_only_exact_known_sentence_ids():
    import pytest
    b=module('build_v3')
    ann=b.Annotation(event_sentence_ids=['E1:S1','E2:S1'],
                     culprit_evidence_ids=['E1:S1','E1'],offence_evidence_ids=['E2:S1'])
    spans=[{'id':'E1:S1','evidence_id':'E1'},{'id':'E2:S1','evidence_id':'E2'}]
    fixed,repairs=b.normalize_annotation_references(ann,spans,{'E1','E2'})
    assert fixed.culprit_evidence_ids==['E1'] and fixed.offence_evidence_ids==['E2']
    assert len(repairs)==2 and ann.culprit_evidence_ids==['E1:S1','E1']
    ann.culprit_evidence_ids=['E9:S1']
    with pytest.raises(ValueError):b.normalize_annotation_references(ann,spans,{'E1','E2'})


def test_judicial_assessments_cannot_hide_in_last_confession():
    from app.source_policy import require_clean_evidence
    import pytest
    b=module('build_v3')
    quote='被告人乙供述：我与丙商量好后，将电脑从仓库搬到车上，后来交给了丙，丙支付了事先商定的价款。'
    for tail in ('此外，在案还有刑事判决书证明。关于被告人乙的辩解，经查本院不予采纳。',
                 '关于被告人乙提出的辩解，经查不能成立。',
                 '本院对相关辩护意见不予采纳。',
                 '上述证据经过庭审举证、质证。'):
        fact='上述事实有下列证据证实：1、'+quote+tail
        items=b.extract_items(fact,'盗窃罪')
        assert len(items)==1 and quote.replace('被告人','当事人') in items[0]['text']
        assert tail not in items[0]['quote']
        require_clean_evidence(items)
        assert fact[items[0]['start']:items[0]['end']]==items[0]['quote']
        with pytest.raises(ValueError):require_clean_evidence([{'text':tail}])
