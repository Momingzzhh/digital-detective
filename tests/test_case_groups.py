from test_v3_source import module


def test_related_cross_source_reference_quarantines_frozen_splits():
    a=module('audit_case_groups')
    rows=[{'case_id':'first','group_id':'old-a','split':'train',
           'raw_record':{'qw':'甲法院（2020）甲0101刑初42号。证据甲。','ajjbqk':'短事实甲'}},
          {'case_id':'appeal','group_id':'old-b','split':'test',
           'raw_record':{'qw':'乙法院（2021）甲01刑终2号。原审（2020）甲0101刑初42号。','ajjbqk':'短事实乙'}},
          {'case_id':'independent','group_id':'old-c','split':'test',
           'raw_record':{'qw':'丙法院（2021）丙0101刑初42号。另一事件。','ajjbqk':'短事实丙'}}]
    mapping,stats=a.link(rows)
    assert mapping['first']['linked_group_id']==mapping['appeal']['linked_group_id']
    assert mapping['first']['cross_split_conflict']
    assert mapping['appeal']['cross_split_conflict']
    assert not mapping['independent']['cross_split_conflict']
    assert stats['quarantined_documents']==2
    assert [r['split'] for r in rows]==['train','test','test']


def test_identical_facts_link_despite_different_primary_numbers():
    a=module('audit_case_groups')
    fact='甲拿走仓库电脑，随后带到住所出售给乙。'*20
    rows=[{'case_id':str(i),'group_id':str(i),'split':'train',
           'raw_record':{'qw':f'（2020）甲0101刑初{i}号。'+fact,'ajjbqk':fact}}
          for i in (1,2)]
    mapping,stats=a.link(rows)
    assert stats['linked_groups']==1
    assert mapping['1']['linked_group_id']==mapping['2']['linked_group_id']
