import copy
import pytest
from test_v3_source import module


def example():
    e=module('export_review_bundle')
    quote='被告人乙供述：我把仓库电脑带到住所，随后卖给丙，收款后存入自己的银行账户。'
    fact='经审理查明：'+quote
    full='首页。'+fact+'本院认为：乙犯盗窃罪。判决如下：有罪。'
    text=quote.replace('被告人','当事人')
    row={'case_id':'fixture','group_id':'group','split':'test',
         'source':{'sha256_fulltext':e.sha(full)},'source_fact_sha256':e.sha(fact),
         'source_judgment_excerpt':'禁止进入可见输入的结局',
         'evidence':[{'id':'E1','kind':'被告人供述','text':text,'quote':quote,
                      'start':len('经审理查明：'),'end':len(fact),
                      'fulltext_start':len('首页。经审理查明：'),
                      'fulltext_end':len('首页。')+len(fact)}],
         'hidden_gold':{'charges':['盗窃罪'],'perpetrators':['乙'],
                        'chronology':[{'summary':text,'evidence_ids':['E1'],
                                       'sentence_start':0,'sentence_end':len(text)}]}}
    return e,row,{'raw_record':{'qw':full,'ajjbqk':fact}}


def test_visible_export_excludes_judgment_gold_and_source_metadata():
    e,row,source=example()
    e.verify(row,source)
    public=e.visible(row)
    assert set(public)=={'case_id','group_id','split','evidence'}
    assert set(public['evidence'][0])=={'id','text','kind'}
    assert public['evidence'][0]['kind']=='当事人供述'
    assert '判决如下' not in str(public) and '盗窃罪' not in str(public)


@pytest.mark.parametrize('corruption',['offset','text','gold','source'])
def test_export_rejects_corrupted_source_or_support(corruption):
    e,row,source=example()
    row=copy.deepcopy(row);source=copy.deepcopy(source)
    if corruption=='offset':row['evidence'][0]['start']+=1
    elif corruption=='text':row['evidence'][0]['text']+='未经支持的指纹信息'
    elif corruption=='gold':row['hidden_gold']['chronology'][0]['summary']='乙留下指纹'
    else:source['raw_record']['qw']+='源文件变化'
    with pytest.raises(ValueError):e.verify(row,source)
