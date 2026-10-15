"""Paid connectivity check using a fictional, non-benchmark example."""
import asyncio
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from app.agent import analyze
from app.store import initialize,budget_status

async def main():
    initialize()
    case={'id':'smoke','title':'虚构联通测试，不计入真实数据集','sources':[{'id':'S-1','name':'虚构测试',
        'text':'研究用虚构场景：仓库一台电脑丢失。管理员称昨晚20点电脑仍在，今天8点发现不见。门禁记录显示甲和乙均在该时段进入大楼。监控设备损坏，没有录像。暂未取得指纹鉴定。'}], 'evidence':[],'messages':[]}
    result=await analyze(case,print)
    print({'status':result['analysis']['status'],'evidence_count':len(result['evidence']),'questions':result['analysis']['questions'],'budget':budget_status()})

if __name__=='__main__':asyncio.run(main())
