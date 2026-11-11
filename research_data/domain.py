"""按中心计算已有控制样本的加权常均值。"""
from collections import defaultdict
from .io import unique,finite,positive,integer,identity

def summarize(v):
    calendar=v['calendar']
    if not calendar or any(type(d) is not int for d in calendar) or calendar!=sorted(set(calendar)):raise ValueError('日历必须严格递增')
    unique(v['measurements'],'id');groups=defaultdict(lambda:[0.0,0.0,0,set()])
    for r in v['measurements']:
        center=identity(r['center']);day=integer(r['day'])
        if day not in calendar:raise ValueError('观测不在公共日历')
        y=finite(r['reading']);w=1/positive(r['variance']);g=groups[center]
        g[0]+=w*y;g[1]+=w;g[2]+=1;g[3].add(day)
    return [{'center':k,'weighted_mean':g[0]/g[1],'weight':g[1],'records':g[2],'days':len(g[3])} for k,g in sorted(groups.items())]
