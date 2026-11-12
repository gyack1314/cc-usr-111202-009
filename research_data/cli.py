"""查看原始研究资料的已有摘要，或运行共同变点分段分析。"""
import argparse
import json
from .io import load
from .domain import summarize
from . import segmentation as seg


def _run_segment(file_value):
    full = seg.solve(file_value)
    loo = seg.leave_one_center_out(file_value)
    stability = seg.stability_analysis(full, loo)
    return {
        'full_analysis': full,
        'leave_one_center_out': {
            c: {k: r[k] for k in
                ('centers', 'excluded_centers', 'budget_exhausted',
                 'segments_considered', 'path', 'solutions')}
            for c, r in loo.items()},
        'stability': stability,
        'method_notes': _method_notes(full),
    }


def _method_notes(full):
    return [
        '段费用=各中心独立加权最小二乘(截距+斜率, 原点=段首日)的加权残差平方和之和；'
        '缺测不补零，任一中心不满足 min_points 或设计矩阵秩不足则整段不可用。',
        '动态规划在共同边界上求全局最小：目标=总段费用+lambda*变点数；'
        '同分先比目标值(tie_tolerance 容差)，再比少变点，再比边界日期字典序。',
        '段费用缓存的合法条件：费用只依赖区间内原始观测、固定已知方差与段首日，'
        '与分段方案和 lambda 无关，跨 lambda 与 DP 路径复用是精确复用而非近似。',
        '未使用任何基于费用单调性的剪枝（如 PELT 类只对常均值/凸费用成立的不等式）；'
        '本模型每中心带截距与斜率且观测稀疏，这类剪枝不等式不适用，仅做合法性过滤。',
        '预算 max_segments_considered 按新评估的候选 (i,j) 段计数，缓存命中不重复计数；'
        '耗尽即标记 budget_exhausted，对应解 status 非 global_optimum，不声明全局完成。',
        '换日历原点不改变 t=day-origin 与费用；同日重复观测合并只改变一个与分段无关的'
        '组内常数项 within_ss，合并前后的最优分段与目标排序一致。',
        '逐中心剔除保持同一日历与参数完整重算；若剔除某中心后所有变点消失，'
        '则该中心贡献全部信号，变点是关联证据而非试剂导致的因果证据。',
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('command', choices=['inspect', 'segment'])
    p.add_argument('file')
    p.add_argument('--out', help='可选：把完整分段结果写入该 JSON 文件')
    a = p.parse_args()
    value, digest = load(a.file)
    if a.command == 'inspect':
        print(json.dumps({'source_sha256': digest, 'summary': summarize(value)},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return
    result = _run_segment(value)
    result = {'source_sha256': digest, **result}
    text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
    if a.out:
        with open(a.out, 'w', encoding='utf8') as fh:
            fh.write(text)
    print(text)


if __name__ == '__main__':
    main()
