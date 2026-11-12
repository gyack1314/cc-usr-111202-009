"""共同变点分段引擎。

模型
----
日历是严格递增整数日 ``calendar[0..n-1]``。分段覆盖全部日历格、左闭右开，
边界只能落在日历格之间且所有中心共享。对段 ``[i, j)``（覆盖
``calendar[i..j-1]``），每个中心分别拟合加权最小二乘直线

    reading = alpha + beta * (day - calendar[i])，权重 w = 1 / variance

时间自变量始终是相对段首日的日历日差，从不使用全局大日值，因此平移日历
原点不会引入大数抵消，分段结论（含同分判定）严格稳定。

段费用是所有中心的加权残差平方和之和。段合法当且仅当：

* 跨 ``min_days`` 个以上日历格；
* 每个中心在段内有至少 ``min_points`` 个不同观测日期；
* 每个中心的加权设计矩阵满秩（2 列满列秩）；
* 缺测不补零：某中心在段内不满足条件则整段不可用。

对每个惩罚 ``lambda`` 求全局最小

    总段费用 + lambda * 变点数

动态规划保证全局最优；同分先按目标值（``tie_tolerance`` 容差内视为相等），
再按较少变点数，再按边界日期序列字典序。

本模块只使用标准库，浮点计算全部为 float64。
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .io import finite, identity, integer, positive, unique

# 满秩判定的相对阈值：det 相对 S_w*S_tt 小于该值视为秩不足。
_RANK_EPS = 1e-12


class BudgetExhausted(RuntimeError):
    """候选段评估数超过 max_segments_considered，不能声明全局完成。"""


@dataclass(frozen=True)
class CenterSegment:
    """单中心在单段内的加权最小二乘结果。"""

    center: str
    intercept: float          # alpha，时间原点为段首日
    slope: float              # beta，每日历日变化
    origin_day: int
    records: int              # 段内原始观测行数（含同日重复）
    distinct_days: int        # 段内不同观测日期数
    weight: float             # sum w
    residual_cost: float      # 该中心加权残差平方和
    support_record_ids: tuple[str, ...]


@dataclass(frozen=True)
class SegmentCost:
    """一个日历区间 [i, j) 的费用与合法性。"""

    start_index: int
    end_index: int
    feasible: bool
    cost: float = math.inf
    reasons: tuple[str, ...] = ()
    centers: tuple[CenterSegment, ...] = ()

    @property
    def length_slots(self) -> int:
        return self.end_index - self.start_index


@dataclass
class _SlotAgg:
    """单中心单个日历格上、同日全部重复观测合并后的加权充分量。"""

    w: float = 0.0
    wy: float = 0.0
    wy2: float = 0.0
    n: int = 0
    ids: list = None


@dataclass
class _CenterData:
    """单中心的按槽聚合数据。"""

    slots: list[_SlotAgg]
    # 不同观测日期数、原始行数的前缀和，用于 O(1) 的 min_points 判定。
    days_prefix: list[int]
    rec_prefix: list[int]


@dataclass
class _DPState:
    feasible: bool = False
    objective: float = math.inf
    n_changepoints: int = 0
    boundaries: tuple[int, ...] = ()
    predecessor: Optional[int] = None
    last_cost: float = math.inf


def validate_material(v: dict) -> None:
    """完整校验输入资料，不合法时抛 ValueError / KeyError / TypeError。"""
    cal = v['calendar']
    if not isinstance(cal, list) or not cal:
        raise ValueError('日历必须是非空列表')
    if any(type(d) is not int for d in cal):
        raise ValueError('日历必须全部为整数')
    if cal != sorted(set(cal)):
        raise ValueError('日历必须严格递增')
    integer(v['min_days'], minimum=1)
    integer(v['min_points'], minimum=1)
    grid = v['lambda_grid']
    if not isinstance(grid, list) or not grid:
        raise ValueError('lambda_grid 必须是非空列表')
    for lam in grid:
        if (isinstance(lam, bool) or not isinstance(lam, (int, float))
                or not math.isfinite(lam) or lam < 0):
            raise ValueError('lambda 必须是非负有限数')
    solver = v.get('solver', {})
    if not isinstance(solver, dict):
        raise ValueError('solver 必须是对象')
    tol = solver.get('objective_tie_tolerance', 1e-9)
    if (isinstance(tol, bool) or not isinstance(tol, (int, float))
            or not math.isfinite(tol) or tol <= 0):
        raise ValueError('objective_tie_tolerance 必须是正数')
    integer(solver.get('max_segments_considered', 1_000_000), minimum=1)
    rows = v['measurements']
    if not isinstance(rows, list):
        raise ValueError('measurements 必须是列表')
    unique(rows, 'id')
    day_set = set(cal)
    for r in rows:
        identity(r['center'])
        day = integer(r['day'])
        if day not in day_set:
            raise ValueError('观测不在公共日历')
        finite(r['reading'])
        positive(r['variance'])


def build_center_data(v: dict,
                      excluded: Optional[set[str]] = None
                      ) -> tuple[list[int], list[str], dict[str, _CenterData]]:
    """构造每中心的按槽聚合。缺测槽保持零聚合，不补任何数据。"""
    cal = v['calendar']
    n = len(cal)
    slot_of = {d: k for k, d in enumerate(cal)}
    centers = sorted({identity(r['center']) for r in v['measurements']
                      if r['center'] not in (excluded or set())})
    data: dict[str, _CenterData] = {}
    for c in centers:
        slots = [_SlotAgg(ids=[]) for _ in range(n)]
        for r in v['measurements']:
            if r['center'] != c:
                continue
            k = slot_of[integer(r['day'])]
            y = finite(r['reading'])
            w = 1.0 / positive(r['variance'])
            a = slots[k]
            a.w += w
            a.wy += w * y
            a.wy2 += w * y * y
            a.n += 1
            a.ids.append(r['id'])
        days_pref = [0] * (n + 1)
        rec_pref = [0] * (n + 1)
        for k, a in enumerate(slots):
            days_pref[k + 1] = days_pref[k] + (1 if a.n else 0)
            rec_pref[k + 1] = rec_pref[k] + a.n
        data[c] = _CenterData(slots, days_pref, rec_pref)
    return cal, centers, data


class SegmentTable:
    """全部日历区间的费用缓存与合法性表。

    缓存合法条件：段费用只依赖区间内的原始行、固定已知方差和段首日原点，
    与分段方案及 lambda 无关，因此同一 (i, j) 在所有 lambda、所有 DP 路径
    下复用是精确的（不是近似剪枝）。
    """

    def __init__(self, calendar: list[int], centers: list[str],
                 data: dict[str, _CenterData], min_days: int, min_points: int,
                 budget: int):
        self.calendar = calendar
        self.centers = centers
        self.data = data
        self.min_days = min_days
        self.min_points = min_points
        self.budget = budget
        self.considered = 0
        self.exhausted = False
        self._cache: dict[tuple[int, int], SegmentCost] = {}

    def segment(self, i: int, j: int) -> SegmentCost:
        key = (i, j)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        self.considered += 1
        if self.considered > self.budget:
            self.exhausted = True
            self._cache[key] = SegmentCost(i, j, feasible=False,
                                           reasons=('budget_exhausted',))
            raise BudgetExhausted('候选段评估数超过预算，全局最优无法保证')
        n = len(self.calendar)
        if j - i < self.min_days or i < 0 or j > n:
            sc = SegmentCost(i, j, feasible=False,
                             reasons=('span_below_min_days',))
            self._cache[key] = sc
            return sc
        return self._fit_range(i, j)

    def _fit_range(self, i: int, j: int) -> SegmentCost:
        """对段 [i,j) 的每个中心做相对时间加权最小二乘并汇总费用。"""
        origin_day = self.calendar[i]
        fitted: list[CenterSegment] = []
        reasons: list[str] = []
        for c in self.centers:
            cd = self.data[c]
            distinct_days = cd.days_prefix[j] - cd.days_prefix[i]
            if distinct_days < self.min_points:
                reasons.append(f'min_points:{c}')
                continue
            sw = st = stt = sty = swy = swy2 = 0.0
            rec0 = cd.rec_prefix[i]
            ids: list[str] = []
            for k in range(i, j):
                a = cd.slots[k]
                if a.n == 0:
                    continue
                t = self.calendar[k] - origin_day
                sw += a.w
                st += a.w * t
                stt += a.w * t * t
                sty += t * a.wy
                swy += a.wy
                swy2 += a.wy2
                ids.extend(a.ids)
            det = sw * stt - st * st
            if not (det > _RANK_EPS * sw * stt):
                reasons.append(f'rank_deficient:{c}')
                continue
            beta = (sw * sty - st * swy) / det
            alpha = (swy - beta * st) / sw
            # RSS = wy2 - (swy^2*stt + sty^2*sw - 2*swy*sty*st)/det
            cost_c = swy2 - (swy * swy * stt + sty * sty * sw
                             - 2.0 * swy * sty * st) / det
            if -1e-12 * max(1.0, abs(swy2)) < cost_c < 0.0:
                cost_c = 0.0
            records = cd.rec_prefix[j] - rec0
            fitted.append(CenterSegment(
                center=c, intercept=alpha, slope=beta, origin_day=origin_day,
                records=records, distinct_days=distinct_days, weight=sw,
                residual_cost=cost_c, support_record_ids=tuple(ids)))
        key = (i, j)
        if reasons:
            sc = SegmentCost(i, j, feasible=False, reasons=tuple(reasons))
        else:
            total = sum(cs.residual_cost for cs in fitted)
            sc = SegmentCost(i, j, feasible=True, cost=total,
                             centers=tuple(fitted))
        self._cache[key] = sc
        return sc

    def all_segments(self) -> list[SegmentCost]:
        """按 (起点, 终点) 排序的全部已评估段。"""
        return [self._cache[k] for k in sorted(self._cache)]


def _precompute_all_segments(table: SegmentTable, n: int) -> None:
    """枚举所有满足跨度下限的区间；受预算约束，耗尽则标记并抛出。"""
    md = table.min_days
    for i in range(0, n - md + 1):
        for j in range(i + md, n + 1):
            table.segment(i, j)


def _better(cand_obj: float, cand_k: int, cand_b: tuple[int, ...],
            best: _DPState, tol: float) -> bool:
    """同分比较器。

    顺序：目标值（容差内相等）→ 较少变点数 → 边界字典序。
    合法性：目标对公共后缀是可加的，变点数与边界元组在追加同一后缀后
    相对次序不变，故前缀上的贪心选择对任何延伸都保持占优，DP 局部选择
    与全局同分规则一致。
    """
    if cand_obj < best.objective - tol:
        return True
    if cand_obj > best.objective + tol:
        return False
    if cand_k != best.n_changepoints:
        return cand_k < best.n_changepoints
    return cand_b < best.boundaries


def dynamic_program(table: SegmentTable, lam: float,
                    tol: float) -> tuple[list[_DPState], bool]:
    """单个 lambda 的 DP。返回逐槽状态表与末槽是否可行。"""
    cal = table.calendar
    n = len(cal)
    states = [_DPState() for _ in range(n + 1)]
    states[0] = _DPState(feasible=True, objective=0.0, n_changepoints=0,
                         boundaries=())
    for j in range(1, n + 1):
        best = _DPState()
        for i in range(0, j - table.min_days + 1):
            prev = states[i]
            if not prev.feasible:
                continue
            sc = table.segment(i, j)
            if not sc.feasible:
                continue
            penalty = lam if i > 0 else 0.0
            cand_obj = prev.objective + sc.cost + penalty
            cand_k = prev.n_changepoints + (1 if i > 0 else 0)
            cand_b = prev.boundaries + ((cal[i],) if i > 0 else ())
            if not best.feasible or _better(cand_obj, cand_k, cand_b,
                                            best, tol):
                best = _DPState(feasible=True, objective=cand_obj,
                                n_changepoints=cand_k, boundaries=cand_b,
                                predecessor=i, last_cost=sc.cost)
        states[j] = best
    return states, states[n].feasible


def _reconstruct(table: SegmentTable, states: list[_DPState]
                 ) -> list[SegmentCost]:
    n = len(table.calendar)
    cuts = [0]
    j = n
    while states[j].predecessor not in (None, 0):
        j = states[j].predecessor
        cuts.append(j)
    cuts = sorted(cuts) + [n]
    return [table.segment(cuts[k], cuts[k + 1])
            for k in range(len(cuts) - 1)]


def solve(v: dict, excluded_centers: Optional[set[str]] = None) -> dict:
    """对一份资料跑完整分段分析（全部 lambda、DP 表、路径）。

    ``excluded_centers`` 用于逐中心剔除：日历与参数保持原样，仅删除行，
    其他中心的时间范围不因缺人而缩短。
    """
    validate_material(v)
    cal, centers, data = build_center_data(v, excluded_centers)
    n = len(cal)
    min_days = v['min_days']
    min_points = v['min_points']
    tol = v.get('solver', {}).get('objective_tie_tolerance', 1e-9)
    budget = v.get('solver', {}).get('max_segments_considered', 1_000_000)
    table = SegmentTable(cal, centers, data, min_days, min_points, budget)

    exhausted = False
    try:
        _precompute_all_segments(table, n)
    except BudgetExhausted:
        exhausted = True

    dp_tables: dict[str, list[dict]] = {}
    solutions: list[dict] = []
    for lam in v['lambda_grid']:
        if exhausted:
            dp_tables[str(lam)] = [{'end_index': j, 'feasible': False}
                                   for j in range(n + 1)]
            solutions.append(_incomplete_solution(lam, 'budget_exhausted'))
            continue
        states, feasible = dynamic_program(table, lam, tol)
        dp_tables[str(lam)] = [_dp_row(k, cal, st)
                               for k, st in enumerate(states)]
        if not feasible:
            solutions.append(_incomplete_solution(lam, 'infeasible'))
            continue
        segments = _reconstruct(table, states)
        solutions.append({
            'lambda': lam,
            'status': 'global_optimum',
            'global_optimal': True,
            'objective': states[n].objective,
            'segment_cost_total': (states[n].objective
                                   - lam * states[n].n_changepoints),
            'n_changepoints': states[n].n_changepoints,
            'n_segments': states[n].n_changepoints + 1,
            'boundaries': list(states[n].boundaries),
            'segments': [_segment_payload(cal, sc) for sc in segments],
        })

    path = [{
        'lambda': s['lambda'],
        'n_changepoints': s.get('n_changepoints'),
        'boundaries': s.get('boundaries'),
        'objective': s.get('objective'),
        'status': s['status'],
    } for s in solutions]

    return {
        'calendar': cal,
        'n_calendar_slots': n,
        'centers': centers,
        'excluded_centers': sorted(excluded_centers or set()),
        'min_days': min_days,
        'min_points': min_points,
        'tie_tolerance': tol,
        'budget': budget,
        'segments_considered': table.considered,
        'budget_exhausted': exhausted,
        'segment_cost_table': [_segment_payload(cal, sc)
                               for sc in table.all_segments()],
        'dp_tables': dp_tables,
        'solutions': solutions,
        'path': path,
    }


def _incomplete_solution(lam, status: str) -> dict:
    return {'lambda': lam, 'status': status, 'global_optimal': False,
            'objective': None, 'segment_cost_total': None,
            'n_changepoints': None, 'n_segments': None,
            'boundaries': None, 'segments': None}


def _dp_row(k: int, cal: list[int], st: _DPState) -> dict:
    return {
        'end_index': k,
        'end_day': cal[k - 1] if k >= 1 else None,
        'feasible': st.feasible,
        'objective': st.objective if st.feasible else None,
        'n_changepoints': st.n_changepoints if st.feasible else None,
        'boundaries': list(st.boundaries) if st.feasible else None,
        'predecessor_index': st.predecessor,
        'last_segment_cost': (st.last_cost if st.feasible and k >= 1 else None),
    }


def _segment_payload(cal: list[int], sc: SegmentCost) -> dict:
    payload = {
        'start_index': sc.start_index,
        'end_index': sc.end_index,
        'start_day': cal[sc.start_index],
        'end_day_exclusive': cal[sc.end_index - 1],
        'length_slots': sc.length_slots,
        'feasible': sc.feasible,
        'infeasible_reasons': list(sc.reasons),
        'cost': sc.cost if sc.feasible else None,
        'per_center': {},
        'support_record_ids': [],
    }
    for cs in sc.centers:
        payload['per_center'][cs.center] = {
            'intercept': cs.intercept,
            'slope': cs.slope,
            'origin_day': cs.origin_day,
            'records': cs.records,
            'distinct_days': cs.distinct_days,
            'weight': cs.weight,
            'residual_cost': cs.residual_cost,
            'support_record_ids': list(cs.support_record_ids),
        }
        payload['support_record_ids'].extend(cs.support_record_ids)
    payload['support_record_ids'].sort()
    return payload


def enumerate_partitions(v: dict, lam: float) -> dict:
    """小日历核对：枚举所有合法分段，独立计算目标并应用同一同分规则。

    与 DP 完全独立的实现：递归枚举所有切点子集，逐段直接调用费用表，
    用于测试中交叉验证全局最优值与同分选择。
    """
    validate_material(v)
    cal, centers, data = build_center_data(v)
    n = len(cal)
    min_days = v['min_days']
    min_points = v['min_points']
    tol = v.get('solver', {}).get('objective_tie_tolerance', 1e-9)
    budget = v.get('solver', {}).get('max_segments_considered', 1_000_000)
    table = SegmentTable(cal, centers, data, min_days, min_points, budget)

    best: Optional[dict] = None
    n_legal = 0
    results: list[dict] = []

    def recurse(start: int, cuts: list[int]) -> None:
        nonlocal best, n_legal
        # 尝试在 start 之后、给末段留足 min_days 的位置切分
        for cut in range(start + min_days, n - min_days + 1):
            cuts.append(cut)
            recurse(cut, cuts)
            cuts.pop()
        # 不继续切，[start, n) 作为末段
        if n - start >= min_days:
            edges = [0] + cuts + [n]
            total = 0.0
            for a, b in zip(edges[:-1], edges[1:]):
                sc = table.segment(a, b)
                if not sc.feasible:
                    return
                total += sc.cost
            n_legal += 1
            k = len(cuts)
            objective = total + lam * k
            boundaries = tuple(cal[c] for c in cuts)
            entry = {'objective': objective, 'n_changepoints': k,
                     'boundaries': list(boundaries), 'cuts': list(cuts),
                     'segment_cost_total': total}
            results.append(entry)
            if (best is None or _better(
                    objective, k, boundaries,
                    _DPState(True, best['objective'], best['n_changepoints'],
                             tuple(best['boundaries'])), tol)):
                best = entry

    recurse(0, [])
    return {'n_legal_partitions': n_legal, 'best': best, 'all': results,
            'budget_exhausted': table.exhausted}


def merge_repeats(v: dict) -> tuple[dict, float, dict]:
    """把相同 center/day 的重复观测等价合并为单行。

    合并保留总权重 W=sum w：新 variance=1/W，新 reading=sum(wy)/W。
    加权最小二乘的设计矩阵只按不同日期取值，故合并后拟合值与斜率不变；
    段费用相差一个与分段无关的组内常数项

        within_ss = 每个 (center,day) 组内 sum_r w_r (y_r - ybar)^2

    即  原始段费用 = 合并段费用 + 该段覆盖组的 within_ss。
    返回 (新资料, within_ss 总和, 每组常数项映射)。
    """
    validate_material(v)
    groups: dict[tuple[str, int], dict[str, float]] = {}
    order: list[tuple[str, int]] = []
    for r in v['measurements']:
        key = (r['center'], r['day'])
        w = 1.0 / r['variance']
        g = groups.get(key)
        if g is None:
            g = {'W': 0.0, 'Wy': 0.0, 'Wy2': 0.0, 'n': 0}
            groups[key] = g
            order.append(key)
        g['W'] += w
        g['Wy'] += w * r['reading']
        g['Wy2'] += w * r['reading'] ** 2
        g['n'] += 1
    merged_rows = []
    constants = {}
    total_within = 0.0
    for idx, key in enumerate(order):
        c, d = key
        g = groups[key]
        within = g['Wy2'] - g['Wy'] ** 2 / g['W']
        if -1e-12 * max(1.0, abs(g['Wy2'])) < within < 0.0:
            within = 0.0
        constants[f'{c}|{d}'] = within
        total_within += within
        merged_rows.append({
            'id': f'merged-{idx}',
            'center': c,
            'day': d,
            'reading': g['Wy'] / g['W'],
            'variance': 1.0 / g['W'],
        })
    new_v = dict(v)
    new_v['measurements'] = merged_rows
    return new_v, total_within, constants


def shift_origin(v: dict, delta: int) -> dict:
    """日历整体平移 delta 日：换日期原点不应改变分段结论。"""
    if type(delta) is not int:
        raise ValueError('平移量必须是整数')
    new_v = dict(v)
    new_v['calendar'] = [d + delta for d in v['calendar']]
    new_v['measurements'] = [dict(r, day=r['day'] + delta)
                             for r in v['measurements']]
    return new_v


def _solution_at(result: dict, lam) -> Optional[dict]:
    for s in result['solutions']:
        if s['lambda'] == lam:
            return s
    return None


def stability_analysis(full: dict, loo: dict[str, dict]) -> dict:
    """汇总全 lambda 路径与逐中心剔除后的边界稳定性。

    注意：变点只表示控制样本读数的整体测量关系在统计上发生改变，
    与换试剂在时间上相关并不能断言因果。
    """
    by_lambda = {}
    for s in full['solutions']:
        lam = s['lambda']
        full_boundaries = s.get('boundaries') or []
        detail = []
        for d in full_boundaries:
            present, absent = [], []
            for c, res in loo.items():
                loo_sol = _solution_at(res, lam)
                bs = (loo_sol or {}).get('boundaries') or []
                (present if d in bs else absent).append(c)
            detail.append({'boundary_day': d,
                           'survives_when_removing': present,
                           'disappears_when_removing': absent})
        single_signal = []
        if full_boundaries:
            for c, res in loo.items():
                loo_sol = _solution_at(res, lam)
                if loo_sol is not None and not (loo_sol.get('boundaries')
                                                or []):
                    single_signal.append(c)
        by_lambda[str(lam)] = {
            'boundaries': full_boundaries,
            'boundary_detail': detail,
            'centers_whose_removal_removes_all_changepoints': single_signal,
            'one_center_contributes_all_signal': bool(single_signal),
        }
    occurrence: dict[int, list[int]] = {}
    for idx, s in enumerate(full['solutions']):
        for d in (s.get('boundaries') or []):
            occurrence.setdefault(d, []).append(idx)
    return {
        'by_lambda': by_lambda,
        'boundary_lambda_occurrence': [
            {'boundary_day': d,
             'lambda_indices': idxs,
             'lambdas': [full['solutions'][k]['lambda'] for k in idxs],
             'stable_across_all_lambda': len(idxs)
             == len(full['solutions'])}
            for d, idxs in sorted(occurrence.items())],
        'causal_note': '变点仅标识加权测量关系的统计改变；与换试剂时间重合是关联，不是因果证据。',
    }


def leave_one_center_out(v: dict) -> dict[str, dict]:
    """逐中心剔除后的完整重算；日历不变，其他中心不缩短时间范围。"""
    _, centers, _ = build_center_data(v)
    return {c: solve(v, excluded_centers={c}) for c in centers}
