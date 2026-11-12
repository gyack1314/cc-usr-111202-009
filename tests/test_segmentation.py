"""共同变点分段引擎的单元测试。

覆盖：DP 全局最优与独立全枚举一致、三级同分规则、段费用独立核对、
换原点/同日合并不变性、逐中心剔除与单一中心信号、预算耗尽不声明全局、
完全不可行、资料校验，以及随机化差分测试。
"""
import copy
import random
import unittest

from research_data.io import load
from research_data import segmentation as seg


def _material(name):
    return load(f'data/{name}.json')[0]


def _independent_cost(v, i, j):
    """与引擎完全独立的逐中心 2x2 正规方程 WLS 费用。"""
    cal = v['calendar']
    if j - i < v['min_days']:
        return None
    total = 0.0
    for c in sorted({r['center'] for r in v['measurements']}):
        rows = [r for r in v['measurements']
                if r['center'] == c and i <= cal.index(r['day']) < j]
        if len({r['day'] for r in rows}) < v['min_points']:
            return None
        origin = cal[i]
        s00 = s01 = s11 = t0 = t1 = y2 = 0.0
        for r in rows:
            w = 1.0 / r['variance']
            tt = r['day'] - origin
            y = r['reading']
            s00 += w
            s01 += w * tt
            s11 += w * tt * tt
            t0 += w * y
            t1 += w * tt * y
            y2 += w * y * y
        det = s00 * s11 - s01 * s01
        if det <= 1e-12 * s00 * s11:
            return None
        total += y2 - (t0 * t0 * s11 + t1 * t1 * s00
                       - 2.0 * t0 * t1 * s01) / det
    return max(total, 0.0)


class SegmentEngineTests(unittest.TestCase):
    def test_dp_matches_exhaustive_enumeration_on_all_materials(self):
        for name in ('normal', 'same-day-repeat', 'sparse-center'):
            v = _material(name)
            full = seg.solve(v)
            for lam in v['lambda_grid']:
                with self.subTest(material=name, lam=lam):
                    en = seg.enumerate_partitions(v, lam)
                    dp = next(s for s in full['solutions']
                              if s['lambda'] == lam)
                    best = en['best']
                    if best is None:
                        self.assertEqual(dp['status'], 'infeasible')
                        continue
                    self.assertAlmostEqual(best['objective'],
                                           dp['objective'], places=9)
                    self.assertEqual(best['boundaries'], dp['boundaries'])
                    self.assertEqual(best['n_changepoints'],
                                     dp['n_changepoints'])

    def test_primary_material_finds_day6_changepoint(self):
        full = seg.solve(_material('normal'))
        by_lam = {s['lambda']: s for s in full['solutions']}
        for lam in (0, 1, 5):
            self.assertEqual(by_lam[lam]['boundaries'], [6])
            self.assertEqual(by_lam[lam]['status'], 'global_optimum')
        for lam in (20, 100):
            self.assertEqual(by_lam[lam]['boundaries'], [])
        # 费用可加性：段费用之和等于总段费用
        for s in full['solutions']:
            self.assertAlmostEqual(
                sum(x['cost'] for x in s['segments']),
                s['segment_cost_total'], places=9)

    def test_dp_table_reports_predecessor_chain(self):
        full = seg.solve(_material('normal'))
        table = full['dp_tables']['5']
        self.assertTrue(table[0]['feasible'])
        self.assertIsNone(table[0]['predecessor_index'])
        final = table[-1]
        self.assertTrue(final['feasible'])
        # 沿前驱链重建的边界与解一致
        chain = []
        j = len(table) - 1
        while table[j]['predecessor_index'] not in (None, 0):
            j = table[j]['predecessor_index']
            chain.append(full['calendar'][j])
        self.assertEqual(sorted(chain), final['boundaries'])

    def test_segment_costs_match_independent_wls(self):
        for name in ('normal', 'same-day-repeat', 'sparse-center'):
            v = _material(name)
            full = seg.solve(v)
            for sc in full['segment_cost_table']:
                if not sc['feasible']:
                    continue
                ref = _independent_cost(v, sc['start_index'], sc['end_index'])
                self.assertIsNotNone(ref)
                self.assertAlmostEqual(ref, sc['cost'], places=8)

    def test_origin_shift_is_bitwise_invariant(self):
        v = _material('normal')
        base = seg.solve(v)
        for delta in (3, 1000, 250_000, 1_000_000_000):
            shifted = seg.solve(seg.shift_origin(v, delta))
            for s1, s2 in zip(base['solutions'], shifted['solutions']):
                self.assertEqual(s2['boundaries'],
                                 [d + delta for d in s1['boundaries']])
                self.assertEqual(s1['objective'], s2['objective'])
                for a, b in zip(s1['segments'], s2['segments']):
                    for c in a['per_center']:
                        self.assertEqual(a['per_center'][c]['slope'],
                                         b['per_center'][c]['slope'])
                        self.assertEqual(a['per_center'][c]['intercept'],
                                         b['per_center'][c]['intercept'])

    def test_repeat_merge_preserves_partition_and_within_constant(self):
        v = _material('same-day-repeat')
        merged, within, _ = seg.merge_repeats(v)
        raw, mer = seg.solve(v), seg.solve(merged)
        self.assertEqual(within, 0.0)  # 重复读数完全相同
        for s1, s2 in zip(raw['solutions'], mer['solutions']):
            self.assertEqual(s1['boundaries'], s2['boundaries'])
            self.assertAlmostEqual(
                s1['segment_cost_total'] - s2['segment_cost_total'],
                within, places=9)
        # 非相同重复观测：组内常数项非零且被精确保留
        v2 = copy.deepcopy(v)
        for r in v2['measurements']:
            if r['id'] == 'repeated':
                r['reading'] = 10.5
        m2, w2, _ = seg.merge_repeats(v2)
        self.assertAlmostEqual(w2, 0.5, places=9)
        a, b = seg.solve(v2), seg.solve(m2)
        for s1, s2 in zip(a['solutions'], b['solutions']):
            self.assertEqual(s1['boundaries'], s2['boundaries'])
            self.assertAlmostEqual(
                s1['segment_cost_total'] - s2['segment_cost_total'],
                w2, places=9)

    def test_sparse_center_allows_only_full_span(self):
        v = _material('sparse-center')
        en = seg.enumerate_partitions(v, 0)
        self.assertEqual(en['n_legal_partitions'], 1)
        full = seg.solve(v)
        self.assertTrue(all(s['boundaries'] == []
                            for s in full['solutions']))

    def test_leave_one_center_out_keeps_calendar_and_full_rerun(self):
        v = _material('normal')
        full = seg.solve(v)
        loo = seg.leave_one_center_out(v)
        self.assertEqual(set(loo), set(full['centers']))
        for c, res in loo.items():
            self.assertEqual(res['calendar'], full['calendar'])
            self.assertEqual(res['excluded_centers'], [c])
            self.assertFalse(res['budget_exhausted'])
            self.assertTrue(all(s['global_optimal']
                                for s in res['solutions']))
        stab = seg.stability_analysis(full, loo)
        # day 6 在剔除任何一个中心后仍存在（a、b 均有跳变，非单一中心信号）
        for lam in ('0', '1', '5'):
            detail = stab['by_lambda'][lam]['boundary_detail'][0]
            self.assertEqual(detail['boundary_day'], 6)
            self.assertEqual(set(detail['survives_when_removing']),
                             {'a', 'b', 'c'})
            self.assertFalse(
                stab['by_lambda'][lam]['one_center_contributes_all_signal'])

    def test_single_center_signal_is_flagged(self):
        v = _material('normal')
        for rec in v['measurements']:
            if rec['center'] == 'b' and rec['day'] >= 6:
                rec['reading'] = 20.0 + 0.1 * rec['day']
            if rec['center'] == 'c':
                rec['reading'] = 30.0 + 0.1 * rec['day']
        full = seg.solve(v)
        loo = seg.leave_one_center_out(v)
        stab = seg.stability_analysis(full, loo)
        self.assertEqual(
            stab['by_lambda']['0'][
                'centers_whose_removal_removes_all_changepoints'], ['a'])
        self.assertTrue(
            stab['by_lambda']['0']['one_center_contributes_all_signal'])
        # 剔除 a 后全 lambda 无变点
        self.assertTrue(all(p['boundaries'] == []
                            for p in loo['a']['path']))

    def test_budget_exhaustion_never_claims_global_optimum(self):
        v = _material('normal')
        v['solver']['max_segments_considered'] = 10
        r = seg.solve(v)
        self.assertTrue(r['budget_exhausted'])
        self.assertGreater(r['segments_considered'], 10)
        self.assertTrue(all(not s['global_optimal']
                            for s in r['solutions']))
        self.assertTrue(all(s['status'] == 'budget_exhausted'
                            for s in r['solutions']))

    def test_completely_infeasible(self):
        v = _material('normal')
        v['min_days'] = len(v['calendar']) + 1
        r = seg.solve(v)
        self.assertTrue(all(s['status'] == 'infeasible'
                            for s in r['solutions']))
        self.assertEqual(seg.enumerate_partitions(v, 0)['n_legal_partitions'],
                         0)

    def test_tie_break_prefers_fewer_changepoints_then_lexicographic(self):
        cal = list(range(8))
        rows = [{'id': f'a{k}', 'center': 'a', 'day': k,
                 'reading': 5.0, 'variance': 1.0} for k in cal]
        v = {'calendar': cal, 'measurements': rows, 'min_days': 2,
             'min_points': 2, 'lambda_grid': [0],
             'solver': {'objective_tie_tolerance': 1e-9,
                        'max_segments_considered': 1000000}}
        r = seg.solve(v)
        # 所有切分费用为 0 且 lambda=0：最少变点（k=0）胜出
        self.assertEqual(r['solutions'][0]['n_changepoints'], 0)
        en = seg.enumerate_partitions(v, 0)
        self.assertEqual(en['best']['n_changepoints'], 0)
        k1 = sorted(x['boundaries'][0]
                    for x in en['all'] if x['n_changepoints'] == 1)
        self.assertEqual(k1, [2, 3, 4, 5, 6])  # 字典序候选确实存在

    def test_invalid_materials_rejected(self):
        bad_cal = dict(_material('normal'), calendar=[1, 1, 2])
        with self.assertRaises(ValueError):
            seg.solve(bad_cal)
        bad_lam = dict(_material('normal'), lambda_grid=[-1])
        with self.assertRaises(ValueError):
            seg.solve(bad_lam)
        bad_day = copy.deepcopy(_material('normal'))
        bad_day['measurements'][0]['day'] = 99
        with self.assertRaises(ValueError):
            seg.solve(bad_day)

    def test_randomized_differential_trials(self):
        random.seed(42)
        for _ in range(120):
            n = random.randint(4, 9)
            cal = list(range(n))
            centers = ['a', 'b'] if random.random() < 0.5 else ['a', 'b', 'c']
            rows, rid = [], 0
            for c in centers:
                for k in range(n):
                    if random.random() < 0.25:
                        continue
                    reps = 1 if random.random() < 0.85 else 2
                    for _ in range(reps):
                        base = random.gauss(0, 2) + 0.2 * k
                        if random.random() < 0.2:
                            base += random.choice([-3, 3])
                        rows.append({'id': str(rid), 'center': c,
                                     'day': cal[k],
                                     'reading': base + random.gauss(0, 0.3),
                                     'variance': round(
                                         random.uniform(0.2, 2), 3)})
                        rid += 1
            v = {'calendar': cal, 'measurements': rows,
                 'min_days': random.randint(2, 3), 'min_points': 2,
                 'lambda_grid': [0, round(random.uniform(0.01, 5), 3),
                                 round(random.uniform(5, 30), 3)],
                 'solver': {'objective_tie_tolerance': 1e-9,
                            'max_segments_considered': 1000000}}
            full = seg.solve(v)
            for sc in full['segment_cost_table']:
                if sc['feasible']:
                    ref = _independent_cost(v, sc['start_index'],
                                             sc['end_index'])
                    self.assertAlmostEqual(ref, sc['cost'], places=7)
            for lam in v['lambda_grid']:
                en = seg.enumerate_partitions(v, lam)
                dp = next(s for s in full['solutions']
                          if s['lambda'] == lam)
                if en['best'] is None:
                    self.assertEqual(dp['status'], 'infeasible')
                else:
                    self.assertAlmostEqual(en['best']['objective'],
                                           dp['objective'], places=9)
                    self.assertEqual(en['best']['boundaries'],
                                     dp['boundaries'])


if __name__ == '__main__':
    unittest.main()
