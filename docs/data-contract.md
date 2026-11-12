# 控制样本日历

calendar是全研究公共日历，严格升序整数；资料中的day必须在其中。measurements每行有全局唯一id、center、day、reading和已知正variance。相同center/day可有独立重复观测，全部权重保留。min_days指占用日历格的数量，不是某中心观测行数；min_points指该段某中心不同观测日期数。

分段覆盖calendar全部日历格且无重叠，边界以新段首日表示，各段左闭右开，最后段覆盖末日。斜率的时间自变量是实际day减本段首day，不用行序号替代。每中心的两参数加权最小二乘只有满秩才合法，任何中心不支持该段就整段不可用。常数项、截距和平移的口径都包含在资料解释中。

已有domain只给整段常均值的描述汇总，不拟合局部趋势或分段。

## 共同变点分段（research_data.segmentation）

分段由 `research_data.segmentation` 实现，`python -m research_data.cli segment <file>` 输出完整结果。口径与本契约一致：

- 边界只能取日历格之间，所有中心共享；分段覆盖全部日历格、左闭右开，最后段覆盖末日。
- 段 `[i,j)` 内每个中心独立拟合加权最小二乘直线 `reading = alpha + beta*(day-calendar[i])`，权重 `1/variance`；时间自变量是相对段首日的实际日历日差，不使用行序号，也不使用全局大日值，故平移日历原点在浮点层面位级不变。
- 段费用为各中心加权残差平方和之和。段合法条件：跨至少 `min_days` 个日历格；每中心至少 `min_points` 个不同观测日期；每中心加权设计矩阵（截距列、时间列）满秩。任一中心不满足则整段不可用，缺测不补零。
- 同日重复观测逐行保留权重参与拟合；`merge_repeats` 的等价合并不改变拟合值与分段，只改变一个与分段无关的组内常数项 `within_ss = 每组 sum w(y-ybar)^2`，即原始段费用 = 合并段费用 + 该段覆盖组的 within_ss。
- 对 `lambda_grid` 中每个 lambda，动态规划求全局最小 `总段费用 + lambda*变点数`。同分依次按：目标值（`objective_tie_tolerance` 容差内相等）、较少变点数、边界日期序列字典序。小日历可用 `enumerate_partitions` 全枚举所有合法分段独立核对。
- `max_segments_considered` 按新评估的候选段计数（费用缓存命中不重复计数）；预算耗尽时解标记为 `budget_exhausted`、`global_optimal=false`，不声明全局完成。
- 段费用缓存是精确复用：费用只依赖区间内原始观测、固定方差与段首日，与分段方案和 lambda 无关。不使用任何依赖费用单调性的剪枝（PELT 类不等式只对常均值/特定凸费用成立，对带截距斜率且稀疏的多中心模型不合法）。
- `leave_one_center_out` 逐中心剔除后用同一日历、同一参数完整重算，其他中心不缩短时间范围；若剔除某中心后所有变点消失，标记该中心贡献全部信号。变点只表示测量关系的统计改变，与换试剂时间重合是关联而非因果证据。
