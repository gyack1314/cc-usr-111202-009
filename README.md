# 中心检验漂移的共同变点与局部趋势

这是研究资料维护用的独立Python项目。`data/`保存合成记录，`research_data.io`负责严格JSON读取与SHA-256来源摘要，`research_data.domain`提供当前原始资料统计，`research_data.cli`将摘要输出为JSON。字段、时间、计数和量纲见[资料约定](docs/data-contract.md)。所有研究、中心及受试者均为虚构。

Python 3.11及以上版本可直接使用，当前没有第三方依赖。在项目根目录运行：

```sh
python -m unittest discover -s tests -v
python -m compileall -q research_data tests
python -m research_data.cli inspect data/normal.json
```

`inspect`只显示已有摘要，不修改来源。文件名以`invalid-`起始的资料包含故意不合法的输入，其他文件是同结构的业务边界。数据摘要与来源指纹应同时保留，原始JSON不覆盖。

## 共同变点分段

`segment`命令在多中心控制样本上做共同变点分析，输出每段每中心截距/斜率、支持记录、段费用、动态规划最优性表、全lambda分段路径、逐中心剔除完整重算与稳定性汇总：

```sh
python -m research_data.cli segment data/normal.json
python -m research_data.cli segment data/normal.json --out out/normal-segment.json
```

所有中心共享边界，段内每中心独立加权最小二乘（截距+斜率，原点=段首日）；动态规划对`lambda_grid`逐个求`总段费用+lambda*变点数`的全局最小，同分按目标值容差、少变点、边界字典序裁决。预算`max_segments_considered`耗尽时只标记不完整、不声明全局最优。方法细节、剪枝与缓存的合法条件、换原点/同日合并不变性见[资料约定](docs/data-contract.md)。结论是统计关联，不断言试剂导致。

可使用固定镜像容器：`docker build -t research-material .`，随后`docker run --rm research-material`查看资料；`docker run --rm research-material python -m unittest discover -s tests -v`运行现有检查。全部资料在项目内，运行不需要外部账号或服务。
