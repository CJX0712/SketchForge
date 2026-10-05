# SketchForge 架构文档

> Author: 晨星
> 版本：0.1.0 · 对应 `sketchforge/__version__ = "0.1.0"`
> 本文描述**设计契约**。凡标注【架构实测】的数字为架构师本机（Windows 11 / Python 3.13.14 / numpy 2.5.3 / scipy 1.18.1 / datasketches 5.2.0）实跑所得；【Pilot 实测】为仓库 `pilot/` 脚本产出；**凡未实测一律标【待验证】，不填预期值**。

---

## 1. 定位与非目标

SketchForge 是一个**固定内存预算下的数据流摘要（stream sketching）工具包**：在给定总字节预算 B 下，同时回答四类查询——基数（Q1）、分位数/秩（Q2）、点查频率（Q3）、重尾项（Q4）。

系统的旗舰能力不是"某一类 sketch 更准"，而是**预算感知的自适应分配**：用流前缀拟合各查询的"误差-预算弹性"，按边际收益均等把 B 分配到各模块。

**非目标（明确不做，防止过度设计）**：不做分布式、不做 GPU、不做在线服务、不接 LLM API、不做微服务编排、不引 `ray`/`dask`/`fastapi`。MVP 形态 = 一个包 + 一个 CLI + 一份可审计的 `benchmark.json`。

---

## 2. 分层与单向无环图

```
L5  cli.py                       examples/run_demo.py
        │
L4  pipeline/  sketch_pipeline.py · benchmark.py · stages.py
        │
L3  hpo/  allocation.py · budget_search.py · grid.py · selector.py
    training/  fit_error_model.py · recommender.py
        │
L2  preprocess/  encode.py · window.py
    eval/        metrics.py · aggregate.py · validate.py · report.py
        │
L1  data/      specs.py · streams.py · loaders.py · oracle.py
    sketches/  base.py · budget.py · registry.py · backend.py
               cardinality.py · quantile.py · frequency.py · heavyhitters.py
        │
L0  core/  types.py · errors.py · config.py · interfaces.py
           seed.py · hashing.py · runtime.py · io_utils.py
```

**三条硬规则（由 `tests/test_layering.py` 用 AST 强制执行）：**

1. **只允许向下 import。** 高层可 import 低层，低层不得 import 高层。
2. **同层不互相 import。** `data` 与 `sketches` 同在 L1，二者之间的数据交换一律经由 L0 的 `core.types` 契约（`Stream` / `Query` / `Estimate`），禁止直接互引。
3. **绝对导入。** 跨包一律 `from sketchforge.core.types import Query`；禁止相对导入越过顶层包边界。

> 为什么把 `data` 与 `sketches` 放在同一层且不互通：真值计算（`ExactOracle`）与 sketch 实现必须**互不依赖**，否则"真值"可能被被测对象的假设污染。真值一律由全量暴力计算（Counter / `np.sort` + `searchsorted`）得出。

---

## 3. 模块职责表

状态图例：✅ 已实现（仓库中已有文件与测试） · 🟡 规划中（本文定义契约，尚未落地）

### L0 `core/`

| 文件 | 状态 | 职责 | 关键符号 |
|---|---|---|---|
| `types.py` | ✅ | 全系统不可变数据契约 | `QueryType{CARDINALITY,QUANTILE,FREQUENCY,HEAVY_HITTERS}`；`Query(kind,q,key,top_k)`；`Estimate(value,lower,upper,n_updates)`；`MetricSpec(name,unit,higher_is_better).score()`；`MethodSpec(name,tier,backend,query_types,supports_merge,deterministic,hash_seed_injectable)`；`BudgetPlan(alloc,total_bytes,strategy)` |
| `errors.py` | ✅ | 错误码 + 注册期冲突检测 | `SketchForgeError(code)`；子类注册时校验**错误码与类名双重唯一**（重复即 `RuntimeError`）。见 §10 |
| `config.py` | ✅ | ENV 覆盖 + 校验 | `Config`（见 §9）；`Config.from_env()`；`validate()` |
| `interfaces.py` | ✅ | `Sketch` Protocol + 预算诚实性断言 | `Sketch`（runtime_checkable）；`assert_budget_honest()`（声明字节 vs `len(serialize())` 偏差 >2% → `E302`） |
| `seed.py` | ✅ | **唯一 seed 入口** | `set_all(seed)->SeedState`；`SeedState.derive(tag)`（blake2b 派生）；`spawn_generators(n)`；`require()`（未设种 → `E102`） |
| `hashing.py` | ✅ | 全系统唯一哈希出口 | `stable_key`；`hash_u64`；`hash_index`；`hash_sign`；`hash_array_u64`（向量化） |
| `runtime.py` | ✅ | 资源与环境探测 | `now_sec()`；`rss_bytes()->int\|None`；`PeakTracker`；`env_probe()` |
| `io_utils.py` | ✅ | Windows 编码安全 IO | `open_text(..., encoding="utf-8")`；`dump_json`（原子写） |

### L1 `data/` · `sketches/`

| 文件 | 状态 | 职责 | 关键符号 |
|---|---|---|---|
| `specs.py` | ✅ | 流规格（可序列化、自校验） | `StreamSpec(kind,n,domain,seed,zipf_alpha,drift_rate,value_dist)`，kind ∈ {uniform,zipf,pareto,gmm,drift_abrupt,drift_gradual,low_card,high_card,elephant_mice} |
| `streams.py` | ✅ | 合成流生成（确定性） | `generate(spec)->Stream`；`Stream(keys,values,spec)` |
| `loaders.py` | ✅ | 文件载入 | `load(path,*,fmt,key_col,value_col)->Stream`（csv/jsonl/npy） |
| `oracle.py` | ✅ | **精确真值**（非 sketch） | `ExactOracle: cardinality/quantile/rank/frequency/top_k/total` |
| `sketches/base.py` | 🟡 | 公共基类 | `AbstractSketch`：`update/update_batch/estimate/merge/memory_slots/memory_bytes/serialize/params_for_budget` |
| `sketches/budget.py` | 🟡 | 预算货币与单查询反解 | `slots_of`；`bytes_after_ingest`；`fit_param_under_budget`；`min_feasible_bytes` |
| `sketches/registry.py` | 🟡 | 方法注册与发现 | `register`；`sketch_registry`；`methods_for(query, backend_mode)`；`select_primary_methods(deterministic_only)` |
| `sketches/backend.py` | 🟡 | 后端探测与降级 | `available_datasketches/available_scipy/backend_status/detect_backends` |
| `sketches/{cardinality,quantile,frequency,heavyhitters}.py` | 🟡 | 四族实现（Tier-0 适配 + Tier-1 手写） | 见 §6 |

### L2 `preprocess/` · `eval/`

| 文件 | 状态 | 职责 | 关键符号 |
|---|---|---|---|
| `preprocess/encode.py` | 🟡 | item ↔ int64/str 规范编码（Tier-0 重尾需 str 键，Tier-1 需 int64） | `KeyEncoder: fit/encode/decode/save/load` |
| `preprocess/window.py` | 🟡 | 翻转窗 / 滑动窗切分 | `tumbling(stream,size)`；`sliding(stream,size,hop)` |
| `eval/metrics.py` | 🟡 | 指标定义 + **方向归一** | `METRICS`；`score_query(query,estimate,oracle)`；`error_bound_declared(sketch,query)`；`SPEC_TOLERANCE` |
| `eval/aggregate.py` | 🟡 | 3-seed 聚合 | `aggregate(records)->list[Aggregate]`（mean / std(ddof=1) / n_seeds / values） |
| `eval/validate.py` | 🟡 | 确定性校验 + 审计 | `VOLATILE_PATHS`；`canonicalize`；`digest`；`assert_two_runs_identical`；`assert_stochastic_in_band`；**非空虚性审计**（§8） |
| `eval/report.py` | 🟡 | 渲染 | `render_markdown(report)->str` |

### L3 `hpo/` · `training/`

| 文件 | 状态 | 职责 | 关键符号 |
|---|---|---|---|
| `hpo/allocation.py` | 🟡 | **跨查询分配**（见 §7） | `AllocationPlan`；`AllocationStrategy`；`UniformStrategy`；`ElasticityWaterFilling`；`StaticOptimalFrozen`；`PerSegmentOptimal`；`SimplexGridSearch`；`DegenerateFallback`；`largest_remainder`；`AllocContext` |
| `hpo/budget_search.py` | 🟡 | 单查询预算反解 | `fit_param_under_budget`；`min_feasible_bytes`；`param_grid` |
| `hpo/grid.py` / `selector.py` | 🟡 | 预算网格；预算内 top-k 配置 | `BudgetGrid`；`select_top` |
| `training/fit_error_model.py` | 🟡 | 拟合"误差-预算律" | `ErrorLaw(a,c,floor,r2,degenerate,reason)`；`fit_error_law_closed_form`；`crosscheck_with_curve_fit`（离线） |
| `training/recommender.py` | 🟡 | 预算感知选型器（**独立实验**） | `SketchRecommender: fit/recommend/evaluate` |

### L4 `pipeline/` · L5 入口

| 文件 | 状态 | 职责 | 关键符号 |
|---|---|---|---|
| `pipeline/sketch_pipeline.py` | 🟡 | 编排 | `SketchForgePipeline.run()->RunResult`；结果字段用 **`_report` 后缀**（见 §11-R4） |
| `pipeline/benchmark.py` | 🟡 | 报告生成 | `benchmark(cfg)->dict`（JSON-ready） |
| `pipeline/stages.py` | 🟡 | 阶段函数 + **唯一评测入口** | `stage_probe_split`；`evaluate_allocation(plan, eval_stream, oracle, seed_state)` |
| `cli.py` | 🟡 | argparse 入口 | `demo/bench/list-methods/list-backends/fit/recommend/verify` |

---

## 4. `Sketch` 协议与四个能力位

```python
@runtime_checkable
class Sketch(Protocol):
    SUPPORTS_MERGE: bool          # 能否 merge
    DETERMINISTIC: bool           # 同 seed 是否逐位可复现
    HASH_SEED_INJECTABLE: bool    # 哈希种子能否从外部注入
    TIER: int                     # 0 = datasketches, 1 = 纯 numpy
    PARAM_AXIS: str               # 可二分的参数名（lg_k / k / num_buckets / lg_max_k）

    def update(self, item: object) -> None
    def update_batch(self, items: np.ndarray) -> None     # 向量化路径（Tier-1 性能关键）
    def estimate(self, query: Query) -> Estimate
    def merge(self, other: Self) -> Self
    def memory_slots(self) -> int
    def memory_bytes(self) -> int
    def serialize(self) -> bytes
    @classmethod
    def params_for_budget(cls, budget_bytes: int) -> dict[str, object]
```

**能力位是"声明即契约"，不是"事后抛异常"。** 实现若不具备某能力，必须在类属性上如实声明 `False`，而不是在方法体里抛 `NotImplementedError` 撑门面——因为 benchmark 与门禁都**读类属性**来决定是否把该方法纳入某项对比。

**能力位实测依据**（datasketches 5.2.0，【架构实测】）：

| 类 | `SUPPORTS_MERGE` | `DETERMINISTIC` | `HASH_SEED_INJECTABLE` | 备注 |
|---|---|---|---|---|
| `cpc_sketch(lg_k, seed=9001)` | **False**（无该方法） | True | **True** | |
| `hll_sketch(lg_k, tgt_type)` | **False**（无该方法） | True | False | 无 `serialize()`，需转发 `serialize_compact()` |
| `kll_floats_sketch(k)` | True | **False** | False | 同进程两次构造结果不同（499 / 493） |
| `req_floats_sketch(k, is_hra)` | True | **False** | False | 同进程两次：495 / 493 |
| `quantiles_doubles_sketch(k)` | True | **False** | False | 同进程两次：499 / 500 |
| `count_min_sketch(d, w, seed=9001)` | True | True | **True** | |
| `frequent_items_sketch(lg_max_k)` | True | — | False | `serialize()` **强制要求 serde 参数**，无参调用 `TypeError` → **弃用** |
| `frequent_strings_sketch(lg_max_k)` | True | True | False | 重尾族 Tier-0 唯一可用实现 |

> 分位数族在 5.2.0 上非确定的根因是 C++ 侧压缩使用 `random_device`，Python 绑定未暴露 seed 入口。这是**库的固有性质，不是我们的 bug**，因此按 §8 的 G2/G3 处理，不做任何掩盖。

**`estimate()` 统一返回 `Estimate`，不允许返回裸 float**：重尾项返回的是列表，四类查询必须共用同一个返回容器，由 `MetricSpec.higher_is_better` 决定方向。

---

## 5. 预算货币三层体系

预算必须是**可测、可比较、能真实区分不同方法成本**的量。三层定义，主次分明：

| 层 | 名称 | 定义 | 用途 | 能否作为比较依据 |
|---|---|---|---|---|
| **L1** | **字节（主币）** | `memory_bytes()` = **ingest 之后**实测的 `len(serialize())`（Tier-0）/ 数组 `nbytes` 之和（Tier-1） | **等预算对比与审计的唯一依据** | ✅ 唯一主币 |
| **L2** | 槽位（注解） | `memory_slots()` = 声明式存储单元数（HLL/CPC `2**lg_k`、CM `d×w`、KLL `≈3k`、SpaceSaving `1.5·2**lg_max_k`） | 参数反解、可读性 | ❌ **跨族不可比**（HLL_8 寄存器 1 字节 vs CM 计数器 8 字节） |
| **L3** | 时间 | wall-clock | 仅当 L1 **完全相等**时作 tie-breaker | ❌ 禁止作为主币（机器相关，且会重演"成本不可区分"） |

**为什么主币必须在 ingest 之后测**【架构实测】：CPC 的序列化大小随基数增长——`lg_k=10` 时 640 B，`lg_k=12` 时 2428 B（n=20 000）。按参数解析推算会把预算系统性算小。

**四条强制不变量**（违反即 `E302` / `E503`）：

1. **可测且非常量**：`memory_bytes() > 0` 且有限；同一类取 3 组不同参数，三组字节必须两两不等。
2. **单调性**：`params_for_budget(2B)` 的实测字节 ≥ `1.5 ×` `params_for_budget(B)` 的字节。
3. **分离度**：同一 `(dataset, query, budget)` 组内，方法两两成本比 `c = max(bytes)/min(bytes)`，若 `c < 1.20` → 标 `cost_indistinguishable: true` 并 FAIL。
4. **实测而非声明**：`assert_budget_honest(sketch)` 校验 `memory_bytes()` 与 `len(serialize())` 偏差 ≤ 2%，在 `Pipeline.run()` 与单测中各调用一次。

---

## 6. Tier-0 → Tier-1 降级路径

### 6.1 后端选型（【架构实测】已确认可行）

| 维度 | Tier-0 `datasketches==5.2.0` | Tier-1 numpy+scipy 手写 |
|---|---|---|
| 轮子先验 | `pip download --no-deps --only-binary=:all: --python-version 3.13 --abi cp313 --platform win_amd64 datasketches` → **成功**（`datasketches-5.2.0-cp313-cp313-win_amd64.whl`，509 KB）；3.12/cp312 同样存在 | 无下载（numpy 2.5.3 / scipy 1.18.1 wheel 已实测可装） |
| 吞吐 | 20 万次 update：CPC **0.028 s**、KLL **0.044 s** | 需 `update_batch` 向量化，逐元素 Python 循环会超时 |
| 确定性 | 部分方法非确定（§4） | **自行接管 RNG → 100% 确定** |
| 结论 | 首选（`backend=auto` 时启用） | **必备兜底**（离线零下载可跑 demo） |

### 6.2 探测顺序（一条容易踩的坑）

`datasketches` **import 时硬依赖 numpy**：未装 numpy 时不是普通 `ImportError`，而是 nanobind 的 `Critical nanobind error`【架构实测】。因此探测必须**先 `import numpy` 成功，再 `import datasketches`**，并使用宽捕获：

```python
def available_datasketches() -> bool:
    try:
        import numpy  # noqa: F401  -- 必须先于 datasketches
        import datasketches  # noqa: F401
    except Exception:
        return False
    return True
```

### 6.3 降级语义

- `Config.backend ∈ {auto, tier0, tier1}`；`auto` = 有 Tier-0 即用，否则全 Tier-1。`SKETCHFORGE_NO_TIER0=1` 等价于强制 `tier1`。
- **缺失即 skipped，不伪造数字**：每条记录强制带 `status ∈ {ok, skipped}`；`skipped` 记录的 `metric.value` 必须是 `null` 且携带 `skip_reason`（`backend_missing:datasketches` / `no_feasible_param_under_budget` / `merge_unsupported` / `law_degenerate`）。
- `skipped` 记录**不参与** mean/std 聚合与排名；`audits.skipped_count` 与 `skip_reasons` 如实落盘。

---

## 7. `hpo/allocation.py`：策略族与分配流程

### 7.1 模块拓扑（按算法侧修订版）

共享 bottom-k 核**只服务 Q1（基数）与 Q3（频率）**；Q2（分位数）与 Q4（重尾项）走独立模块，并通过**跨模块误差补偿**通道与共享核联动。因此分配的对象是**模块**而非查询：

| 模块 | 服务的查询 | 说明 |
|---|---|---|
| `M_shared` | Q1 + Q3 | 共享 bottom-k 核，一个参数同时影响两类查询误差 |
| `M_q2` | Q2 | 独立分位数模块 |
| `M_q4` | Q4 | 独立重尾模块 |

> 修订原因（算法侧发现）：bottom-k 是**去重后的均匀样本**，对多重集秩有系统性偏差，低基数时重尾召回失效，因此不能服务 Q2/Q4。

### 7.2 策略族

```python
class AllocationStrategy(Protocol):
    name: str
    def allocate(self, laws, budget: int, ctx: AllocContext) -> dict[str, int]: ...

UniformStrategy        # 消融对照：b_m = B/|M|
StaticOptimalFrozen    # 主门禁基线：在校准 seed 集上 grid search 后**冻结**的单一分配
ElasticityWaterFilling # 旗舰：按段拟合弹性做边际收益均等
PerSegmentOptimal      # 金标准（不可实现上界）：每段各自 grid search
SimplexGridSearch      # 求解器（在拟合律上解析求值），供 StaticOptimal 与 PerSegment 使用
DegenerateFallback     # 律退化（c ≤ 0 或 r² < 0.9）时退回均匀并记 reason
```

### 7.3 分配流程

```
stream ──► stage_probe_split ──► probe（前缀） + eval（余下）
                                    │              │
                    拟合网格上构建 → (b, err) 样本   │
                                    │              │
                        fit_error_law_closed_form  │
                                    │              │
                    ErrorLaw(a, c, floor, r²)      │
                                    │              │
                        allocate(laws, B, ctx)     │
                                    │              │
                           AllocationPlan ─────────┘
                                    │
                     evaluate_allocation(plan, eval, oracle, seed_state)
                                    │
                       err_i ─► e_i = err_i/ε_i ─► G = ∏ e_i^{w_i}
```

### 7.4 目标函数与闭式解

主口径为**几何平均** `G = ∏_i e_i^{w_i}`（权重默认等权 1/4）。在 `err_i(b) = max(a_i·b^{-c_i}, floor_i)` 下：

```
minimize   Σ_i w_i · log err_i(b_{m(i)})
s.t.       Σ_m b_m = B,  b_m ≥ b_min_m
```

目标函数**凸**（`-c·log b` 凸、常数凸、取 max 保凸）→ KKT 即全局最优。无约束绑定时闭式为：

```
b_m = ( Σ_{i ∈ served(m)} w_i · c_i ) / λ        ← 模块级推广
```

即：**共享核的边际收益是它所服务查询的加权弹性之和**，因此它天然拿到比任一单一查询更多的预算——这是"共享"在预算层面的结构性理由。当弹性全相等时，`b_m ∝ |served(m)|·w·c`，退化为按服务查询数分配。

**两条必须披露的性质**【架构实测，合成参数下的数值验证】：

1. **纯幂律区间内最优解与 `a_i` 和总预算 B 无关**（`b_m ∝ Σ w_i c_i`）。"自适应"只在 `b_min` / 饱和（`floor`）约束生效时才真正体现为预算的函数。不得对外表述为"随预算动态调整"。
2. **归一化常数 ε 在 `G_ratio` 中完全抵消**：`argmin ∏(err/ε)^w` 的第二项 `−Σ w_i log ε_i` 与 b 无关 ⇒ **最优解与 ε 的选取无关**，不存在"调 ε 让旗舰赢"的空间；求和口径则依赖 ε，是可调旋钮。这是选几何平均作主口径的**数学理由，而非审美理由**。

合成参数对照（B=32768，`c=[0.50,0.35,0.62,0.30]`）：优化几何平均的闭式解 `G_ratio=0.9823`；优化误差和的 water-filling `G_ratio=1.0862`（同一分配在原始和口径下却是 +5.5%）；网格穷举金标准 `0.9828`（gap 0.05%）。**这组对照说明"分配器优化的目标"必须与"门禁测的口径"一致**，否则会出现伪胜。上述为**合成参数**，真实弹性以 pilot 实测为准（见 `model_card.md`）。

### 7.5 共享评测入口（禁止两套口径）

`pipeline/stages.py::evaluate_allocation()` 是**唯一**评测入口，`UniformStrategy` 与 `ElasticityWaterFilling` 都走它，函数本身不知道调用者是谁。三条结构约束：

- **C1**：基线不是一个"对比脚本"，而是 `AllocationStrategy` 的一个实现。
- **C2**：`method_per_query` 由 `AllocContext` 统一注入，策略**无权**自选方法（否则"分配"与"选型"两个变量混在一起，赢了不知赢在哪）。方法选型是独立实验，单独出一行。
- **C3**：单测断言两次调用的参数除 `per_query` 外**逐字段相等**。

**整数化**：字节是整数，规则固定为**最大余数法 + 枚举序打破平局**（`np.argsort(..., kind="stable")`），并断言 `Σ b_m == B` 精确成立（不差 1 字节）。

---

## 8. 三级确定性门禁

### 8.1 唯一 seed 入口

```python
state = core.seed.set_all(cfg.seed)      # random.seed + np.random.seed 一次设齐
state.derive("data") / ("probe:Q2") / ("sketch:Q1") / ...
```

- `derive()` 用 blake2b 派生互不重叠的子流，避免组件争抢同一 RNG 状态。
- **禁止内置 `hash()`**（`PYTHONHASHSEED` 运行时不可改，str 哈希跨进程随机）：`core/hashing.py` 是全系统唯一哈希出口，由单测扫描源码强制。
- **分配器零 RNG**：`ElasticityWaterFilling.allocate()` 是 `(laws, B, ctx)` 的**纯函数**。两条硬测试：同输入两次调用结果逐位相等；monkeypatch `np.random`/`random` 使其抛异常后 `allocate()` 仍正常返回。

### 8.2 拟合必须走闭式最小二乘

【架构实测】`scipy.optimize.curve_fit` 换初值后结果在第 10 位起不同（`p0=[1,0.5]`→0.901549125854 / `p0=[5,0.05]`→0.901549126148）——它是迭代求解器，**不是良定义的函数**。

**规定**：主链路只用闭式 log-log OLS（`ĉ = −Σ(x−x̄)(y−ȳ)/Σ(x−x̄)²`，`â = exp(ȳ + ĉ·x̄)`），用 `np.mean/np.sum` 显式表达，**不用 `lstsq`/`polyfit`**（LAPACK 跨平台不保证），不用 `curve_fit`。scipy 仅作离线交叉校验（`crosscheck_with_curve_fit`，结果只进 `audits`）。取对数前须 `max(err, 1e-6)` 防 `log(0)`（重尾 `1−F1` 与相对误差都可能恰为 0），被 clamp 的点如实上报。

### 8.3 三级门禁

| 门禁 | 作用域 | 判据 | 强度 |
|---|---|---|---|
| **G1** | `--backend tier1`（或 `SKETCHFORGE_NO_TIER0=1`）跑两遍 | 剔除 volatile 后 canonical digest **完全相同，包含所有 value** | 强，红即失败 |
| **G2** | 默认 `auto` 跑两遍 | 剔除 volatile **且**把 `DETERMINISTIC=False` 记录的 value/score 替换为哨兵 `"<stochastic>"` 后 digest 一致 | 强 |
| **G3** | 所有 `DETERMINISTIC=False` 记录 | `value ≤ error_bound_declared × safety`（`safety=3.0`）；`error_bound_declared` **必须来自库自身 API**（KLL `get_normalized_rank_error`、REQ `get_RSE`、HLL `get_rel_err`），不得自填 | 强（违者 `E501`/`E502`） |

G3 把"不可复现"转成"**有界**"——既不假装复现，也不删除这些方法。

**volatile 白名单**（显式列举，不做"看起来像时间就删"的启发式）：

```python
VOLATILE_PATHS = ("generated_at", "records[].elapsed_sec",
                  "aggregates[].elapsed_sec", "environment.hostname", "environment.pid")
```

`canonicalize()`：深拷贝 → 白名单路径置 `null` → 随机记录 value/score 置 `"<stochastic>"` → `json.dumps(sort_keys=True, ensure_ascii=False)` → sha256。两次运行须走**子进程**（避免同进程缓存污染）。CI 固定 `PYTHONHASHSEED=0`、`n_jobs=1`（`Config.validate()` 对 `n_jobs != 1` 直接 `E101`）。

---

## 9. 配置项与环境变量

`core/config.py::Config`（✅ 已实现）：

| 字段 | 默认值 | 说明 |
|---|---|---|
| `seed` / `seeds` | `1337` / `(1337,1338,1339)` | 评测 seed 三连 |
| `backend` | `"auto"` | `auto`/`tier0`/`tier1` |
| `n` | `100_000` | 流长 |
| `datasets` | `("zipf_a1.1","drift_abrupt","low_card","elephant_mice")` | 预注册评测集 |
| `budgets_bytes` | `(4096, 65536)` | 预算档位 |
| `deterministic_only` | `False` | 仅纳入确定性方法 |
| `out_dir` | `Path("artifacts")` | 产物目录 |
| `time_budget_sec` / `n_jobs` / `safety` | `60.0` / `1` / `3.0` | 超时；并行（必须为 1）；G3 带宽系数 |

环境变量：

| 变量 | 映射到 | 状态 |
|---|---|---|
| `SKETCHFORGE_SEED` | `seed`（并派生 `seeds = (s, s+1, s+2)`） | ✅ |
| `SKETCHFORGE_BACKEND` | `backend` | ✅ |
| `SKETCHFORGE_N` | `n` | ✅ |
| `SKETCHFORGE_BUDGETS` | `budgets_bytes`（逗号分隔） | ✅ |
| `SKETCHFORGE_OUT_DIR` | `out_dir` | ✅ |
| `SKETCHFORGE_DETERMINISTIC_ONLY` | `deterministic_only` | ✅ |
| `SKETCHFORGE_SAFETY` | `safety` | ✅ |
| `SKETCHFORGE_NO_TIER0` | 真值 → 强制 `backend="tier1"`（跑 G1 用） | ✅ |
| `SKETCHFORGE_PROBE_FRACTION` | 探针前缀占比（默认 `0.10`，下限 10 000 条） | 🟡 规划中 |
| `SKETCHFORGE_GATE_DELTA` | 门禁阈值 δ（**默认 `None` = 未冻结**；未冻结时门禁报 `UNFROZEN` 而非 PASS/FAIL） | 🟡 规划中 |
| `SKETCHFORGE_GATE_NO_HARM` | 是否启用"无查询劣化"约束（默认 `False`） | 🟡 规划中 |
| `SKETCHFORGE_GATE_WEIGHTS` | 聚合权重（默认等权） | 🟡 规划中 |
| `PYTHONHASHSEED` | 由 CI 固定为 `0`（本进程内不可改，仅影响子进程） | ✅ CI |

**探针纪律**：探针必须是**流的前缀**（`stream[:probe_n]`），`oracle` 只在 eval 段上计算，`probe ∩ eval = ∅`；探针不做 bootstrap/随机抽样（如需降采样用固定步长）。校准 seed 集（用于 `StaticOptimalFrozen`）必须与评测 seed 集**不相交**，否则基线会"看过答案"。

---

## 10. 审计与非空虚性规则

### 10.1 通用规则（本项目最有价值的一条工程约束）

> **任何比值型审计，若其分母是所有被比较对象共享的常量，一律自动标 `vacuous_denominator: true` 并判 FAIL。**

分配实验中所有策略的总成本都被构造成 B，若仍取 `c = cost(A)/cost(B)` 则 `c ≡ 1.00` 恒成立——与"调用次数当预算"是同一个病，只是换了一层皮。禁止的分母：`B`、sketch 个数（两策略相同）、评测次数。

### 10.2 分配场景的替换审计

**外层（比较对象是策略）**

| 审计项 | 定义 | 阈值 |
|---|---|---|
| `allocation_separation` | `max_m max(b^A_m, b^U_m)/min(...)` | ≥ 1.20 |
| `budget_utilisation` | `Σ achieved_m / B`（每策略） | ≥ 0.50 |
| `budget_response` | `\|G(2B) − G(B)\| / G(B)` | ≥ 0.02 |
| `spend_parity` | `\|Σb^A − Σb^U\| ≤ 1` 字节 | 必须成立 |
| `method_parity` | 两策略 `method_per_query` 完全相同 | 必须成立 |

**内层（比较对象是单查询方法，分配实验的根）**

| 审计项 | 定义 | 阈值 |
|---|---|---|
| `per_query_cost_discriminability` | 拟合网格上相邻预算点的**实测字节比** | ≥ 1.50 |
| `law_quality` | 每查询 `r² ≥ 0.9` 且 `c_i > 0` | 必须成立 |

任一项 FAIL → 主门禁判红，且**此时 `G_ratio` 一律不许写进 headline**。

### 10.3 错误码（`core/errors.py`，✅ 已实现，注册期校验码与类名双重唯一）

| 段 | 码 | 含义 |
|---|---|---|
| 配置 | E101 配置非法 · E102 缺失 seed · E103 后端模式非法 · E104 预算守恒失败 | |
| 数据 | E201 spec 非法 · E202 载入失败 · E203 空流 · E204 真值不可用 | |
| Sketch | E301 参数非法 · **E302 预算货币不可区分** · E303 merge 类型不匹配 · E304 merge 不支持 · E305 序列化失败 · E306 估计不可用 | |
| 后端 | E401 缺失 · E402 import 失败 | |
| 管线 | E501 确定性违规 · E502 指标越界 · **E503 成本不可区分** · E505 超时 | |

规划中新增：E307 律退化 · E506 分配超预算 · E507 审计分母空虚 · E508 门禁阈值未冻结。

---

## 11. 已知工程风险与规避

| # | 风险 | 规避 |
|---|---|---|
| R1 | Windows 编码（cp936 控制台） | 所有 `open()` 走 `io_utils.open_text`（显式 utf-8）；`cli.py` 首行 `sys.stdout.reconfigure(encoding="utf-8", errors="replace")`；输出禁用 emoji/制表符画线 |
| R2 | numpy 2.x 移除别名（`np.float_`→`np.float64`、`np.NaN`→`np.nan`、`np.trapz`→`np.trapezoid`、`np.int0`、`RandomState`） | 源码正则扫描单测；依赖锁 `numpy>=2.1,<3` |
| R3 | `n_jobs>1` 破坏 RNG 可复现 | `Config.validate()` 对 `n_jobs != 1` 直接 `E101`；全文无 `joblib`/`multiprocessing` |
| R4 | **属性名遮蔽方法名**（`self.benchmark` 遮蔽模块级 `benchmark()`） | 结果字段统一 `_report` 后缀；pipeline 内部方法 `_run_benchmark()` |
| R5 | 相对导入越界 | 绝对导入；`tests/test_layering.py` AST 扫描 |
| R6 | datasketches import 硬依赖 numpy（nanobind 崩溃而非 ImportError） | 探测顺序：先 numpy 后 datasketches；宽捕获 |
| R7 | `frequent_items_sketch.serialize()` 需 serde | 改用 `frequent_strings_sketch` + `KeyEncoder` 稳定编码 |
| R8 | `hll_sketch` 无 `serialize()` | 适配器转发 `serialize_compact()` |
| R9 | CPC/HLL 无 merge | `SUPPORTS_MERGE=False` + `merge()` 抛 `E304`；merge 一致性用例标 skipped |
| R10 | 分位数族跨进程随机 | G2 哨兵掩码 + G3 误差带 + Tier-1 兜底走 G1 |
| R11 | CPC 序列化大小随基数增长 | 预算在 ingest 后测；同时记录 `memory_bytes_init` |
| R12 | REQ 极贵【架构实测】k=100 → 15 680 B，k=200 → 23 308 B | 小预算下大概率无可行参数 → 真实 skipped，禁止硬塞 |
| R13 | `count_min_sketch.get_upper_bound()` 实测返回 0.0 | 频率指标自算 `\|est−true\|/max(true,1)`，不依赖库 upper_bound |
| R14 | Windows 无 stdlib RSS | `rss_bytes()`：psutil（可选）→ tracemalloc → `None`，不伪造 |
| R15 | λ 二分用固定区间会越界【架构实测】 | λ 区间解析导出：`[min_i w_i c_i / b_cap_i, max_i w_i c_i / b_min_i]`；固定 200 次迭代 |

---

## 12. 待验证清单

以下均**未实测**，文档中不得当作事实引用：

1. HLL / CPC 的 pilot 标定（`pilot/p2_hll_cpc.py` 已写，**尚无输出文件**）→ 基数族弹性 `c_Q1` 未知。
2. Q3（频率）与 Q4（重尾）的弹性 `c_Q3` / `c_Q4` 未知 → 分配实验的效应量无法预估。
3. `b_shared` 同时影响 Q1/Q3 的联合弹性（双弹性是否可加）→ 待算法侧与工程实测确认。
4. 跨模块误差补偿的开/关消融结果 → 待实测（需 2×2 消融，否则无法归因）。
5. 分配实验端到端 `G_ratio`、`optimality_ratio`、`allocation_separation` 实测值 → 待实测。
6. `datasketches` 在 ubuntu-latest / manylinux 上 cp312、cp313 wheel 是否存在（本机只验了 win_amd64）；Docker（`python:3.13-slim`）内是否可装 → CI 已定为 `windows-latest`，ubuntu job 若添加须 `continue-on-error`。
7. `pilot/` 目录当前不在 pytest 与 ruff 的收敛范围内（`testpaths=["tests"]`），其产出的可复现性未纳入门禁。
