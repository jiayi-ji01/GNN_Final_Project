# 实验报告（中文版）

## 1. 研究目标与项目背景

本项目研究的核心问题是：当推断模型与真实数据生成机制存在 misspecification 时，amortized Bayesian posterior 是否仍然可靠。这里的“可靠”不仅指 posterior predictive 是否能画出“看起来合理”的曲线，还包括：

- 参数后验是否仍能正确恢复真实参数；
- 后验置信区间是否具有合理的 empirical coverage；
- 在曲线层或特征层看起来拟合良好的情况下，参数层 posterior 是否可能已经失真。

围绕这一问题，项目构建了四组实验：

- E0：matched baseline  
  用作正常基线，truth 与 inference 都是 SIR。
- E1：structural misspecification  
  truth 是 SEIR，但 inference 仍然是 SIR。
- E2：dynamical misspecification  
  truth 中传播率 beta 随时间变化，但 inference 仍然假设常数 beta 的 SIR。
- E3：observation-process misspecification  
  latent dynamics 仍是 SIR，但观测层加入时间变化的 reporting process。

项目统一采用 matched SIR-trained BayesFlow amortizer 作为推断器，以保证不同实验之间的比较是公平的。这样设计的好处是：所有实验共用同一套训练好的 posterior amortizer，区别只来自 truth model 与 observation process 的变化，因此更容易将 posterior failure 归因于 misspecification 本身。

---

## 2. 实验评价框架

为了避免只看某一种指标，本项目使用四个层次的评价：

### 2.1 Parameter recovery

比较 posterior median 与真实参数（或 pseudo-truth/reference truth）之间的偏差、相关性、MAE、RMSE。

### 2.2 Empirical coverage

比较 nominal 50% / 90% credible interval 与 empirical coverage 的差异，重点判断 posterior 是否 under-covered 或 over-confident。

### 2.3 Feature-level PPC

对 epidemic curve 的关键 summary feature 做 posterior predictive check，包括：

- peak time
- peak height
- cumulative infected
- early growth slope

并在扩展版本中加入更敏感的 shape / tail 指标。

### 2.4 Curve-level PPC

对整条曲线做 pointwise coverage、curve RMSE、分段误差分析，用于回答“即使参数 posterior 已经失真，曲线层看起来是否仍然合理”。

这一套评价框架贯穿 E0-E3 四个实验，是本项目设计最关键的部分。项目最终要回答的不是“哪个模型拟合得最好”，而是“predictive adequacy 与 posterior reliability 是否一致”。

---

## 3. E0：Matched Baseline

### 3.1 实验设定

E0 是匹配条件下的基线实验：

- truth model：SIR
- inference model：SIR-trained BayesFlow amortizer

它的任务是建立一个对照标准，回答“在模型设定正确时，当前 workflow 是否能给出可信 posterior”。

### 3.2 结果概述

E0 的 parameter recovery 表现较好：

- `beta` 的 median bias 为 `-0.0047`，相关系数为 `0.9702`
- `gamma` 的 median bias 为 `0.0071`，相关系数为 `0.8711`
- `R0` 的 median bias 为 `-0.0313`，相关系数为 `0.9963`

coverage 也基本合理：

- `beta` 的 empirical 90% coverage 为 `0.94`
- `gamma` 的 empirical 90% coverage 为 `0.90`
- `R0` 的 empirical 90% coverage 为 `0.97`

说明在 matched setting 下，这套 workflow 至少在参数恢复和 uncertainty calibration 上是站得住的。

在 feature-level PPC 上，E0 也表现良好：

- `peak_time` empirical 90% coverage 为 `0.92`
- `peak_height` empirical 90% coverage 为 `0.95`
- `cumulative_infected` empirical 90% coverage 为 `1.00`
- `early_growth_slope` empirical 90% coverage 为 `0.92`

整体来看，E0 建立了一个重要事实：如果模型设定正确，BayesFlow posterior 可以同时在参数层和 predictive 层表现稳定。

### 3.3 图片分析

主图：[E0_matched_overall_baseline_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E0/E0_matched_overall_baseline_figure.png)

这张图展示了 E0 的“理想状态”：

- recovery scatter 基本围绕 `y=x` 分布，说明 posterior median 与真值一致性较高；
- coverage bar 与 nominal 50% / 90% 参考线接近，说明 posterior interval 没有明显塌缩；
- feature-level PPC 也保持较高覆盖率，说明参数层与 predictive 层结论是相互一致的。

辅助图：

- [E0_matched_parameter_coverage_with_CI.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E0/E0_matched_parameter_coverage_with_CI.png)
- [E0_gamma_diagnostics_by_true_gamma.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E0/E0_gamma_diagnostics_by_true_gamma.png)
- [E0_gamma_diagnostics_by_latent_peak_height.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E0/E0_gamma_diagnostics_by_latent_peak_height.png)

这些图说明即使在 matched setting 下，不同 truth region 仍会带来轻微难度差异，但整体不会改变“posterior 基本可靠”的主结论。

### 3.4 E0 小结

E0 的意义不是单独证明模型很强，而是作为后续所有 misspecification 实验的对照基线。后面如果出现 coverage 崩塌、bias 扩大或参数层与 predictive 层脱钩，就可以明确判断这不是 workflow 本身的基础失效，而是 misspecification 引起的系统性问题。

---

## 4. E1：Structural Misspecification

### 4.1 实验设定

E1 研究的是结构失配：

- truth model：SEIR
- inference model：SIR-trained BayesFlow amortizer

也就是说，真实传播过程包含 exposed compartment，但 inference model 并不知道这部分 latent stage。为了让 recovery / coverage 有可比性，E1 使用 pseudo-true SIR reference，把 SEIR 真值轨迹投影到 SIR 参数空间。

这组实验本质上是在问：当模型缺少一个关键 compartment 时，posterior failure 会如何表现。

### 4.2 结果概述

E1 的 parameter recovery 已经出现明显恶化：

- `beta` median bias 为 `0.0216`
- `gamma` median bias 为 `0.0390`
- `R0` median bias 接近 `0.0000`

单看 bias 可能会觉得“问题不算特别大”，但 coverage 已经明显下降：

- `beta` empirical 90% coverage 只有 `0.79`
- `gamma` empirical 90% coverage 为 `0.85`
- `R0` empirical 90% coverage 仍有 `0.95`

这意味着结构失配首先破坏的是 posterior reliability，而不是所有参数都同时崩掉。特别是 `beta` 和 `gamma`，posterior 已经表现出 under-coverage。

但 predictive 层没有同步崩塌：

- `peak_time` empirical 90% coverage 为 `0.95`
- `peak_height` empirical 90% coverage 为 `0.91`
- `cumulative_infected` empirical 90% coverage 为 `0.97`
- `early_growth_slope` empirical 90% coverage 为 `0.85`

curve-level PPC 也仍然相对正常：

- pointwise empirical 90% coverage 均值为 `0.9414`
- full-curve RMSE 均值仅为 `0.0214`

这正是 E1 的核心发现：  
即使参数 posterior 已经开始偏、开始 under-cover，posterior predictive curves 仍然可能“看起来合理”。

### 4.3 图片分析

主图：[E1_summary_4panel.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1/E1_summary_4panel.png)

这张 4-panel summary figure 是 E1 最重要的一张图，逻辑非常清楚：

- Panel A 和 Panel B 说明 `beta`、`gamma` 的 posterior median 不再紧贴 `y=x`，尤其 `gamma` 的散点偏离更明显；
- Panel C 显示参数层 coverage 明显下降，尤其 `beta` 和 `gamma` 的 90% 区间不再达到 nominal 0.9；
- Panel D 则表明 posterior predictive curves 与 observed curve 仍可较好重合。

这张图直接支持项目核心论点：  
`predictive adequacy != posterior reliability`。

辅助图：

- [E1_structural_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1/E1_structural_overall_figure.png)
- [E1_example_case.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1/E1_example_case.png)
- [E1_curve_level_ppc_structural.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1/curve_level/E1_curve_level_ppc_structural.png)

这些图一起说明：

- latent exposed stage 会让 epidemic curve 在时序上产生延迟和形状变化；
- SIR inference 会尝试用 `beta/gamma` 的组合去“解释”这个额外结构；
- 结果是在参数层吸收了 misspecification，但在数据空间里仍能凑出看似合理的预测曲线。

### 4.4 E1 severity 扩展

severity summary：[E1_severity_3panel_summary.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1/severity_summary/E1_severity_3panel_summary.png)

E1 进一步加入 mild / medium / strong 三档结构失配强度，用 latent period 的强弱控制 SEIR 与 SIR 的偏离程度。这个扩展的意义在于：不只是说明“结构失配会出问题”，还要说明“失配越强，问题是否越严重”。

### 4.5 E1 小结

E1 是本项目论点最清晰的一组实验。它说明：

- 结构失配会优先损害参数后验的可靠性；
- posterior predictive curve 仍可能在视觉上“足够好”；
- 因此只做 PPC 不足以判断后验是否可信。

---

## 5. E2：Dynamical Misspecification

### 5.1 实验设定

E2 研究动力学失配：

- truth model：SIR with time-varying `beta(t)`
- inference model：constant-beta SIR-trained BayesFlow amortizer

其中 severity 用 `kappa` 控制变点之后传播率变化的强度。项目同时使用两种 reference truth 定义：

- `time_avg`
- `infection_weighted`

用于检验结论是否依赖 reference 选择。

### 5.2 结果概述

E2 的 severity 结果呈现出非常清晰的退化趋势。

在 mild / medium / strong 下，`beta` median bias 分别为：

- mild：`0.0298`
- medium：`0.1049`
- strong：`0.2093`

与此同时，`beta` empirical 90% coverage 明显下降：

- mild：`0.78`
- medium：`0.31`
- strong：`0.23`

`R0` empirical 90% coverage 也同步恶化：

- mild：`0.82`
- medium：`0.36`
- strong：`0.19`

`gamma` 的 coverage 虽然也下降，但退化幅度较缓：

- mild：`0.96`
- medium：`0.78`
- strong：`0.73`

这说明 E2 的 misspecification 主要优先破坏与 transmission intensity 更直接相关的参数，即 `beta` 和 `R0`。

此外，feature-level predictive 指标没有像参数层那样同步崩塌。比如 `peak_time_MAE` 在三档 severity 下分别约为：

- mild：`5.195`
- medium：`5.365`
- strong：`5.235`

这提示一个重要事实：  
动力学失配造成的 posterior failure 非常明显，但部分 summary-level predictive 指标并不会以同样速度恶化。

### 5.3 图片分析

示例图：[E2_example_case.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/E2_example_case.png)

该图展示了 E2 的机制来源：真实过程中的 `beta(t)` 在变点前后发生变化，而 inference model 只能用一个常数 beta 去拟合。图像层面上，真实曲线通常并不会立刻显得“完全不合理”，但参数层解释已经发生系统偏移。

主图：

- [E2_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/mild/E2_overall_figure.png)
- [E2_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/medium/E2_overall_figure.png)
- [E2_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/strong/E2_overall_figure.png)

这些图的主要信息是：

- mild 时 recovery 还能维持；
- medium 时 `beta` 与 `R0` 的 coverage 开始显著塌缩；
- strong 时 recovery scatter 与 coverage bar 已经明显偏离 matched baseline。

severity 主图：[E2_severity_3panel_summary.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/severity_summary/E2_severity_3panel_summary.png)

这张图非常有说服力，因为它展示了一个单调而清晰的趋势：

- 随着 severity 增强，`beta` bias 持续增大；
- `beta` coverage 与 `R0` coverage 同时下降；
- posterior failure 与 misspecification 强度之间存在很稳定的对应关系。

reference sensitivity 图：[E2_reference_sensitivity_3panel.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2/reference_sensitivity/E2_reference_sensitivity_3panel.png)

该图说明 reference 定义会影响数值幅度，但不会改变总体趋势：

- `infection_weighted` reference 下结果相对缓和；
- `time_avg` reference 下 medium / strong 的 posterior failure 更明显；
- 这说明 E2 的结论不依赖单一 reference 定义，而是具有一定稳健性。

### 5.4 E2 小结

E2 说明了另一类非常重要的 posterior failure 机制：  
即使模型的 compartment 结构没错，只要关键动力学参数是 time-varying 而 inference 假设它是 constant，posterior 就会对 transmission parameters 产生显著系统误判。

---

## 6. E3：Observation-Process Misspecification

### 6.1 实验设定

E3 研究观测过程失配：

- latent dynamics：SIR
- observation process：引入 time-varying reporting factor `rho(t)`
- inference model：仍然假设标准 SIR 观测

severity 通过 `eta` 控制 reporting factor 在变点后的下降幅度。和 E2 不同，E3 的 misspecification 不在 latent dynamics 本身，而在“观测到的 infected fraction 已经不是 latent infected 的直接反映”。

### 6.2 结果概述

E3 呈现出与 E2 不同但同样重要的图景。

从 severity summary 看：

- `beta` bias 相对不算特别大：
  - mild：`0.0030`
  - medium：`0.0149`
  - strong：`0.0339`
- `gamma` bias 稍大一些：
  - mild：`0.0079`
  - medium：`0.0190`
  - strong：`0.0370`
- 但 `R0` bias 明显变大且为负：
  - mild：`-0.1145`
  - medium：`-0.2584`
  - strong：`-0.4671`

coverage 也随 severity 明显下降：

- `beta` empirical 90% coverage：`0.96 -> 0.86 -> 0.66`
- `gamma` empirical 90% coverage：`0.83 -> 0.57 -> 0.48`
- `R0` empirical 90% coverage：`0.88 -> 0.68 -> 0.57`

与 E2 相比，E3 的一个特点是：

- 参数层已经明显失真，尤其 `R0`；
- 但若只看部分 predictive feature，结论不一定显得那么糟。

例如：

- `peak_height` empirical 90% coverage：`0.97 / 0.89 / 0.78`
- `cumulative_infected` empirical 90% coverage：`1.00 / 1.00 / 0.97`
- `early_growth_slope` empirical 90% coverage：`0.95 / 0.94 / 0.95`

也就是说，即使 reporting process 已经扭曲了参数解释，某些 summary-level predictive checks 仍可能保持“挺好看”的数值。

### 6.3 图片分析

示例图：[E3_example_case.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/E3_example_case.png)

这张图很适合解释 E3 的机制：

- 左图展示 `rho(t)` 的变化；
- 中图展示 latent infected 与 `rho(t) * I(t)` 之间的差距；
- 右图展示最终 observed curve。

这说明 E3 里的 misspecification 并不是 epidemic dynamics 变了，而是“观测层把 latent curve 扭曲了”。

受控 severity 图：[E3_controlled_severity_grid.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/E3_controlled_severity_grid.png)

这张图把 mild / medium / strong 的 reporting distortion 摆在同一网格中，非常有助于读者直观看到 severity 如何在观测层逐步放大。

overall figure：

- [E3_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/mild/E3_overall_figure.png)
- [E3_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/medium/E3_overall_figure.png)
- [E3_overall_figure.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/strong/E3_overall_figure.png)

这些图通常会显示：

- mild 时参数层仍较稳定；
- medium 后 coverage 开始明显下降；
- strong 时 `gamma` 与 `R0` 的 posterior reliability 已较差。

case-level analysis 图：

- [E3_eta_vs_R0_err_median.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/caselevel_analysis/E3_eta_vs_R0_err_median.png)
- [E3_eta_vs_gamma_err_median.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/caselevel_analysis/E3_eta_vs_gamma_err_median.png)
- [E3_tau_vs_R0_err_median.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/caselevel_analysis/E3_tau_vs_R0_err_median.png)

结合相关系数结果可见：

- pooled 层面 `eta` 与 `R0_err_median` 的相关系数为 `0.2600`
- `eta` 与 `R0_covered90` 的相关系数为 `0.3138`
- `tau` 与 `R0_covered90` 的相关系数为 `0.3089`

这些结果说明 posterior failure 的严重程度并不是均匀分布的，而会受到 reporting distortion 强度和 change point 位置共同影响。

truth-region 图：

- [E3_true_gamma_group_panels.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/truth_region_analysis/E3_true_gamma_group_panels.png)
- [E3_true_R0_group_panels.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/truth_region_analysis/E3_true_R0_group_panels.png)
- [E3_true_beta_group_panels.png](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3/truth_region_analysis/E3_true_beta_group_panels.png)

这些图进一步说明：  
E3 的 misspecification 不是对所有 truth region 一视同仁，而是会在某些参数区域放大误差和 under-coverage。

### 6.4 E3 小结

E3 展示了一种更隐蔽的 posterior failure：  
latent dynamics 看起来没被改动，真正被扭曲的是 observation process。结果是：

- 参数解释会明显偏离，尤其 `R0`；
- 但部分 predictive features 仍然可能保持较好表现；
- 因而如果只做表面拟合检验，很容易低估 posterior 失真的程度。

---

## 7. 四个实验的综合比较

把 E0-E3 连起来看，可以得到一个非常清楚的研究结论链条。

### 7.1 从 E0 到 E3 的主线

- E0 证明：在 matched setting 下，这套 workflow 可以给出比较可靠的 posterior。
- E1 证明：结构失配会让 posterior reliability 先于 predictive adequacy 失效。
- E2 证明：动力学失配会最先破坏 transmission-related parameters 的 recovery 与 coverage。
- E3 证明：观测过程失配可以在不显著破坏某些 predictive summaries 的情况下，系统性扭曲参数 posterior。

### 7.2 最核心的科学结论

四组实验共同支持如下判断：

1. posterior predictive 看起来合理，并不意味着 posterior 参数可靠；
2. misspecification 往往优先体现为 bias 和 under-coverage，而不是显著的 predictive breakdown；
3. 若只看 curve fit 或 PPC，很可能低估 posterior failure；
4. recovery、coverage、feature-level PPC、curve-level PPC 必须联合评估。

换言之，本项目最终不是在证明“BayesFlow 好不好”，而是在证明：  
**在 misspecified epidemic inference 中，predictive adequacy 不能替代 posterior reliability diagnosis。**

---

## 8. 项目当前完成情况与不足

### 8.1 已完成部分

- 四个实验的核心实验框架已经建立；
- E0 / E1 / E2 / E3 均已有主结果图与 csv 输出；
- severity 分析在 E1 / E2 / E3 中已经引入；
- E2 reference sensitivity 与 E3 case-level / truth-region analysis 已完成；
- E1 已补充 summary 主图，适合报告和论文使用。

### 8.2 当前不足

目前的主要工作重点集中在四个主实验结果的统一整理与表达。

这不影响四个主实验的整体研究逻辑，但会影响某个补充对照实验的完整复现。

---

## 9. 结论

本项目通过 E0-E3 四组实验系统展示了 epidemic misspecification 下 posterior failure 的不同形态：

- 在 matched baseline 下，posterior 与 predictive 都表现正常；
- 在 structural、dynamical、observation-process misspecification 下，posterior reliability 会以不同方式退化；
- 但 predictive adequacy 往往不会同步失效；
- 因此 posterior 诊断不能只依赖 predictive 检查，而必须引入 recovery 与 empirical coverage 分析。

从课程项目或论文写作的角度看，这套实验体系已经形成了较完整的叙事闭环，既有统一方法设计，也有清晰的对照链条和可视化支撑，已经具备较强的实验报告与论文初稿基础。
