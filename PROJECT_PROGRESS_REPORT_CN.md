# 项目进度报告（中文版）

## 1. 项目概述

本项目围绕“amortized Bayesian inference 在 epidemic misspecification 场景下的可靠性”展开，核心目标是系统比较：

- 在模型完全匹配时，BayesFlow 后验是否能够正确恢复参数并给出可靠不确定性。
- 在不同类型的 misspecification 下，参数后验、区间覆盖率、posterior predictive checks（PPC）之间是否会出现脱钩。
- predictive adequacy 是否足以说明 posterior reliability，还是需要额外的 calibration / coverage / recovery 诊断。

整个项目以一个统一的 SIR-trained BayesFlow amortizer 为推断器，然后构造四组实验：

- E0：matched baseline  
  truth 与 inference 都是 SIR，用来建立“正常情况下”的对照基线。
- E1：structural misspecification  
  truth 使用 SEIR，inference 仍用 SIR，用来考察遗漏 exposed compartment 的结构性偏差。
- E2：dynamical misspecification  
  truth 使用时间变化的传播率 beta(t)，inference 仍假设常数 beta 的 SIR，用来考察动力学失配。
- E3：observation-process misspecification  
  latent dynamics 仍是 SIR，但观测过程增加时间变化报告率 rho(t)，inference 仍假设标准 SIR 观测，用来考察观测失配。

项目总体设计强调三点：

1. 训练流程统一  
   只训练 matched SIR workflow，然后把它应用到不同失配实验上，这样可以把失配影响和训练策略区分开。
2. 评价维度分层  
   同时做 parameter recovery、empirical coverage、feature-level PPC、curve-level PPC，不只依赖单一指标。
3. 实验结构模块化  
   通过 `src/*.py` 封装 simulator、workflow、diagnostics、experiments、plotting，再由 `scripts/*.py` 和 `notebooks/*.ipynb` 驱动运行与展示。

---

## 2. 项目整体架构设计

项目采用典型的“配置层 - 模拟层 - 推断层 - 诊断层 - 实验编排层 - 可视化层”结构。

### 2.1 配置层

文件：[src/config.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/config.py)

该模块负责统一管理：

- 全局仿真常量  
  包括 `N_DAYS`、`TIME_GRID`、初始状态、观测噪声 `OBS_NOISE`。
- 参数先验范围  
  例如 `PRIOR_BOUNDS["beta"]` 与 `PRIOR_BOUNDS["gamma"]`。
- 各实验 severity 配置  
  E1 通过 `sigma` 控制 latent period 强弱，E2 通过 `kappa` 控制 beta(t) 变化强度，E3 通过 `eta` 控制报告率变化强度。
- 输出目录管理  
  例如 `outputs/E0`、`outputs/E1/severity_summary`、`outputs/E3/truth_region_analysis` 等。
- 环境初始化  
  `setup_environment()` 会设置随机种子、绘图风格、版本检查和部分运行环境变量。

这一层的价值在于把实验超参数、输出组织和环境依赖统一下来，避免 notebook 中出现大量硬编码。

### 2.2 模拟层

文件：[src/simulators.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/simulators.py)

该模块定义了所有实验对应的 truth model 与观测生成逻辑，主要包括：

- SIR 常微分方程与求解  
  `sir_rhs()`、`solve_sir()`、`simulate_curve()` 是 matched baseline 的核心。
- SEIR 常微分方程与求解  
  `seir_rhs()`、`solve_seir()`、`simulate_curve_seir()` 支持 E1 的结构失配实验。
- pseudo-true SIR projection  
  `fit_pseudotrue_sir_reference()` 将 SEIR 轨迹投影到 SIR 参数空间，构造 E1 的 pseudo-truth。
- E1 case simulator  
  `simulate_case_E1()` 支持 severity、reference_mode、early_phase_days 等扩展参数。
- E2 time-varying beta simulator  
  `beta_t_piecewise()`、`solve_sir_timevarying_beta()`、`simulate_case_E2()` 构成 E2 的动力学失配生成机制。
- E3 reporting-process simulator  
  该模块后半部分定义了 `rho_t_piecewise()` 以及 `simulate_case_E3()` 等函数，用时间变化观测比例扭曲 latent infected curve。

这层的设计思路是：  
所有实验都尽量保持“返回结构一致”，即 case 字典里都包含真值参数、实验特有 latent 参数以及 `obs`。这样上层诊断函数可以最大化复用。

### 2.3 推断层

文件：[src/workflow.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/workflow.py)

该模块封装 BayesFlow workflow 的构建、训练和采样逻辑，是整套项目的推断核心。主要功能包括：

- 参数约束变换  
  `to_unconstrained()` / `from_unconstrained()` 把有界参数映射到无界空间，便于 BayesFlow 学习。
- 训练模拟器  
  `make_training_simulator()` 用 matched SIR 生成训练样本。
- workflow 构建  
  `build_workflow()` 用 `bf.BasicWorkflow` 配合 `time_series_network` summary network 和 inference network 进行搭建。
- 训练入口  
  `train_workflow()` 封装训练轮数、batch、随机种子等。
- 后验采样接口  
  `sample_posterior()` 从 workflow 输出后验样本，并自动转回原始参数空间。

这里的核心设计思想是：

- 训练只基于 matched SIR simulator。
- 不同 misspecification 实验不重训新的 truth-specific 模型，而是把同一个 amortizer 用在不同 test distribution 上。
- 这样可以更直接观察 posterior failure 是由模型失配引起，而不是训练样本变化引起。

### 2.4 诊断层

文件：[src/diagnostics.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/diagnostics.py)

该模块统一实现“如何评价一个实验结果”，主要包括四类诊断：

- 后验摘要  
  `summarize_posterior_samples()` 计算 mean、median、50%/90% CI、区间宽度等。
- 参数层评价  
  `evaluate_testset()` 逐 case 计算 recovery、bias、coverage、MAE、RMSE，并汇总成 `case_df / recovery_df / coverage_df`。
- feature-level PPC  
  `extract_curve_features()` 与 `evaluate_feature_level_ppc()` 对峰值时间、峰值高度、累计感染量、早期增长斜率等特征做 posterior predictive evaluation。
- extended feature PPC  
  `extended_extract_curve_features()` 与 `extended_evaluate_feature_level_ppc()` 进一步加入峰后衰减、late-phase AUC 等更敏感形状特征。
- curve-level PPC  
  `evaluate_curve_level_ppc()` 与 `evaluate_curve_level_ppc_tau_aligned()` 对整条曲线做 pointwise coverage、曲线 RMSE 以及 split-specific 诊断。

这一层的设计非常关键，因为项目的核心论点并不是“谁拟合得更好”，而是：

- 参数后验是否可靠；
- 可信区间是否 calibration 失效；
- predictive 检查是否会掩盖 posterior failure。

所以 diagnostics 层本质上是项目科学结论的基础设施。

### 2.5 实验编排层

文件：[src/experiments.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/experiments.py)

该模块负责把 simulator、workflow、diagnostics 组合成可直接调用的实验接口。主要内容包括：

- E0 接口  
  `run_E0_evaluation()`、`run_single_case_posterior()`、`run_e0_seed_sensitivity()` 等。
- E1 接口  
  `make_E1_case_simulator()`、`run_E1_evaluation()`、`run_E1_example_case()`、`run_extended_E1_feature_evaluation()`。
- E1 汇总与报告  
  `build_E1_severity_summary()`、`build_E1_reference_ablation_summary()`、`build_E1_results_report()`。
- E2 接口  
  `run_E2_evaluation()`、`run_extended_E2_feature_evaluation()`、`build_E2_severity_summary()`、`build_E2_reference_sensitivity_summary()` 等。
- E3 接口  
  `run_E3_example_case()`、`build_E3_controlled_cases()`、`run_E3_evaluation()`、`build_E3_severity_summary()`。
- E3 深度分析  
  `build_E3_caselevel_correlation_summary()`、`build_E3_eta_tau_bin_summaries()`、`build_E3_truth_region_tables()`。

这层的设计作用是把“实验逻辑”从底层函数中提炼出来，使 notebook 只保留调用、展示与解释，不再承担复杂计算逻辑。

### 2.6 可视化层

文件：[src/plotting.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/src/plotting.py)

该模块集中管理所有实验图形输出，主要包括：

- 训练与单案例图  
  如 `plot_training_loss()`、`plot_single_case_posterior()`、`plot_single_case_ppc()`。
- E1 图  
  `plot_e1_example_case()`、`plot_e1_overall_figure()`、`plot_E1_severity_3panel_summary()`、`plot_E1_ablation_comparison()`。
- E2 图  
  `plot_e2_example_case()`、`plot_e2_overall_figure()`、`plot_E2_severity_lines()`、`plot_E2_reference_sensitivity()`。
- E3 图  
  `plot_e3_example_case()`、`plot_e3_controlled_grid()`、`plot_e3_overall_figure()`、`plot_E3_severity_lines()`、truth-region 和 case-level panel 图。
- 通用图  
  `plot_truth_vs_median_scatter()`、`plot_feature_residuals()`、`plot_curve_level_ppc_summary()` 等。

项目的可视化设计基本遵循“主图 + severity 图 + 深度分析图”的层次：

- 主图用于表达核心结论；
- severity 图用于表达退化趋势；
- deeper analysis 图用于支撑机制解释。

---

## 3. notebooks 的作用

当前 notebook 结构如下：

- [00_E0_matched_baseline.ipynb](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/notebooks/00_E0_matched_baseline.ipynb)  
  展示 matched baseline，包括训练、参数恢复、coverage、PPC、seed sensitivity 等。
- [01_E1_structural_misspecification.ipynb](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/notebooks/01_E1_structural_misspecification.ipynb)  
  展示 E1 主结果，并已经补充 severity、pseudo-truth ablation、4-panel summary figure。
- [02_E2_dynamical_misspecification.ipynb](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/notebooks/02_E2_dynamical_misspecification.ipynb)  
  展示时间变化 beta(t) 下的 dynamical misspecification。
- [03_E3_observation_process_misspecification.ipynb](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/notebooks/03_E3_observation_process_misspecification.ipynb)  
  展示报告率变化导致的 observation misspecification。

notebook 的设计原则比较清晰：

- notebook 主要做展示和调用；
- 底层函数尽量放在 `src/`；
- 图和 csv 统一保存到 `outputs/`；
- 这样既方便论文写作，也方便后续脚本化复现。

---

## 4. scripts 逐个说明

### 4.1 `scripts/train_e0_workflow.py`

文件：[train_e0_workflow.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/scripts/train_e0_workflow.py)

作用：

- 初始化环境；
- 训练 matched SIR workflow；
- 输出训练 loss 图；
- 作为最小训练入口验证 BayesFlow 训练是否通畅。

它的实现最简单，但作用很基础，相当于整个项目的“workflow smoke test”。

### 4.2 `scripts/run_e0_evaluation.py`

文件：[run_e0_evaluation.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/scripts/run_e0_evaluation.py)

作用：

- 训练 matched workflow；
- 运行 E0 baseline 评价；
- 保存 case-level、recovery、coverage、feature-level PPC 结果；
- 输出 matched baseline 主图。

它的意义在于先建立一个“模型匹配时系统应该长什么样”的对照标准，后续 E1/E2/E3 的所有结论都需要和它比较。

### 4.3 `scripts/run_e1_evaluation.py`

文件：[run_e1_evaluation.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/scripts/run_e1_evaluation.py)

这是当前项目里结构最复杂的脚本之一，承担了 E1 的完整扩展版实验。它做了以下事情：

- 训练 matched SIR workflow（flow posterior）。
- 生成 E1 示例案例图。
- 运行 E1 主实验：
  - case-level posterior recovery
  - coverage
  - feature-level PPC
  - extended feature PPC
  - curve-level PPC
  - 结果报告文本
- 运行 E1 severity 扩展：
  - mild / medium / strong 三档
  - 保存每档结果与 overall figure
  - 汇总 severity summary 与 3-panel 图
- 运行 E1 ablation-1：
  - `full_curve` vs `early_phase` pseudo-truth 定义对照
  - 输出 summary 与 comparison 图

设计上，这个脚本已经不只是“跑 E1 主结果”，而是承担了 E1 章节的整套实验编排工作。

当前状态说明：

- E1 主结果、severity 和 reference ablation 已经有稳定输出产物。

### 4.4 `scripts/run_e2_evaluation.py`

文件：[run_e2_evaluation.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/scripts/run_e2_evaluation.py)

作用：

- 训练 matched workflow；
- 分别运行 mild / medium / strong 三档 E2 实验；
- 输出每档：
  - case-level
  - recovery
  - coverage
  - feature-level PPC
  - extended feature PPC
  - overall figure
- 运行 curve-level PPC：
  - fraction split
  - tau-aligned split
- 构建 severity summary 和 uncertainty-aware summary；
- 运行 reference sensitivity（`time_avg` vs `infection_weighted`）。

这个脚本已经体现出项目的第二层成熟度：不仅有主结果，还开始从“reference definition 是否影响结论”这个角度做稳健性分析。

### 4.5 `scripts/run_e3_evaluation.py`

文件：[run_e3_evaluation.py](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/scripts/run_e3_evaluation.py)

这是项目中分析深度最高的脚本。它包括：

- 训练 matched workflow；
- 生成 E3 示例案例图；
- 生成 controlled severity grid；
- 运行 mild / medium / strong 三档 E3 主实验；
- 运行 fraction split 与 tau-aligned split 的 curve-level PPC；
- 生成 severity summary；
- 拼接全 severity case-level 数据；
- 做 case-level correlation analysis：
  - `eta`、`tau` 与 `R0_err_median` / `gamma_err_median` / `R0_covered90` 的关系
- 做 eta / tau 分箱分析；
- 做 truth-region analysis：
  - 按 `true_beta` / `true_gamma` / `true_R0` 分组
  - 分 severity 构造 pivot 表
  - 输出一系列分组 panel 图

从脚本结构上可以看出，E3 不再只是简单的“再跑一套 misspecification”，而是已经进入“机制解释”和“误差分布异质性”的分析阶段。

---

## 5. 当前各实验完成进度

### 5.1 E0：Matched Baseline

当前状态：基本完成

已完成内容：

- matched SIR workflow 训练
- parameter recovery
- empirical coverage
- feature-level PPC
- baseline overall figure
- 部分不确定性汇总和 stratified gamma diagnostics

当前产物可见于：

- [outputs/E0](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E0)

E0 的作用已经从“跑通基线”升级为“为后续 misspecification 实验提供 calibration 对照标准”。

### 5.2 E1：Structural Misspecification

当前状态：主线基本完成

已完成内容：

- 主实验：SEIR truth vs SIR inference
- example case
- recovery / coverage / feature PPC / extended feature PPC
- curve-level PPC
- 结果自动报告
- severity 扩展：mild / medium / strong
- pseudo-truth definition ablation：
  - `full_curve`
  - `early_phase`
- notebook 4-panel summary 主图

已生成的主要输出位于：

- [outputs/E1](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E1)

换句话说，E1 章节从论文结构上已经相当完整，目前保留的扩展重点是 severity 与 pseudo-truth reference 两条主线。

### 5.3 E2：Dynamical Misspecification

当前状态：完成度高

已完成内容：

- mild / medium / strong severity 版本
- main evaluation
- extended feature PPC
- curve-level PPC（fraction split）
- curve-level PPC（tau-aligned split）
- severity summary
- severity uncertainty summary
- reference sensitivity

主要输出位于：

- [outputs/E2](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E2)

E2 已经具备较完整的“主结果 + 机制解释 + 稳健性分析”结构，是当前项目中成熟度较高的一部分。

### 5.4 E3：Observation-Process Misspecification

当前状态：完成度最高

已完成内容：

- mild / medium / strong severity 主实验
- example case
- controlled severity grid
- overall figures
- curve-level PPC（fraction split）
- curve-level PPC（tau-aligned split）
- severity summary
- case-level correlation analysis
- eta / tau bin summary
- truth-region analysis

主要输出位于：

- [outputs/E3](/Users/jijiayi/Documents/25WS/GNN/GNN_Final_Project/outputs/E3)

E3 的分析已经不只是“证明 misspecification 会造成问题”，而是开始回答：

- 问题在哪些 truth region 最严重；
- 改变量 `eta` 与变点 `tau` 如何影响 recovery 和 coverage；
- 为什么部分 predictive diagnostics 仍可能显得正常。

从研究深度上看，E3 已经具备很强的报告和论文支撑能力。

---

## 6. 当前成果总结

截至目前，项目已经形成一个比较完整的实验研究框架，具有以下特点：

### 6.1 方法层面

- 统一使用 matched SIR-trained BayesFlow amortizer 作为 inference engine。
- 通过 E0/E1/E2/E3 构造“从匹配到不同失配类型”的连续实验链条。
- 在每个实验中同时比较 parameter recovery、coverage、feature PPC、curve PPC。

### 6.2 工程层面

- 核心逻辑已经从 notebook 中抽离到 `src/`。
- 各实验均有独立 `scripts/` 入口。
- 输出目录统一、命名风格一致，便于后续写作与复现。
- 图表与 csv 结果已经系统化保存。

### 6.3 研究层面

项目已经逐步形成一个明确论点：

- misspecification 不一定立刻表现为 predictive failure；
- posterior 参数层的偏差与 under-coverage 可能先于 predictive breakdown 出现；
- 因此 predictive adequacy 不等于 posterior reliability；
- 需要把 recovery、coverage 和 PPC 联合起来看。

这条主线在 E1 尤其清晰，在 E2 和 E3 中则进一步扩展到更复杂的机制与异质性分析。

---

## 7. 当前存在的问题与风险

### 7.1 部分图名和旧输出残留

从输出目录可以看出，某些历史输出或中间版本文件仍然存在，例如少量命名遗留和旧 notebook 导出的图。这不会影响核心实验逻辑，但在最终交付前建议统一清理一次，以保持成果目录整洁。

### 7.2 结果文稿仍需进一步标准化

目前结果表、图和分析接口都已经较完整，但若面向正式论文提交，还需要把：

- 图注风格
- 表格命名
- severity 定义说明
- pseudo-truth 定义说明

进一步标准化，确保全文叙事一致。

---

## 8. 下一步建议

下一阶段建议按以下优先级推进：

### 第一优先级：统一报告材料

- 把 E0/E1/E2/E3 的主图和 summary 表整理成统一风格。
- 为每个实验补齐一段标准化中文结果分析。

### 第二优先级：形成最终论文叙事

建议最终论文结构采用：

1. matched baseline  
2. structural misspecification  
3. dynamical misspecification  
4. observation-process misspecification  
5. predictive adequacy vs posterior reliability 的综合讨论

这条结构与当前项目实现高度一致，后续整理成本较低。

---

## 9. 总结

从当前实现情况看，本项目已经不是“若干零散 notebook 的集合”，而是一个具有较清晰研究问题、统一工程结构和系统输出组织的实验平台。

完成度大致可以概括为：

- E0：完成
- E1：主体完成
- E2：高完成度
- E3：高完成度且分析最深入

如果后续继续统一整理图表与文字，本项目已经具备较强的课程项目/实验报告/论文初稿基础。
