# Experiments

## Experimental Goal

在 epidemic model 参数进行 posterior inference 中，BayesFlow 在 matched setting 与 misspecified setting 下输出的 posterior 是否可靠，以及这种可靠性会以何种方式退化。

实验主要分为三个部分：

- matched baseline：`SIR truth -> SIR inference`。用于验证在模型设定正确时，BayesFlow 能否恢复合理且 calibrated 的 posterior。
- simulation-gap study：用于分析不同类型的模型不匹配如何影响 posterior fit、posterior uncertainty 和 predictive adequacy。
- focused ablation：用于分析 representation 的选择是否改变 reliability 结论。

## Setup

我们考虑一个 epidemic inverse problem。训练阶段的主推断模型为 SIR。

## Matched Baseline

训练数据和测试数据都由 SIR 模型生成，并使用同一 SIR family 做 posterior inference。

在 baseline 中，我们重点检查：

- posterior 是否围绕真值集中；
- posterior predictive trajectories 是否能重现观测曲线的主要特征；
- posterior uncertainty 是否具有良好的 calibration。

## Misspecification Study

目的是研究 posterior reliability 的退化方式。

### E1: Structural misspecification

真实数据由 SEIR 模型生成，而推断仍使用 SIR 模型。代表的是推断模型忽略了潜伏阶段。目的在于分析当动力学机制本身被简化时，posterior 会出现何种偏差、过度自信或 predictive mismatch。

构造方式是把训练用的 SIR simulator 换成 SEIR simulator 来生成测试数据。也就是：

- 训练时：SIR
- 测试时：SEIR
- inference：仍然使用 SIR-trained BayesFlow posterior amortizer

### E2: Dynamical misspecification

真实数据由具有时变传播率 `beta(t)` 的 SIR 模型生成，而推断仍假设常数 `beta`。模拟的是现实中因干预、行为变化或季节效应导致的 transmission changes。目的是分析 posterior 对时变机制简化的敏感性。

构造方式是：

- 训练时：`beta` 是常数
- 测试时：`beta` 变成时间函数 `beta(t)`

### E3: Observation-process misspecification

真实数据由具有时变报告率 `rho(t)` 的 epidemic process 生成，而推断仍使用常数 `rho` 的观测模型。对应的是 observation model 层面的错配。目的在于研究即便动力学本身近似正确，观测过程的错误建模是否也会显著破坏 posterior calibration 与 predictive fit。

构造方式是：

- 训练时：`rho` 为常数
- 测试时：`rho` 变为 `rho(t)`

或者：

- 训练时：Poisson observation
- 测试时：Negative Binomial observation

## Evaluation Metrics

我们采用两类 diagnostics 来评估 posterior reliability。

### Posterior Predictive Checks (PPC)

从 posterior 中采样参数后重新模拟 epidemic trajectories，并将 posterior predictive 分布与观测曲线进行比较。比较对象包括峰值时间、峰值高度、累计病例数以及早期增长趋势等。

PPC 检验 posterior 是否仍然支持能够再现实测模式的参数区域，因此反映的是 predictive adequacy。

### Calibration via empirical coverage

在真值已知的模拟测试集上，计算 50% 和 90% credible intervals 的经验覆盖率，并与对应的 nominal levels 比较。

我们把 coverage 视为 posterior calibration 的主要操作化指标：

- 若经验覆盖率显著低于 nominal coverage，则说明 posterior uncertainty 存在过度自信和 miscalibration。
- 若显著高于 nominal coverage，则说明 posterior 过于保守。

我没有选择 misspecification paper 中的 calibration 方法（SBC），是因为这里研究的是 posterior 的可靠性如何退化，重点包括三部分：

- posterior 的中心位置是否变差；
- posterior 的不确定性是否变差；
- posterior 的 predictive 含义是否变差。

其中：

- PPC 负责 posterior predictive 还能不能生成像观测那样的数据，也就是 posterior 的 predictive 含义是否变差。
- Coverage 负责 posterior uncertainty 还准不准，也就是 posterior 的不确定性是否变差。
- Posterior recovery 或 posterior plots 可以看出 posterior 中心是不是偏了，所以它负责 posterior 的中心位置是否变差。

但我们也可以进一步使用 misspecification paper 中的 SBC，来回答 posterior 是否校准。

## Focused Ablation

用来分析 BayesFlow design choices 对 reliability 的影响，可以在 E0 与 E1 上进行两个小规模的消融实验。

### Ablation-1: representation choice

- 自己定义一组特征，不把整条时间序列原封不动地交给模型，而是先压缩成一组特征，比如手工提取 `peak time`、`peak height`、`cumulative cases` 等等。这样原始时间序列就变成了一个低维向量 `s(y)`。
- 让神经网络自己从原始时间序列里学一个 representation。

这样可以研究不同 representation 是否会影响 posterior quality，尤其是在 structural misspecification 下是否会改变 diagnostics 的敏感性。
