# PRT/HRT 近场单用户与多用户定位

本项目使用均匀线阵接收宽带高斯脉冲，通过 **匹配滤波 → 抛物 Radon 变换（PRT）→ 自适应搜索区域（ROI）→ 双曲 Radon 变换（HRT）** 估计用户位置。单用户基础实验直接使用HRT结果；单用户细化实验复用多用户算法中的精确信号模型搜索、联合Powell优化与逐次干扰消除（SIC）。多用户支持未知人数、最多3个用户。

本文按当前代码说明文件用途、运行方式与输出。实验顶部的常量决定实际设置；修改参数后，应以运行时使用的代码为准。文中的输出说明不代表预先承诺具体数值结果。

## 1. 运行环境与入口

运行环境需要 Python、NumPy、SciPy、Matplotlib、tqdm，以及与本机 CUDA 环境兼容的 CuPy。当前 `transforms/prt.py`、`transforms/hrt.py` 使用 NVIDIA GPU，没有实现自动 CPU 回退。使用已配置好这些依赖的 Python 环境运行。

在项目根目录执行：

```bash
# 单用户 PRT → ROI → HRT
python experiments/single_user.py

# 单用户随机场景，复用完整多用户定位算法
python experiments/single_user_refine.py

# 1～3 用户定位误差主实验
python experiments/multi_user_final.py

# 固定场景的多用户 SIC 迭代演示
python experiments/multi_user_SIC.py
```

`experiments` 内的入口脚本会将项目根目录加入模块搜索路径。图片使用 Matplotlib 窗口展示，需要可显示图形的运行环境。主实验运行期间只展示进度条，结束后打印误差表格、导出一个 CSV 并展示图片。

## 2. 坐标、信号与共享配置

阵列沿 x 轴排列，以原点为中心，`theta=0°` 指向 +y 方向：

```text
x = R × sin(theta)
y = R × cos(theta)
d_m = sqrt(R² + x_m² - 2 R x_m sin(theta))
tau = R/c + b
t_m = tau + (d_m - R)/c
```

`R` 为到阵列中心的距离，`b` 为用户时钟偏移/等效发送延迟，`tau` 为阵列中心参考到达时间。距离和到达时间可独立设置，通过 `b=tau-R/c` 重建信号。位置是二维位置，优化的三个参数是 `(R, theta, tau)`。

每个阵元的接收信号包含高斯包络、载波传播相位和自由空间路径损耗。多用户先叠加复数信号，再统一添加一次复高斯白噪声；每个用户使用相同的默认发射功率，不按用户数归一化总功率。

当前 `config.py`：

| 参数 | 当前值 |
|---|---:|
| 光速 `C` | 299792458 m/s |
| 载波频率 | 24 GHz |
| 带宽 | 8 GHz |
| 过采样倍数 | 8 |
| 采样率 | 64 GHz |
| 采样间隔 `SYSTEM.dt` | 15.625 ps |
| 高斯脉冲 sigma | 约 28.1 ps |
| 阵元数量 | 513 |
| 阵元间距 | 半波长 |
| 热噪声温度 | 300 K |
| 参考距离、参考 SNR | 300 m、4 dB |
| 全局 p 网格点数 | 500 |
| q 网格步长 `PRT.dq` | 1e-13 s/m² |

发射功率由参考距离处的 SNR 确定；噪声功率为 `kTB`。每个用户的距离变化会改变接收 SNR，实验打印的参考距离换算值为：

```text
SNR(R) = SIM.reference_snr_db + 20 log10(SIM.reference_range / R)
```

`PRT.theta_step_deg=0.1°` 供部分辅助实验使用。主流水线采用等间隔 p 网格，并不是固定 0.1° 的角度网格。

## 3. 单用户定位流程

```text
生成精确球面波接收信号 + AWGN
                  ↓
           各阵元匹配滤波
                  ↓
      PRT 搜索 (p, q, tau) 峰值
                  ↓
       转换为粗估计 (R, theta, tau)
                  ↓
         根据粗估计构造自适应 ROI
                  ↓
        精确球面波 HRT 网格搜索
                  ↓
          (R_hat, theta_hat, tau_hat)
                  ↓
              二维位置 (x_hat, y_hat)

```

### PRT、ROI 与 HRT

PRT 用二阶近似描述阵列波前：

```text
t(x) ≈ tau + p x + q x²
p = -sin(theta)/c
q = cos²(theta)/(2Rc)
R = [1 - (pc)²]/(2qc)
```

`localization/pipeline.py` 对 p、q 候选进行有效距离掩码，支持范围为 50～400 m、角度 −60°～60°。

`localization/roi.py` 根据 q 不确定度和角度不确定度设置距离半宽。主流水线通过 `p_hat ± 2DP` 计算角度区间，使用 `K_Q=4.5`；tau 半宽为 `2 × SYSTEM.dt`。ROI 限制在支持的距离和角度范围内。

HRT 对 ROI 使用精确球面波轨迹及相位补偿，取响应幅度最大的点。当前主流水线网格：

| 维度 | 步长 |
|---|---:|
| R | 1 m |
| theta | 0.01° |
| tau | dt/4 = 3.90625 ps |

### 真值辅助搜索

单用户实验沿用已验证的 theta、tau 真值辅助方法来缩小搜索范围、加速计算：PRT 使用真实角度附近的 4 个 p 网格点，并在真实 tau 附近的 4 个时间样点选峰，q 仍在全局网格上搜索。PRT 变换本身仍计算完整输入时间轴。

调用接口：

```python
from localization.pipeline import localize

# 实验采用的辅助搜索
result = localize(rx, time_axis, antenna_x, theta_hint, tau_hint)

# 不提供提示：全局 PRT 搜索
result = localize(rx, time_axis, antenna_x)
```

两个提示必须一起提供。返回值包括最终 `R, theta, tau, x, y`，PRT 粗估计、ROI 和 PRT/HRT 峰值。

### 两个单用户入口的区别

| 文件 | 当前实验 | 运行结果 |
|---|---|---|
| [`experiments/single_user.py`](experiments/single_user.py) | 2000 次随机位置；R 为 50～400 m，theta 为 −60°～60°，时钟偏移 10 ns；直接采用 HRT 结果 | 进度条；发射功率、SNR 范围、ALE、RMSE、最大误差；误差 CDF 与真实位置按误差着色的散点图。不自动保存文件 |
| [`experiments/single_user_refine.py`](experiments/single_user_refine.py) | 当前500次；保留原随机位置、10 ns时钟偏移及噪声生成，定位直接复用 `multi_user_final.run_sic()` | 打印误差统计、有效误差样本数、采样步幅与SNR；展示CDF和空间误差图。不自动保存文件 |

单用户细化入口直接复用第4节的多用户定位算法：候选搜索、Powell边界、迭代次数和SIC停止条件均由 `multi_user_final.py` 提供，不另设单用户优化参数。保留原单用户随机场景；提供真实theta/tau用于辅助搜索，但不强制检测器在输出1人后停止。只有输出人数为1的试验进入误差统计和图片。

两份入口的定位算法与优化参数相同，但场景生成保持独立：single_user_refine.py 使用自己的500次随机位置和噪声序列，固定时钟偏移10 ns；多用户主实验使用独立tau抽样和自己的随机种子。因此两者不保证生成相同数据或得到完全相同的误差统计。单用户细化现在也包含全距离候选扫描、多候选优化和后续残差检测，耗时会高于旧版仅对HRT做一次局部Powell的实现。

## 4. 多用户定位流程

多用户实现复用共享的 PRT 网格、ROI 和 HRT，同时在实验文件内实现候选搜索、联合优化与 SIC。

```text
多个用户的信号叠加 + 一次 AWGN
                    ↓
             当前残差匹配滤波
                    ↓
             辅助 PRT → ROI → HRT
                    ↓
          HRT 峰值不超过门限？→ 停止
                    ↓ 否
      精确信号模型候选搜索（降采样）
                    ↓
       候选 + 已接受用户的联合 Powell 优化
                    ↓
      完整数据残差能量没有下降？→ 停止
                    ↓ 否
        接受候选并更新全部用户的参数
                    ↓
       原始信号减去全部用户的重建信号
                    ↓
       下一轮，最多接受 3 个用户
```

真实人数用于生成场景及统计，不用于决定 SIC 在第几轮停止。theta、tau 真值提示沿用已验证的加速方式。`multi_user_final.py` 的候选扫描包含每个用户的 tau 提示，包括三用户场景的中间 tau；距离仍在全支持区间搜索。

当前主实验参数：

| 参数 | 值 |
|---|---:|
| HRT 停止门限 | 6e-6 |
| 最大 SIC 次数 | 3 |
| 精确模型距离扫描步长 | 4 m |
| 提示角度偏移 | −0.15°、0°、+0.15° |
| 提示 tau 偏移 | −0.015 ns、0、+0.015 ns |
| 候选数上限 | 2 |
| 优化阵元/时间步幅 | 2 / 2 |
| Powell 范围 | 初值 R ±15 m、theta ±0.25°、tau ±0.035 ns |
| 联合逐用户优化轮数上限 | 3 |
| 单次 Powell 最大迭代数 | 100 |

`multi_user_final.py` 固定额外增益为 1，依靠原始物理信号模型重建。`multi_user_SIC.py` 还提供 `GAIN_MODE="complex"` 以拟合复数增益，默认仍为 `"fixed"`。

## 5. 多用户误差主实验：multi_user_final.py

### 场景生成

当前 `N_MC_BY_USER_COUNT = {1: 600, 2: 300, 3: 200}`：单用户600次、双用户300次、三用户200次，共1100次，顺序随机打乱。修改字典即可分别设置各组次数。总体ALE/RMSE仍按有效用户汇集计算，不按试验或人数组等权平均；全部有效时三组各贡献600个用户误差，共1800个。

每个用户独立生成：

| 参数 | 抽样范围 |
|---|---|
| R | 均匀分布于 50～400 m |
| theta | 均匀分布于 −60°～60° |
| tau | 均匀分布于 1500～1502 ns |

任意一对用户至少满足以下一个条件；否则重新抽取整个场景：

```text
|R_i - R_j| >= 100 m
       或
|theta_i - theta_j| >= 1°
       或
|tau_i - tau_j| >= 125 ps
```

100 m、1°、125 ps 是本项目采用的实验分离尺度。这里是“或”，不是三个条件同时满足；这些尺度与 PRT/HRT 的网格步长含义不同。

### 位置匹配和统计

人数一致的试验通过匈牙利算法，最小化真实位置与估计位置的总二维距离，完成一一匹配。误差按真实用户编号排列：

```text
e_i = hypot(x_hat_i - x_i, y_hat_i - y_i)
ALE = mean(e_i)
RMSE = sqrt(mean(e_i²))
P95 = percentile(e_i, 95)
Max = max(e_i)
```

当前主实验只关注定位误差，不打印检出率，也不绘制检出率图。代码仍保留人数检查：若某组意外出现少检或多检，整组不进入误差统计、CDF 和扇形图。没有有效误差时显示 NaN。

结束后打印表格：

| True N | Trials | Matched | ALE(m) | RMSE(m) | P95(m) | Max(m) |
|---|---|---|---|---|---|---|
| 1 | 单用户试验数 | 纳入误差的用户数 | … | … | … | … |
| 2 | 双用户试验数 | 纳入误差的用户数 | … | … | … | … |
| 3 | 三用户试验数 | 纳入误差的用户数 | … | … | … | … |
| ALL | 全部试验数 | 全部有效用户数 | … | … | … | … |

ALL 行汇集全部有效用户的原始误差后计算，不直接平均各组 ALE、RMSE 或 P95。

同时展示三张图：

1. 按真实人数分组的 ALE、RMSE 对比图。
2. 1、2、3 用户组及总体的定位误差 CDF。
3. 50～400 m、−60°～60°扇形区域内的真实位置散点图；颜色条为绝对定位误差，单位米。图中仅显示有效组的用户；不显示标题，左上角统计框仅显示总体 ALE、RMSE，不显示样本数或用户数。底部图例为 `User positions`，右侧颜色条为 `Absolute Localization Error (m)`。

图片只展示，不自动保存。没有逐次参数详情打印，也没有压力测试。

### CSV 文件与 Monte Carlo 分组

运行完成后写入：

```text
experiments/multi_user_localization_errors.csv
```

使用 UTF-8 BOM 编码，便于表格软件读取；每次运行覆盖同名文件。通常每行对应一对匹配后的真实用户与估计用户，同一次试验的所有用户连续排列。

| 列名 | 含义 |
|---|---|
| `monte_carlo_id` | 本次运行内全局唯一的试验编号；相同编号的行属于同一个场景 |
| `user_count` | 该场景的真实用户数量 |
| `trial_within_user_count` | 在相同用户数量组内的第几次试验 |
| `user_in_trial` | 当前场景内的真实用户编号，从 1 开始 |
| `noise_seed` | 该场景的噪声随机种子 |
| `estimated_user_id` | 与真实用户匹配的估计结果编号，可能与真实用户编号不同 |
| `included_in_error_statistics` | 1 表示纳入误差统计，0 表示未纳入 |
| `true_R_m`, `true_theta_deg`, `true_tau_ns` | 真实距离、角度、参考到达时间 |
| `true_x_m`, `true_y_m` | 真实二维坐标 |
| `estimated_R_m`, `estimated_theta_deg`, `estimated_tau_ns` | 匹配后的估计参数 |
| `estimated_x_m`, `estimated_y_m` | 匹配后的估计二维坐标 |
| `dx_m`, `dy_m` | 估计坐标减真实坐标的有符号误差 |
| `error_2d_m` | 二维欧氏定位误差 |

例如，相同 `monte_carlo_id=4` 的三行，`user_count=3`、`user_in_trial=1/2/3`，表示第 4 次全局试验中的三个用户。若出现人数不一致，CSV 单独保留每个真实用户和估计用户，不强制配对，误差留空，统计标志为 0。

## 6. experiments 中其余文件的用途与输出

以下命令均在根目录运行：`python experiments/文件名.py`。单用户入口和多用户误差主入口已在前文展开。

| 文件 | 实验内容和当前关键设置 | 运行后得到的结果 |
|---|---|---|
| [`multi_user_SIC.py`](experiments/multi_user_SIC.py) | 固定 USERS 场景，多用户共用阵列中心 tau；辅助 PRT、ROI、HRT、全距离候选搜索、联合优化与 SIC。角度/距离分辨实验导入并复用此文件 | 控制台逐轮打印粗估计、HRT峰值、候选、联合优化、能量变化和最终用户参数；无自动图片或结果文件 |
| [`angle_test.py`](experiments/angle_test.py) | 固定三用户，距离/角度/tau 独立设置；当前距离 390.27、200.27、80.27 m，角度均为 −57.862°，tau 为 400、400、300 ns。这里“3D”指三个参数，不是三维位置 | 控制台打印 SIC 过程、真实与估计用户、时钟偏移和残差能量；无自动保存 |
| [`2_user_angle_resolution.py`](experiments/2_user_angle_resolution.py) | 两用户同距离、同 tau，改变角度差；200 次，当前 `DELTA_THETA=0.1°` | 逐次人数、恰好输出2人的比例、人数分布、Wilson 95%区间；保存 `experiments/mc_two_user_resolution.csv`，不绘图 |
| [`2_user_range_resolution.py`](experiments/2_user_range_resolution.py) | 两用户同角度、同 tau，改变距离差；200 次，当前 `DELTA_R=50 m` | 逐次结果、人数恢复比例、人数分布、Wilson 95%区间；保存 `experiments/mc_two_user_range_resolution.csv`，不绘图 |
| [`2_user_tau_resolution.py`](experiments/2_user_tau_resolution.py) | 两用户同位置，改变 tau 差；200 次，当前 `15.625*8=125 ps`；文件内实现独立的辅助搜索、联合优化及 SIC | 进度条、恰好输出2人的次数/比例、输出0～3人的次数；不自动保存或绘图 |
| [`2_user_peaks.py`](experiments/2_user_peaks.py) | 50.2 m 与 300.2 m，两用户角度为0°、同 tau；默认无噪声，比较复数相干叠加 | 打印各自与叠加信号的 PRT 距离峰值，展示三条距离响应及真实距离/叠加峰位置；不自动保存 |
| [`SIC_effect.py`](experiments/SIC_effect.py) | 60 m 与100 m，同角度、同 tau，默认加噪声；精确模型细化首个用户后重建消除 | 打印消除前后峰值、能量及弱用户突出程度；展示三面板响应对比。`SAVE_FIGURE=True` 时保存根目录 `sic_prt_demo.png` |
| [`exp01_prt_range_error.py`](experiments/exp01_prt_range_error.py) | PRT 单独测距；距离100～400 m、步长25 m，角度−50°～0°、步长10°，人为设置角度和 tau 不落网格 | 打印各位置误差、p/q估计；展示距离误差曲线、热力图与按角度汇总的均值/最大误差曲线；不自动保存 |
| [`q_reso.py`](experiments/q_reso.py) | 固定theta=−53.5°，距离200～400 m、步长20 m；比较多种dq，对同一位置使用同一份含噪数据 | 打印估计误差，展示不同dq下的距离误差对比曲线；不自动保存 |
| [`find_roi.py`](experiments/find_roi.py) | 随机1000个位置，距离50～1000 m，每个位置5次噪声；统计 `Kq=abs(q_hat-q_true)/dq` | 打印Kq分位数、建议有效dq尺度；展示Kq与距离/角度关系和距离误差图，帮助选ROI宽度；不自动保存 |
| [`roi_test.py`](experiments/roi_test.py) | 500个随机位置，距离50～1000 m，K=6；使用脚本内距离半宽公式，检验区间覆盖真实距离。不是完整三维ROI/HRT测试 | 打印距离覆盖率、失败数、安全余量；展示覆盖位置、余量随距离/角度变化、半宽与实际误差对比；不自动保存 |
| [`hrt_tau_reso.py`](experiments/hrt_tau_reso.py) | 独立HRT测试，不经过PRT/ROI；固定R=200.37 m、theta=30.03°，100次噪声，比较dt的1、1/2、1/4、1/8、1/16时间步长 | 打印不同tau网格下的测距统计，展示比较图；不自动保存 |
| [`time_comparison.py`](experiments/time_comparison.py) | 无theta/tau提示；比较全局HRT与全局PRT→ROI→HRT。当前1次，包含GPU预热、同步计时及分块搜索 | 打印各方法误差、总/平均/中位耗时、时间比和ROI覆盖；分别展示CDF与空间误差图；不自动保存 |

分辨实验的当前角度差0.1°、距离差50 m是可调测试值，与主实验采用的1°、100 m分离条件分开设置。三个分辨脚本一次只测试一个间隔，不会自动扫描整条曲线。tau网格步长研究与双用户tau间隔研究也是两种不同实验。

## 7. 核心代码文件说明

以下模块主要被实验导入；需要单独检查时，从项目根目录用 `python -m 模块路径`，例如 `python -m signal_model.propagation`，避免直接运行子目录文件时出现根模块导入问题。

| 文件 | 作用 | 单独运行结果 |
|---|---|---|
| [`config.py`](config.py) | 不可变配置类：载波/采样、阵列位置、PRT步长、噪声、参考SNR和发射功率 | 只定义配置，无终端输出 |
| [`signal_model/gaussian_source.py`](signal_model/gaussian_source.py) | 高斯复基带脉冲、参考脉冲及局部时间轴 | 打印参考脉冲样点数、采样率、sigma与时间跨度 |
| [`signal_model/propagation.py`](signal_model/propagation.py) | 坐标、精确球面距离/时延、含路径损耗及载波相位的接收信号 | 打印示例信号矩阵形状、发射功率、中心时延、阵列时延跨度 |
| [`signal_model/noise.py`](signal_model/noise.py) | 热噪声功率与独立复AWGN，每个复样本方差为kTB | 打印噪声功率和示例矩阵形状 |
| [`signal_model/matched_filter.py`](signal_model/matched_filter.py) | 对各阵元独立卷积归一化高斯匹配滤波器 | 打印滤波前后矩阵形状 |
| [`transforms/prt.py`](transforms/prt.py) | CuPy频域PRT、相位补偿、峰值提取；返回NumPy数组，维度为时间×p×q | 打印GPU模块加载信息与设备名；不运行完整定位 |
| [`transforms/hrt.py`](transforms/hrt.py) | CuPy精确球面波HRT、插值/相位补偿、分批候选计算、峰值提取；还提供tau/时钟偏移换算 | 打印CUDA设备和可用/总显存；不运行完整定位 |
| [`localization/parameter_mapping.py`](localization/parameter_mapping.py) | p与角度、pq与距离的双向映射 | 打印示例参数往返转换值 |
| [`localization/roi.py`](localization/roi.py) | ROI数据类、由q/角度误差推导距离半宽、p区间转换角度范围和边界裁剪 | 打印一个示例ROI |
| [`localization/pipeline.py`](localization/pipeline.py) | PRT/HRT网格、峰值选择和 `localize()` 公共接口 | 仅加载定义，无主动实验输出；加载时需要CuPy环境 |
| [`utils/plotting.py`](utils/plotting.py) | q切片、距离误差曲线、距离/角度误差热力图函数 | 无主动输出；调用具体函数才会展示图形 |
| [`README.md`](README.md) | 使用说明、流程、文件与输出清单 | 文档，无可执行实验 |

## 8. 调参位置与结果解释

- `config.py`：共享物理参数、阵元数、参考SNR、随机种子、p点数和dq。
- `localization/pipeline.py`：全局支持范围、辅助搜索点数、HRT步长。
- `localization/roi.py`：ROI半宽系数、p与tau窗口。
- 各实验顶部：试验次数、用户场景、分离条件、优化边界、候选数量和输出路径。
- `multi_user_final.py` 的 `N_MC_BY_USER_COUNT` 分别设置1、2、3用户的试验次数；其他脚本的 `N_MC` 含义由各文件决定。

载波相位随tau变化，因此时间间隔既影响包络分离，也影响相干叠加。对双用户时间间隔的扫描结果不必单调。主实验的ALE是二维位置误差，不等同于距离误差或到达时间误差；CSV中的tau便于进一步分析这些维度。

当前目录未包含旧版README曾提到的 `SIC_2.py`、`hrt_test.py`、DBSCAN实验或峰值标定脚本，因此本文不将它们列为可运行入口。