# PRT/HRT Near-Field Localization

A compact research prototype for near-field localization with a uniform linear array (ULA). The project combines a **Parabolic Radon Transform (PRT)** for coarse localization with an adaptive **region of interest (ROI)** and an **Hyperbolic Radon Transform (HRT)** for accurate spherical-wave refinement.

The main processing chain is:

```text
Received array signal
        ↓
Matched filtering
        ↓
PRT coarse localization
        ↓
(R̂PRT, θ̂PRT, τ̂PRT)
        ↓
Adaptive ROI construction
        ↓
HRT refinement inside the ROI
        ↓
(R̂, θ̂, τ̂)
        ↓
Cartesian position (x̂, ŷ)
```

---

## 1. System Configuration

The shared system configuration is defined in `config.py`.

Default parameters:

| Parameter | Value |
|---|---:|
| Carrier frequency | 24 GHz |
| Signal bandwidth | 8 GHz |
| Oversampling factor | 8 |
| Sampling frequency | 64 GHz |
| Sampling interval | 15.625 ps |
| Number of antennas | 513 |
| Antenna spacing | λ / 2 |
| Default SNR | 10 dB |
| PRT angle step | 0.1° |
| PRT q step | 1 × 10⁻¹³ s/m² |

The array is centered at the origin and lies along the x-axis. The angle convention used throughout the project is:

```text
θ = 0°     broadside
θ < 0°     left side of broadside
θ > 0°     right side of broadside
```

The polar-to-Cartesian mapping is

```text
x = R sin(θ)
y = R cos(θ)
```

---

## 2. Signal Model

The signal model is implemented in `signal_model/`.

### Gaussian source

`signal_model/gaussian_source.py` generates the Gaussian pulse used by the simulator and the matched filter.

The baseband pulse is

```text
g(t) = exp[-0.5 (t / σ)²]
```

with

```text
σ = 1 / (√2 π B)
```

where `B` is the signal bandwidth.

### Exact spherical-wave propagation

`signal_model/propagation.py` generates the received signal using the exact spherical-wave geometry.

For a user at range `R` and angle `θ`, the source position is

```text
xₛ = R sin(θ)
yₛ = R cos(θ)
```

For antenna position `xₘ`, the exact propagation distance is

```text
dₘ = √(R² + xₘ² - 2 R xₘ sin(θ))
```

and the received arrival time is

```text
tₘ = dₘ / c + b
```

where `b` is the clock offset.

The simulated complex received signal includes:

- Gaussian pulse delay,
- exact spherical-wave propagation,
- carrier phase,
- 1 / dₘ path loss.

A simplified expression is

```text
rₘ(t) = (A / dₘ) exp(-j 2π f₍c₎ tₘ) g(t - tₘ)
```

### Noise

`signal_model/noise.py` adds complex AWGN. The SNR is defined using the discrete pulse energy of the center antenna.

### Matched filtering

`signal_model/matched_filter.py` applies a Gaussian matched filter independently to each antenna channel before localization.

---

## 3. PRT Coarse Localization

The PRT implementation is in `transforms/prt.py`.

The exact spherical-wave delay is approximated by a second-order model:

```text
t(x) ≈ τ + p x + q x²
```

with

```text
p = -sin(θ) / c
q = cos²(θ) / (2 R c)
```

Therefore,

```text
θ = asin(-p c)
R = [1 - (p c)²] / (2 q c)
```

The conversion between physical parameters `(R, θ)` and PRT parameters `(p, q)` is implemented in:

```text
localization/parameter_mapping.py
```

### PRT implementation

For each candidate `(p, q)`, the relative delay across the array is

```text
Δtₘ = p xₘ + q xₘ²
```

The PRT aligns the antenna signals using FFT-based fractional time shifting and applies carrier-phase compensation before coherent summation.

The transform produces a three-dimensional search volume:

```text
PRT(τ, p, q)
```

The peak is converted back into coarse estimates:

```text
R̂PRT, θ̂PRT, τ̂PRT
```

---

## 4. Global and Hint-Assisted PRT Modes

`localization/pipeline.py` supports two PRT operating modes.

### Global mode

```python
localize(rx, time_axis, antenna_x)
```

The PRT searches the full configured parameter space:

```text
θ : -60° to 60°
R : 40 m to 500 m
τ : full available time axis
q : full global q grid
```

The global angle grid uses a 0.1° step.

### Hint-assisted mode

```python
localize(rx, time_axis, antenna_x, theta_hint, tau_hint)
```

`theta_hint` and `tau_hint` provide approximate prior information to reduce the PRT search space.

- `theta_hint` selects the four nearest PRT angle/p bins.
- `tau_hint` selects the four nearest time samples during PRT peak selection.
- The q dimension still uses the global q grid.

Therefore, the local search is approximately

```text
4 p bins × global q grid × 4 τ bins
```

instead of the complete global `(p, q, τ)` search.

After the PRT estimate is obtained, the ROI and HRT stages do **not** use the hints.

In `experiments/single_user.py`, the current Monte Carlo workflow uses

```python
localize(noisy_rx, time_axis, antenna_x, theta_true, tau_true)
```

so the true angle and true arrival time are supplied as PRT hints for that experiment.

---

## 5. ROI Construction

The ROI is implemented in `localization/roi.py` and is centered on the PRT estimate:

```text
(R̂PRT, θ̂PRT, τ̂PRT)
```

The purpose of the ROI is to restrict the expensive HRT search to a physically meaningful local region.

### Angle ROI

The current angle half-width is

```text
Wθ = 0.2°
```

so

```text
θROI = [θ̂PRT - 0.2°, θ̂PRT + 0.2°]
```

The total angular width is 0.4°.

### τ ROI

The current τ half-width is two signal samples:

```text
Wτ = 2 Δτ
```

With

```text
Δτ = 15.625 ps
```

this gives

```text
Wτ = 31.25 ps
```

and therefore

```text
τROI = [τ̂PRT - 31.25 ps, τ̂PRT + 31.25 ps]
```

### Range ROI

Range is related to `q` and `θ` by

```text
R = cos²(θ) / (2 c q)
```

A first-order error propagation gives the q-dependent range uncertainty

```text
|ΔR_q| ≈ [2 c R² / cos²(θ)] |Δq|
```

and the angle-dependent contribution

```text
|ΔR_θ| ≈ 2 R |tan(θ)| |Δθ|
```

The implementation uses

```text
Kq = 4.5
Δq_eff = Kq Δq
```

with

```text
Δq = 1 × 10⁻¹³ s/m²
```

The range ROI half-width is

```text
WR = [2 c R̂PRT² / cos²(θ̂PRT)] Δq_eff
     + 2 R̂PRT |tan(θ̂PRT)| Δθ_eff
```

where

```text
Δθ_eff = 0.2°
```

converted to radians inside the implementation.

The final range interval is

```text
RROI = [R̂PRT - WR, R̂PRT + WR]
```

and is clipped to the global physical range bounds when required.

This construction naturally produces a larger range ROI when the user is farther away or when `|θ|` becomes larger.

---

## 6. HRT Refinement

The HRT implementation is in `transforms/hrt.py`.

Unlike the PRT, the HRT does not use the second-order parabolic approximation. It uses the exact spherical-wave distance:

```text
dₘ(R, θ) = √(R² + xₘ² - 2 R xₘ sin(θ))
```

The relative spherical-wave delay is

```text
Δtₘ(R, θ) = [dₘ(R, θ) - R] / c
```

and the candidate arrival trajectory is

```text
tₘ = τ + Δtₘ(R, θ)
```

This formulation separates:

- `R` and `θ`, which determine the spatial wavefront shape,
- `τ`, which determines the common arrival-time offset.

This also allows a non-zero clock offset because

```text
τ = R / c + b
```

and therefore

```text
b = τ - R / c
```

The HRT evaluates the matched-filtered signals using time-domain linear interpolation and carrier-phase compensation before coherent summation.

### HRT grid used by the pipeline

The current refinement steps are:

| Dimension | Step |
|---|---:|
| Range | 1 m |
| Angle | 0.005° |
| τ | Δτ / 4 = 3.90625 ps |

The HRT searches only inside the ROI generated from the PRT estimate.

---

## 7. Complete Localization Pipeline

The complete processing chain is implemented in `localization/pipeline.py`.

```text
raw received signal
        ↓
matched_filter(...)
        ↓
run_prt(...)
        ↓
R̂PRT, θ̂PRT, τ̂PRT
        ↓
build_roi(...)
        ↓
run_hrt(...)
        ↓
R̂, θ̂, τ̂
        ↓
polar_to_xy(...)
        ↓
x̂, ŷ
```

The returned `LocalizationResult` contains:

```text
Final estimate:
    R
    θ
    τ
    x
    y

PRT estimate:
    prt_R
    prt_theta
    prt_tau

Diagnostics:
    roi
    prt_peak
    hrt_peak
```

---

## 8. Repository Structure

```text
prt_hrt_localization/
├── config.py
├── README.md
│
├── experiments/
│   ├── exp01_prt_range_error.py
│   ├── exp02_prt_roi_hrt.py
│   ├── find_roi.py
│   ├── hrt_tau_reso.py
│   ├── hrt_test.py
│   ├── q_reso.py
│   ├── roi_test.py
│   └── single_user.py
│
├── localization/
│   ├── parameter_mapping.py
│   ├── pipeline.py
│   └── roi.py
│
├── signal_model/
│   ├── gaussian_source.py
│   ├── matched_filter.py
│   ├── noise.py
│   └── propagation.py
│
├── transforms/
│   ├── hrt.py
│   └── prt.py
│
└── utils/
    └── plotting.py
```

### Core files

| File | Purpose |
|---|---|
| `config.py` | Shared physical, array, sampling, PRT, SNR, and random-seed settings |
| `gaussian_source.py` | Gaussian pulse generation |
| `propagation.py` | Exact spherical-wave signal simulation |
| `noise.py` | Complex AWGN generation |
| `matched_filter.py` | Per-antenna Gaussian matched filtering |
| `parameter_mapping.py` | Conversion between `(R, θ)` and `(p, q)` |
| `prt.py` | Parabolic Radon Transform |
| `roi.py` | Adaptive `(R, θ, τ)` ROI construction |
| `hrt.py` | Exact spherical-wave HRT refinement |
| `pipeline.py` | Complete matched-filter → PRT → ROI → HRT localization pipeline |
| `plotting.py` | General plotting helpers |

`experiments/exp02_prt_roi_hrt.py` is currently an empty placeholder. The implemented PRT → ROI → HRT workflow is contained in `localization/pipeline.py` and exercised by the other experiment scripts.

---

## 9. Experiments

The experiment scripts follow the development and calibration of the complete localization pipeline.

### `exp01_prt_range_error.py`

Studies basic PRT range-estimation behavior over different true ranges and angles.

Typical outputs include:

- range error versus true range,
- range-angle error maps,
- mean and maximum range error.

### `q_reso.py`

Studies the influence of q-grid spacing on PRT range accuracy.

Several values of `Δq` are tested while the true parameters are intentionally off-grid.

The current working value is

```text
Δq = 1 × 10⁻¹³ s/m²
```

which is stored in `config.py` as `PRT.dq`.

### `find_roi.py`

Uses Monte Carlo simulation to characterize the PRT q-estimation error.

The principal normalized quantity is

```text
Kq = |q̂ - q_true| / Δq
```

The resulting statistics are used to select a conservative q-error factor for ROI construction.

The current ROI implementation uses

```text
Kq = 4.5
```

### `roi_test.py`

Validates the range ROI derived from the PRT estimate.

For each trial it checks whether the true range lies inside the predicted interval and reports quantities such as:

- ROI coverage,
- number of failures,
- worst margin,
- mean ROI radius,
- maximum ROI radius.

### `hrt_tau_reso.py`

Studies the HRT τ-grid resolution by testing several subdivisions of the original signal sampling interval.

The pipeline currently uses

```text
Δτ_HRT = Δτ / 4
```

which corresponds to 3.90625 ps for the default configuration.

### `hrt_test.py`

Tests the HRT refinement stage independently by constructing a local search region around known true parameters.

Reported metrics include:

- range error,
- angle error,
- τ error,
- 2-D Euclidean localization error,
- x RMSE,
- y RMSE,
- total 2-D RMSE.

### `single_user.py`

Runs the complete Monte Carlo localization workflow for randomly generated users.

Current user generation:

```text
R ~ Uniform(50 m, 400 m)
θ ~ Uniform(-60°, 60°)
SNR = 10 dB
clock offset = 10 ns
```

Each run performs:

```text
random user generation
        ↓
exact spherical-wave signal simulation
        ↓
AWGN
        ↓
matched filtering
        ↓
PRT
        ↓
ROI
        ↓
HRT
        ↓
2-D localization error
```

The current script uses the true `θ` and true `τ` as PRT hints:

```python
result = localize(
    noisy_rx,
    time_axis,
    antenna_x,
    theta_true,
    tau_true,
)
```

The final 2-D absolute error for user `i` is

```text
eᵢ = √[(x̂ᵢ - xᵢ)² + (ŷᵢ - yᵢ)²]
```

The script reports:

```text
ALE  = mean(eᵢ)
RMSE = √mean(eᵢ²)
```

and produces two main visualizations.

#### Error CDF

The original CDF plot shows the statistical distribution of the final 2-D localization error.

#### Spatial absolute-error map

Each simulated user is plotted at its true Cartesian position `(x, y)`, while the point color represents its absolute localization error.

This makes it possible to see whether localization accuracy varies systematically with range or angle.

The figure also shows the approximate valid user region:

```text
50 m ≤ R ≤ 400 m
-60° ≤ θ ≤ 60°
```

using:

- the `R = 50 m` inner arc,
- the `R = 400 m` outer arc,
- the `θ = -60°` and `θ = 60°` angular boundaries,
- intermediate range guides,
- an error colorbar in meters.

The array itself is intentionally not drawn in this visualization.

---

## 10. Experiment Relationship

The main experiment sequence can be interpreted as:

```text
exp01_prt_range_error.py
        ↓
understand basic PRT error
        ↓
q_reso.py
        ↓
select Δq
        ↓
find_roi.py
        ↓
calibrate Kq
        ↓
roi_test.py
        ↓
validate ROI coverage
        ↓
hrt_tau_reso.py
        ↓
select HRT τ resolution
        ↓
hrt_test.py
        ↓
validate HRT refinement
        ↓
single_user.py
        ↓
Monte Carlo evaluation of the complete pipeline
```

---

## 11. Coordinate and Timing Conventions

### Position

```text
x = R sin(θ)
y = R cos(θ)
```

so `θ = 0°` lies along the positive y-axis.

### Absolute arrival time

For the array center,

```text
τ = R / c + b
```

where `b` is the clock offset.

### PRT wavefront approximation

```text
t(x) ≈ τ + p x + q x²
```

### HRT exact wavefront

```text
tₘ = τ + [dₘ(R, θ) - R] / c
```

These conventions are shared across the simulator, transforms, ROI construction, and localization pipeline.

---

## 12. Running the Code

From the repository root, install the required packages:

```bash
pip install numpy scipy matplotlib tqdm
```

Run individual experiments with:

```bash
python experiments/exp01_prt_range_error.py
python experiments/q_reso.py
python experiments/find_roi.py
python experiments/roi_test.py
python experiments/hrt_tau_reso.py
python experiments/hrt_test.py
python experiments/single_user.py
```

The main end-to-end Monte Carlo experiment is:

```bash
python experiments/single_user.py
```

---

## 13. Summary

The project implements a two-stage near-field localization strategy:

```text
PRT
↓
fast coarse localization using a parabolic wavefront model
↓
adaptive ROI
↓
small physically motivated search region
↓
HRT
↓
accurate refinement using the exact spherical-wave model
```

The PRT provides efficient coarse estimates of range, angle, and arrival time. The ROI converts the PRT uncertainty into a compact three-dimensional search region. The HRT then refines the estimate with the exact spherical-wave geometry, producing the final `(R, θ, τ)` and Cartesian `(x, y)` localization result.
