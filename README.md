# PRT/HRT Near-Field Localization

Research prototype for near-field localization using a uniform linear array (ULA). The primary single-user workflow is **matched filtering → Parabolic Radon Transform (PRT) → adaptive region of interest (ROI) → Hyperbolic Radon Transform (HRT)**. The multi-user experiment in `experiments/SIC_2.py` extends this with **successive interference cancellation (SIC), exact-spherical-wave candidate search, and joint parameter refinement**.

> **Scope:** `single_user.py` is a truth-hint-assisted Monte Carlo evaluation, not a fully blind detection benchmark. `SIC_2.py` also uses scenario-derived angle and common-arrival-time hints. Neither experiment should be interpreted as fully blind multi-user localization.

## 1. Setup and quick start

From the `prt_hrt_localization/` directory:

```bash
pip install numpy scipy matplotlib tqdm
python experiments/single_user.py
python experiments/SIC_2.py
```

Both experiment scripts bootstrap the project root onto `sys.path`. The single-user script displays plots interactively; the multi-user script prints the SIC iteration log and recovered users. The example settings can be changed directly in the relevant experiment script.

## 2. System and coordinate conventions

Shared parameters are set in `config.py`:

| Parameter | Current default |
|---|---:|
| Carrier frequency | 24 GHz |
| Bandwidth | 8 GHz |
| Oversampling factor / sample rate | 8 / 64 GHz |
| Sampling interval (`SYSTEM.dt`) | 15.625 ps |
| ULA antennas | 513 |
| Antenna spacing | Half wavelength |
| Reference SNR | 10 dB **at 300 m** |
| Reference range | 300 m |
| PRT global p-grid points (`PRT.num_p_points`) | 500 |
| PRT q step (`PRT.dq`) | 1 × 10⁻¹³ |
| Noise temperature | 300 K |

The array lies along the x-axis, centered at the origin; `θ = 0°` is broadside (+y direction). Both experiments use `x = R sin(θ)` and `y = R cos(θ)`. The reference-element arrival time is `τ = R/c + b`, where `b` is clock offset. The PRT approximates the wavefront as `t(x) ≈ τ + p x + q x²`, with `p = −sin(θ)/c` and `q = cos²(θ)/(2Rc)`. The HRT uses exact spherical-wave propagation delays.

The physical signal simulator is in `signal_model/`: Gaussian pulse (`gaussian_source.py`), spherical-wave propagation (`propagation.py`), thermal complex AWGN (`noise.py`), and matched filtering (`matched_filter.py`). The transmitted power is fixed by the **reference** SNR at 300 m, so received SNR varies with range rather than remaining 10 dB for every user.

## 3. Single-user localization workflow

Core API (`localization/pipeline.py`):

```python
from localization.pipeline import localize

# Blind/global PRT search:
result = localize(rx, time_axis, antenna_x)

# Hint-assisted PRT search (both hints required):
result = localize(rx, time_axis, antenna_x,
                  theta_hint=theta_approx, tau_hint=tau_approx)
```

`rx` is the received antenna-by-time complex signal, `time_axis` is the sampled time vector, and `antenna_x` contains antenna positions. Supplying only one of the two hints raises `ValueError`.

1. **Matched filter** each antenna's received signal.
2. **PRT coarse estimate:** search p, q and arrival time for the strongest *valid* PRT response, and convert `(p, q)` into `(R_prt, θ_prt)`.
3. **Adaptive ROI:** call `build_roi()` with the PRT estimate and global p-grid spacing `dp=DP`. Bound the ROI to the supported distance and angular domain.
4. **HRT refinement:** search the ROI with the exact spherical-wave model and pick the maximum-magnitude HRT response.
5. **Position conversion:** return `(x, y)` alongside `(R, θ, τ)`.

### PRT search: current code, not the older README values

`localization/pipeline.py` currently uses `PRT_THETA_MIN/MAX = −60°/+60°`, `PRT_R_MIN/MAX = 50/400 m`, **500 evenly spaced global p values** (not a fixed 0.1° global angle grid), and a q grid with `PRT.dq = 1e-13`, derived from the configured angle/range limits with an extra two q steps at each end. Candidate `(p,q)` cells are masked to reconstructed ranges between 50 and 400 m.

| Mode | p-axis | τ-axis for peak selection | q-axis |
|---|---|---|---|
| Global (no hints) | All 500 p values | All sampled times | Global q grid |
| Hint-assisted (both hints) | 4 nearest p bins to `theta_hint` | 4 nearest time samples to `tau_hint` | Global q grid |

The implementation computes the PRT on the entire input time axis even in hint-assisted mode; the four τ samples constrain **peak selection**, not the time dimension used to construct the transform. The hints are not passed to ROI/HRT after coarse localization.

### ROI and HRT

`localization/roi.py` uses `K_Q = 4.5`, `P_HALF_WIDTH_BINS = 2`, and `TAU_HALF_WIDTH_SAMPLES = 2`. In the standard pipeline, `dp=DP` determines angular limits from `p_hat ± 2·DP` rather than applying a fixed ±0.2° angle interval. The resulting maximum angular half-width also feeds the range uncertainty:

```text
Δq_eff = K_Q × PRT.dq
W_R = [2 c R_prt² / cos²(θ_prt)] Δq_eff
      + 2 R_prt |tan(θ_prt)| Δθ
τ ROI = τ_prt ± 2 × SYSTEM.dt
```

Here `Δθ` is in **radians**. The range ROI is `[R_prt − W_R, R_prt + W_R]`, clipped to 50–400 m; angular bounds are clipped to −60° to +60°. The current HRT grid increments in `localization/pipeline.py` are **1 m**, **0.005°**, and **SYSTEM.dt/4 = 3.90625 ps** for range, angle, and arrival time respectively.

The returned `LocalizationResult` has final fields `R`, `theta`, `tau`, `x`, `y`; coarse fields `prt_R`, `prt_theta`, `prt_tau`; and diagnostics `roi`, `prt_peak`, `hrt_peak`.

### `experiments/single_user.py`: Monte Carlo evaluation

Run:

```bash
python experiments/single_user.py
```

Current scenario constants:

| Setting | Value |
|---|---:|
| Trials (`N_MC`) | 200 |
| True range | Uniform(50, 400) m |
| True angle | Uniform(−60°, +60°) |
| Clock offset | 10 ns |
| Reference SNR | 10 dB at 300 m; range-dependent in trials |
| Position and noise RNG seeds | `SIM.rng_seed`, `SIM.rng_seed + 1` |

For each trial the script samples one user, constructs a time axis around all antenna propagation delays, generates the exact spherical-wave received signal, adds AWGN, and runs:

```python
result = localize(noisy_rx, time_axis, antenna_x,
                  theta_true, tau_true)
```

Thus **true angle and true arrival time are provided as PRT hints**, while range is still estimated. The 2-D position error is `e = hypot(x_hat − x_true, y_hat − y_true)`. The script prints trial count, transmitted power, observed minimum/maximum SNR, **ALE = mean(e)**, **RMSE = sqrt(mean(e²))**, and maximum error. It displays (1) the empirical 2-D-error CDF and (2) a scatter map of true `(x,y)` locations colored by error with a colorbar. The helper `draw_user_region()` exists, but its invocation is currently **commented out**; region outlines are not shown by default. Figures are displayed via `plt.show()` and are not saved automatically.

## 4. Multi-user localization: `experiments/SIC_2.py`

Run:

```bash
python experiments/SIC_2.py
```

This is a **separate, scenario-assisted SIC experiment**, not simply repeated calls to `localize()`. It reuses PRT settings and `build_roi()`/`run_hrt()` but implements its own local PRT selection, exact-model parameter search, and joint reconstruction/refitting. The number of detected users is decided iteratively (up to the configured maximum); it is not supplied to the stopping check as a required exact count.

### Simulated users and shared arrival time

Current `USERS` in the script:

| User | Range (m) | Angle (°) |
|---|---:|---:|
| 1 | 300.1 | 59.7 |
| 2 | 150.1 | 59.7 |
| 3 | 60.2 | 59.7 |

The script sets:

```python
BASE_CLOCK_OFFSET = 10e-9
COMMON_TAU = max(u["range_m"] for u in USERS) / C + BASE_CLOCK_OFFSET
# Each user's simulation clock offset:
clock_offset_i = COMMON_TAU - range_i / C
```

All three users therefore share the **same array-center arrival time**, despite different ranges, because their clock offsets differ. The generated antenna signals are summed **before** AWGN is added. This is a deliberately challenging same-angle/same-arrival-time separation scenario, not a generic asynchronous multi-user benchmark. The local PRT angle hints are taken from `USERS` and its τ hint from `COMMON_TAU`; the full-range exact search also uses those angle centers and common τ.

### SIC iteration steps

```text
Summed noisy multi-user array signal
        ↓
Residual matched filter → hint-assisted local PRT
        ↓
PRT-derived ROI → exact-wavefront HRT peak
        ↓
HRT peak ≤ stop threshold? ── yes → stop
        ↓ no
Full-range exact-model candidate search on residual
        ↓
Try top separated candidates + joint refit of all accepted users
        ↓
Does any candidate strictly lower full residual energy?
        ├── no → stop
        └── yes → accept; update all users and subtract reconstructions
                         ↓
                   next SIC iteration
```

Per iteration:

1. Match-filter the current residual. Construct the local p-grid as the **union of four neighboring global p bins for each configured user's angle** and select four τ samples near `COMMON_TAU`; q remains global. Locate the masked PRT peak.
2. Build an adaptive ROI around the PRT result and run HRT. **Stop** if `hrt_peak <= HRT_STOP_THRESHOLD`.
3. Perform an exact spherical-wave grid search on subsampled residual data across the **full configured range interval**, small offsets around the scenario's angle hints, and offsets around common τ. Also evaluate the PRT and HRT estimates as extra seeds when valid.
4. Rank seeds by template-explained energy, retain at most `N_SEARCH_CANDIDATES` and require distinct seed ranges separated by at least `CANDIDATE_SEPARATION_M` under the script's `<` test.
5. For each candidate, jointly optimize the candidate **and previously accepted users** in `(range, angle, arrival time)` using bounded Powell optimization and the signal reconstruction model. Evaluate the final **full-resolution** residual energy; accept the best candidate only when its energy is **strictly lower** than before the iteration.
6. Replace accepted user estimates with the joint-fit results, reconstruct their waveforms, and continue with the updated residual.

**Stopping logic:** the implemented acceptance rules are (a) HRT-peak hard threshold and (b) strictly positive full-residual-energy reduction; the loop also ends if no candidate exists or after `MAX_SIC_ITERATIONS`. There is **no separate duplicate-detection rejection**, no minimum relative-energy-improvement threshold, and no DBSCAN stage in this workflow. Candidate seed diversity by range is distinct from a final duplicate-rejection condition.

### Important adjustable settings

All of these are defined near the top of `experiments/SIC_2.py`:

| Parameter | Current value | Effect |
|---|---|---|
| `HRT_STOP_THRESHOLD` | `6e-6` | Stops if current residual HRT peak is at/below threshold |
| `MAX_SIC_ITERATIONS` | `3` | Maximum attempts to accept users |
| `GAIN_MODE` | `"fixed"` | Reconstruction amplitudes fixed to `1+0j`; alternative `"complex"` fits complex gains |
| `SEARCH_R_MIN/MAX/STEP` | `50 / 400 / 4` m | Full-range exact-model seed scan |
| `SEARCH_THETA_OFFSETS` | `[-0.15, 0, 0.15]` degrees | Offsets from configured users' angle hints |
| `SEARCH_TAU_OFFSETS_NS` | `[-0.015, 0, 0.015]` ns | Offsets from `COMMON_TAU` |
| `OPT_ANTENNA_STRIDE`, `OPT_TIME_STRIDE` | `2`, `2` | Candidate search and iterative optimization subsampling |
| `N_SEARCH_CANDIDATES` | `2` | Maximum seeds trialed per SIC iteration |
| `CANDIDATE_SEPARATION_M` | `8` m | Minimum separation of retained range seeds |
| `REFINE_R_RADIUS` | `15` m | Range bound around each seed |
| `REFINE_THETA_RADIUS` | `0.25` degrees | Angular optimization bound around each seed |
| `REFINE_TAU_RADIUS_NS` | `0.035` ns | Arrival-time optimization bound around each seed |
| `MAX_JOINT_SWEEPS` | `3` | Maximum sequential joint-refinement sweeps |
| `OPT_MAXITER`, `OPT_FTOL` | `100`, `1e-10` | Powell optimizer settings |

**Gain-mode caveat:** the exact-grid seed ranking uses `fit_complex_gain()`, which obeys `GAIN_MODE`. With `GAIN_MODE="fixed"` the fit does not freely rescale a waveform; with `"complex"`, the per-template complex amplitude is estimated. The full joint residual is evaluated on the original-resolution signal after subsampled optimization.

### Output and interpretation

The script logs per-iteration PRT/HRT estimates, exact-search candidates, joint-refinement sweeps, energy reduction, and accepted users. At the end it prints the simulated true users, the final detected count and estimates, recovered `clock_offset_s = tau_s − range_m/C`, Cartesian locations, fitted `alpha`, PRT/HRT peak diagnostics, and initial/final energy and noise variance. The `run_sic()` function returns `(accepted, residual)`. Accepted users may be **updated in later iterations** by joint fitting; the iteration recorded with each accepted user is the iteration in which it was first added.

Because the search uses the configured users' angle hints and a known common τ, its results evaluate the **assisted SIC/refinement pipeline**, not unknown-user blind discovery. The HRT peak threshold is an absolute numerical threshold for this simulation setting, not a calibrated general-purpose probability-of-detection threshold.

## 5. Repository layout

```text
prt_hrt_localization/
├── README.md
├── config.py
├── experiments/
│   ├── single_user.py         # one user per Monte Carlo trial
│   ├── SIC_2.py               # multi-user assisted SIC + joint refit
│   ├── 2_user_peaks.py        # auxiliary multi-user experiment
│   ├── exp01_prt_range_error.py
│   ├── find_roi.py
│   ├── hrt_tau_reso.py
│   ├── hrt_test.py
│   ├── noise_peak.py
│   ├── prt_hrt_peak_level.py
│   ├── q_reso.py
│   ├── roi_test.py
│   └── time_comparison.py
├── localization/
│   ├── parameter_mapping.py
│   ├── pipeline.py
│   └── roi.py
├── signal_model/
│   ├── gaussian_source.py
│   ├── matched_filter.py
│   ├── noise.py
│   └── propagation.py
├── transforms/
│   ├── prt.py
│   └── hrt.py
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


Find your system's supported CUDA version using:

```bash
nvidia-smi
```

Then, install as appropriate:
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

You may also need to install CUDA toolkit headers (replace with ```cuda11x``` if needed):
```bash
pip install cupy-cuda12x[ctk]
```