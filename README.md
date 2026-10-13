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

The repository also contains exploratory experiments not needed for the two workflows documented here. `config.py` is shared, but several ROI, HRT, PRT, and SIC parameters are constants in their corresponding implementation files rather than entries in `config.py`. Adjust these values with care when comparing runs.
