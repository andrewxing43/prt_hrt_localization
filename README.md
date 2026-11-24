# Near-Field Multiuser Localization with PRT and HRT

This project estimates the positions of one or more users from noisy, overlapping wideband Gaussian pulses received by a uniform linear array. For each user, the algorithm estimates range `R` from the array center, angle `theta`, and reference time of arrival (TOA) `tau`. It converts range and angle to a 2D position. Estimating TOA independently of range lets the simulation represent unknown transmission times, simultaneous arrivals, and users that are closely spaced in any of the three parameters.

The experiments evaluate positioning error, user separability, and search runtime. A paired experiment holds user positions and receiver noise fixed to measure how adding users changes the error for the same user.

## Signal Model and Localization

The array lies along the x-axis, with `theta = 0°` pointing along +y. For an element at `x_m`, the user position, propagation distance, and arrival time follow these equations.

```text
x = R sin(theta),  y = R cos(theta)
d_m = sqrt(R² + x_m² − 2 R x_m sin(theta))
tau = R/c + b
t_m = tau + (d_m − R)/c
```

The offset `b` represents a clock offset or equivalent transmission delay. The received signal includes a Gaussian envelope, carrier propagation phase, and free-space path loss. User signals are summed in the complex domain before adding complex white Gaussian noise. Defaults are a 24 GHz carrier, 8 GHz bandwidth, 64 GHz sample rate, and 513 half-wavelength-spaced elements. Transmit power is set using a reference SNR of 4 dB at 300 m.

1. **Matched filtering.** Filter the time series from each array element against the reference pulse.
2. **Coarse PRT search.** The parabolic Radon transform approximates the wavefront as `t(x) ≈ tau + p x + q x²`, where `p = −sin(theta)/c` and `q = cos²(theta)/(2cR)`. Select the strongest response among valid `(p, q, tau)` combinations and map it to range and angle.
3. **ROI and HRT.** Build a region of interest around the coarse estimate and search for the strongest hyperbolic Radon transform response along exact spherical wavefronts. The shared search bounds are 50–400 m and −60° to 60°. HRT grid steps are 1 m, 0.01°, and `dt/4` (about 3.91 ps).
4. **Joint refinement and SIC.** In each round of the main multiuser experiment, initialize a new user from the HRT peak in the residual. Use bounded Powell optimization on subsampled data to refine the new user and previously accepted users. Accept the update only if the peak exceeds the detection threshold and joint reconstruction reduces the residual energy on the full data. Subtract all accepted users' reconstructed signals from the original observations and repeat up to the iteration limit.

The simulation uses ground-truth angle and TOA only to speed up the PRT search by narrowing the bins examined. When the local window contains the same winning PRT peak as the full search, the shortcut preserves the resulting estimates and error metrics. PRT and HRT select peaks from the received data, and Powell refinement fits the reconstructed waveforms to those data. The true user count is used for scene generation and scoring. The single-user `localization.pipeline.localize()` API accepts both hints together or runs global PRT when both are omitted.

## Running the Experiments

Install NumPy, SciPy, Matplotlib, tqdm, and a CuPy build compatible with your NVIDIA CUDA environment. The shared PRT and HRT implementations use the GPU. Plotting scripts need a Matplotlib backend that can display figures.

Run these commands from the repository root.

```bash
python experiments/single_user.py
python experiments/single_user_refine.py
python experiments/multi_user_final.py
python experiments/multi_user_test.py
python experiments/5_users.py
```

Each script defines its trial counts, random seeds, and scene constraints near the top. Shared signal, array, and PRT defaults live in `config.py`.

## File Guide

### Configuration, Signal Model, and Localization

| File | Purpose |
| --- | --- |
| [`config.py`](config.py) | Physical constants, system and array settings, noise and transmit power, and default PRT grids. |
| [`signal_model/gaussian_source.py`](signal_model/gaussian_source.py) | Gaussian pulses, pulse time axes, and matched-filter reference waveforms. |
| [`signal_model/propagation.py`](signal_model/propagation.py) | User coordinates, per-element spherical propagation distances and delays, and complex received signals with phase and path loss. |
| [`signal_model/noise.py`](signal_model/noise.py) | Thermal noise power and complex additive white Gaussian noise. |
| [`signal_model/matched_filter.py`](signal_model/matched_filter.py) | Per-element temporal matched filtering. |
| [`transforms/prt.py`](transforms/prt.py) | CuPy implementation of the coherent parabolic Radon response and peak extraction. |
| [`transforms/hrt.py`](transforms/hrt.py) | Spherical relative delays, TOA and clock-offset conversion, and CuPy HRT response and peak extraction. |
| [`localization/parameter_mapping.py`](localization/parameter_mapping.py) | Conversion between PRT parameters `p, q` and physical parameters `R, theta`. |
| [`localization/roi.py`](localization/roi.py) | Adaptive range, angle, and TOA search bounds based on coarse estimates and grid uncertainty. |
| [`localization/pipeline.py`](localization/pipeline.py) | The shared matched-filter → PRT → ROI → HRT pipeline, with local and global search, valid-grid masking, and coordinate conversion. |

### Localization and Controlled Comparisons

| File | Purpose |
| --- | --- |
| [`experiments/single_user.py`](experiments/single_user.py) | Randomized single-user baseline. Reports 2D errors using the HRT estimates and plots an error CDF and spatial error map. |
| [`experiments/single_user_refine.py`](experiments/single_user_refine.py) | Randomized single-user experiment with joint Powell refinement and SIC from the main experiment. Error statistics include scenes with one detection. |
| [`experiments/multi_user_final.py`](experiments/multi_user_final.py) | Main Monte Carlo experiment for one to three users. Runs joint refinement and SIC, matches estimates to true positions, and plots grouped and spatial errors. Exposes the localization routine used by other scripts. |
| [`experiments/multi_user_test.py`](experiments/multi_user_test.py) | Paired experiment with seven scenes per block (A, B, C, AB, AC, BC, and ABC). Shared user parameters, time axis, and receiver noise allow comparisons for the same users across scene sizes. |
| [`experiments/5_users.py`](experiments/5_users.py) | Fixed five-user stress test with a common center-array TOA of 1500 ns and a six-round SIC limit. Prints detection and matching results and plots positions. |
| [`experiments/multi_user_SIC.py`](experiments/multi_user_SIC.py) | Standalone fixed-scene SIC demonstration with exact-model candidate scanning, joint refinement, and fixed or complex gain fitting. |
| [`experiments/angle_test.py`](experiments/angle_test.py) | Fixed three-user diagnostic scene with a shared angle and a shared TOA for two users. Uses SIC with candidate scanning. |
| [`experiments/time_comparison.py`](experiments/time_comparison.py) | Runtime and localization-error comparison of global HRT and PRT → ROI → HRT on the same inputs. Processes HRT in blocks. |

### Resolution, ROI, and Peak Diagnostics

| File | Purpose |
| --- | --- |
| [`experiments/2_user_range_resolution.py`](experiments/2_user_range_resolution.py) | Two-user range resolution test with a prescribed distance separation. |
| [`experiments/2_user_angle_resolution.py`](experiments/2_user_angle_resolution.py) | Two-user angle resolution test with a prescribed angular separation. |
| [`experiments/2_user_tau_resolution.py`](experiments/2_user_tau_resolution.py) | Two-user TOA resolution test with a prescribed arrival-time separation. |
| [`experiments/2_user_peaks.py`](experiments/2_user_peaks.py) | PRT range responses for two users with a shared angle and TOA, plotted separately and after superposition. |
| [`experiments/SIC_effect.py`](experiments/SIC_effect.py) | PRT responses before and after estimating and subtracting the first of two users. Can save `sic_prt_demo.png`. |
| [`experiments/exp01_prt_range_error.py`](experiments/exp01_prt_range_error.py) | PRT coarse range error across a grid of distances and angles. |
| [`experiments/q_reso.py`](experiments/q_reso.py) | Effect of the PRT curvature grid step on range estimation. |
| [`experiments/hrt_tau_reso.py`](experiments/hrt_tau_reso.py) | Effect of HRT time-grid spacing on range, angle, and TOA errors in a fixed scene. |
| [`experiments/find_roi.py`](experiments/find_roi.py) | High-percentile PRT curvature and range errors across random positions and noise trials, used to select the ROI width. |
| [`experiments/roi_test.py`](experiments/roi_test.py) | Validation of true-range coverage and worst-case margin for the selected distance ROI. |

### Plotting, Documentation, and Generated Files

| File | Purpose |
| --- | --- |
| [`plotting/received_signal.py`](plotting/received_signal.py) | Example received signal over array position and time. |
| [`plotting/transforms.py`](plotting/transforms.py) | Example 2D responses for linear, parabolic, and hyperbolic Radon transforms. |
| [`utils/plotting.py`](utils/plotting.py) | Reusable plots for PRT curvature slices, range errors, and spatial error maps. |
| [`README.md`](README.md) | Project overview, run instructions, and file index. |
| `experiments/paired_user_results.csv` | Per-scene, per-user estimate details generated by the paired experiment. |
| `experiments/paired_user_comparison.csv` | Paired user-level error comparisons generated by the paired experiment. |
| `experiments/multi_user_localization_errors.csv` | Legacy main-experiment details. The current entry point keeps trial records in memory and produces summary statistics and plots. |
| `sic_prt_demo.png` | Figure produced by the SIC peak demonstration when saving is enabled. |
| `__pycache__/*.pyc` | Python bytecode caches generated on import. |

## Main Experiment Settings and Metrics

The main experiment runs 600 single-user, 300 two-user, and 200 three-user scenes. It samples range, angle, and center-array TOA over 50–400 m, −60° to 60°, and 1500–1502 ns. Each pair must satisfy at least one separation condition (range ≥ 200 m, angle ≥ 1°, or TOA ≥ 125 ps). The two-user resolution scripts reuse the main localization routine and configure their own test separations.

The main experiment uses an HRT detection threshold of `6e-6` and allows up to four SIC rounds. Powell search bounds are centered on each initializing HRT peak, with half-widths of 50 m in range, 0.25° in angle, and 0.035 ns in TOA.

When the estimated and true user counts agree, the Hungarian algorithm finds the one-to-one assignment that minimizes total 2D distance. Each matched Euclidean distance contributes one error sample. The summary reports mean absolute localization error (ALE), RMSE, P95, and maximum error. Overall metrics pool valid users across all groups. Scenes with incorrect counts contribute to trial totals, and only scenes with correct counts contribute to the error statistics and plots.
