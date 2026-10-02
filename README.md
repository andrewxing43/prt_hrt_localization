# PRT/HRT Near-Field Localization

A compact research prototype for near-field localization with a uniform linear array. The current work focuses on using the **Parabolic Radon Transform (PRT)** for coarse localization and designing a small **region of interest (ROI)** for a later HRT refinement stage.

The PRT model is

\[
t(x)=\tau+px+qx^2,
\]

with

\[
p=-\frac{\sin\theta}{c},\qquad
q=\frac{\cos^2\theta}{2Rc},
\]

and

\[
\theta=\arcsin(-pc),\qquad
R=\frac{1-(pc)^2}{2qc}=\frac{\cos^2\theta}{2cq}.
\]

## Repository Structure

```text
prt_hrt_localization/
├── config.py
├── experiments/
│   ├── exp01_prt_range_error.py
│   ├── q_reso.py
│   ├── find_roi.py
│   └── exp02_prt_roi_hrt.py
├── localization/
│   ├── parameter_mapping.py
│   ├── roi.py
│   └── pipeline.py
├── signal_model/
│   ├── gaussian_source.py
│   ├── propagation.py
│   ├── noise.py
│   └── matched_filter.py
├── transforms/
│   ├── prt.py
│   └── hrt.py
└── utils/
    └── plotting.py
```

### Main files

- `config.py` — system, array, sampling, SNR, and random-seed configuration.
- `signal_model/` — Gaussian pulse generation, exact spherical-wave propagation, AWGN, and matched filtering.
- `transforms/prt.py` — PRT implementation and search over \((\tau,p,q)\).
- `localization/parameter_mapping.py` — conversion between \((R,\theta)\) and \((p,q)\).
- `exp01_prt_range_error.py` — basic PRT localization and range-error experiment.
- `transforms/hrt.py`, `localization/roi.py`, `localization/pipeline.py`, and `exp02_prt_roi_hrt.py` — reserved for the later ROI/HRT refinement pipeline.

## `q_reso.py`

`q_reso.py` studies how the **q-grid resolution** affects range estimation error.

For a fixed off-grid angle, it scans multiple true ranges and repeats the PRT search for several values of

\[
\Delta q.
\]

Both \(p\) and \(\tau\) are intentionally off-grid so that the test is not artificially favorable. The script plots

\[
R_{\text{true}} \rightarrow |\hat R-R_{\text{true}}|
\]

for each q resolution.

The current experiments indicate that decreasing \(\Delta q\) below roughly

\[
\boxed{\Delta q=10^{-13}\ \text{s/m}^2}
\]

provides little additional improvement in range error. This value is therefore used as the working q-grid resolution for ROI experiments.

## `find_roi.py`

`find_roi.py` is used to determine a statistically reliable ROI.

It:

1. randomly samples user positions in
   \[
   R\in[50,400]\ \text{m},\qquad \theta\in[-60^\circ,60^\circ],
   \]
2. performs multiple Monte Carlo trials with different AWGN realizations,
3. uses fixed global \(p,q,\tau\) grids,
4. searches only the nearby p-grid points and tau-grid points for each user,
5. derives a local q search interval from the nearby angle grid while keeping the global q spacing fixed,
6. measures
   \[
   K_q=\frac{|\hat q-q_{\text{true}}|}{\Delta q},
   \]
7. studies the high-percentile value \(K_{q,99}\) as a function of estimated range and angle.

The corresponding effective q uncertainty is

\[
\boxed{\Delta q_{\text{eff}}=K_{q,99}\Delta q.}
\]

Monte Carlo simulation is therefore used to calibrate the q-error term rather than treating one q-grid spacing as the full estimation uncertainty.

## ROI Definition

The ROI is centered at the PRT estimate \((\hat\tau,\hat\theta,\hat R)\).

### Theta-axis ROI

Experiments show that the angle estimate stays within approximately two angle-grid steps. With

\[
\Delta\theta_{\text{grid}}=0.1^\circ,
\]

define the **theta half-width** as

\[
\boxed{H_\theta=2\Delta\theta_{\text{grid}}=0.2^\circ.}
\]

Thus

\[
\theta_{\text{ROI}}=[\hat\theta-H_\theta,\hat\theta+H_\theta].
\]

The total theta-axis width is \(0.4^\circ\).

### Tau-axis ROI

The sampling interval is

\[
\Delta\tau_{\text{grid}}=\frac{1}{f_s}.
\]

For the default 64 GHz sampling rate,

\[
\Delta\tau_{\text{grid}}=15.625\ \text{ps}.
\]

Using two grid steps on each side gives the **tau half-width**

\[
\boxed{H_\tau=2\Delta\tau_{\text{grid}}=31.25\ \text{ps}.}
\]

Therefore

\[
\tau_{\text{ROI}}=[\hat\tau-H_\tau,\hat\tau+H_\tau].
\]

The total tau-axis width is \(62.5\) ps.

### Range-axis ROI

Range is related to angle and q by

\[
R=\frac{\cos^2\theta}{2cq}.
\]

A first-order error propagation gives

\[
|\Delta R_q|\approx
\frac{2cR^2}{\cos^2\theta}|\Delta q|,
\]

and

\[
|\Delta R_\theta|\approx
2R|\tan\theta||\Delta\theta|,
\]

where the angle error in the second expression is in radians.

The proposed **range half-width** is therefore

\[
\boxed{
H_R(\hat R,\hat\theta)=
\frac{2c\hat R^2}{\cos^2\hat\theta}\Delta q_{\text{eff}}
+2\hat R|\tan\hat\theta|\Delta\theta_{\text{eff}}
}
\]

with

\[
\Delta q_{\text{eff}}=K_{q,99}(\hat R,\hat\theta)\Delta q.
\]

The range ROI is

\[
\boxed{
R_{\text{ROI}}=[\hat R-H_R,\hat R+H_R].
}
\]

This gives the ROI a physical interpretation: it becomes wider at long range and at large absolute angles because the range estimate becomes more sensitive to q and angle errors.

## Intended Processing Flow

```text
Received array data
      ↓
PRT coarse localization
      ↓
(τ_hat, θ_hat, R_hat)
      ↓
ROI from H_tau, H_theta, H_R(R_hat, θ_hat)
      ↓
HRT refinement inside the ROI
```

## Run

From the repository root:

```bash
python experiments/q_reso.py
python experiments/find_roi.py
```

Required packages:

```bash
pip install numpy scipy matplotlib tqdm
```

Required for fast transform:
```bash
pip install cupy
```
