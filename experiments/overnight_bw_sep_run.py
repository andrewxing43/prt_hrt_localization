"""
Blind two source HRT studies
Varying BW and the gap between the two sources in time of arrival
"""

import csv
import json
from pathlib import Path
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import next_fast_len
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

HERE = Path(__file__).resolve().parent
ROOT = HERE if (HERE / "config.py").is_file() else HERE.parent
if not (ROOT / "config.py").is_file():
    ROOT = HERE / "sic2_integration_fixture"
sys.path.insert(0, str(ROOT))

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import (
    propagation_delays, propagation_distances, received_signal,
)
from signal_model.qam_source import MultiSourceGen


# The fixed pair is repeated at EVERY arrival gap. The broader and null
# experiments use independent trials, with separate random streams.
N_MC = 300
N_GENERAL_MC = 300
N_SINGLE_MC = 300
N_CALIBRATION = 200
N_QAM_MC = 200
N_QAM_CALIBRATION = 100
GAPS_PS = (0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 192, 224, 256)
BANDWIDTHS_GHZ = (1.0, 4.0, 8.0)
ILLUSTRATION_GAP_PS = 160.0
RANGES_M = np.arange(30.0, 75.0 + 0.25, 0.5)
R1_M, R2_M = 40.0, 60.0
QAM_DURATION_S = 2e-9
QAM_ROLLOFF = 0.35
SEED = int(getattr(SIM, "rng_seed", 0))
TIME_OVERSAMPLE = 1
MIN_PEAK_DISTANCE_SAMPLES = 2
CALIBRATED_FALSE_SPLIT_QUANTILE = 0.95
NOISE_MAX_QUANTILE = 0.99
RANGE_TOL_M = 5.0
TAU_TOL_SAMPLES = 1.0
OUTPUT = HERE / "overnight_bw_separation_results"


def bw_sigma(bandwidth_ghz):
    return 1 / (np.sqrt(2) * np.pi * bandwidth_ghz * 1e9)


def reference_noise(x):
    ref_r = float(getattr(SIM, "reference_range", 300.0))
    ref_snr = float(getattr(SIM, "reference_snr_db", 10.0))
    center = int(np.argmin(abs(x)))
    delay = propagation_delays(ref_r, 0.0, antenna_x=x)[center]
    record, _ = received_signal(np.array([delay]), ref_r, 0.0, antenna_x=x)
    variance = float(abs(record[center, 0])**2 / 10**(ref_snr / 10))
    return variance, ref_r, ref_snr


def axis(center, half_before, half_after, dt):
    start = int(np.floor((center - half_before) / dt))
    stop = int(np.ceil((center + half_after) / dt))
    return np.arange(start, stop + 1) * dt


def gaussian_source(t, x, r, center_tau, bandwidth_ghz, tx_phase=0.0):
    offset = center_tau - r / C
    distances = propagation_distances(r, 0.0, x)
    delays = distances / C + offset
    sigma = bw_sigma(bandwidth_ghz)
    # Same continuous-time transmitted pulse energy at all B, with 8 GHz
    # unit-peak Gaussian as the energy reference.
    amplitude = np.sqrt(bw_sigma(8.0) / sigma)
    envelope = np.exp(-0.5 * ((t[None, :] - delays[:, None]) / sigma)**2)
    f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))
    return (amplitude * envelope / distances[:, None]
            * np.exp(1j * (tx_phase - 2*np.pi*f0*delays))[:, None])


class RRCAtBandwidth(MultiSourceGen):
    """Ordinary RRC-shaped QAM; vary symbol rate but keep burst duration."""

    def __init__(self, bandwidth_ghz, x):
        super().__init__(pw_sec=QAM_DURATION_S, M=16, beta=QAM_ROLLOFF,
                         antenna_x=x, seed=SEED)
        self.bandwidth = bandwidth_ghz * 1e9
        self.n_symbols = int(round(QAM_DURATION_S * self.bandwidth))


def qam_pair(t, x, bandwidth_ghz, gap_ps, symbol_seed=SEED+310,
             active_sources=(0, 1), phases=(0.0, 0.0)):
    generator = RRCAtBandwidth(bandwidth_ghz, x)
    symbol_rng = np.random.default_rng(symbol_seed)
    center_tau1 = R1_M / C
    clean = np.zeros((len(x), len(t)), complex)
    info = []
    for k, (r, tau) in enumerate(((R1_M, center_tau1),
                                  (R2_M, center_tau1 + gap_ps*1e-12))):
        symbols = generator._qam_symbols(generator.n_symbols, symbol_rng)
        if k not in active_sources:
            continue
        source = dict(range_m=r, theta_deg=0.0,
                      clock_offset=tau-r/C, symbols=symbols)
        component, metadata = generator.received_signal(t, [source])
        # The propagation models use different nominal transmit amplitudes.
        # Normalize EACH received QAM source to the received energy of the
        # corresponding 8 GHz Gaussian source at the same range.
        target = gaussian_source(t, x, r, tau, 8.0)
        center = int(np.argmin(abs(x)))
        e_qam = SYSTEM.dt * np.sum(abs(component[center])**2)
        e_target = SYSTEM.dt * np.sum(abs(target[center])**2)
        if e_qam <= 0:
            raise ValueError("QAM component has zero energy")
        clean += component * np.sqrt(e_target / e_qam) * np.exp(1j*phases[k])
        waveform = metadata[0]["waveform_reference"]
        spectrum = abs(np.fft.fft(waveform, n=next_fast_len(8*len(waveform))))**2
        freq = np.fft.fftfreq(len(spectrum), SYSTEM.dt)
        mu = np.sum(freq*spectrum) / np.sum(spectrum)
        effective_bw = 2*np.sqrt(np.sum((freq-mu)**2*spectrum)/np.sum(spectrum))
        info.append(dict(source=k+1, symbol_count=generator.n_symbols,
                         symbol_rate_ghz=bandwidth_ghz,
                         effective_2xrms_bandwidth_ghz=float(effective_bw/1e9),
                         received_energy_at_center_j=float(e_target)))
    return clean, info


class BroadsideHRT:
    """Exact spherical fractional-delay HRT, cached across MC trials."""

    def __init__(self, t, x, ranges):
        self.t = t
        self.x = x
        self.ranges = ranges
        self.dt = float(t[1] - t[0])
        self.nt = len(t)
        self.guard = int(np.ceil(np.ptp(x)/(C*self.dt))) + 12
        self.nfft = next_fast_len(self.nt + 2*self.guard)
        freq = np.fft.fftfreq(self.nfft, self.dt)
        f0 = float(getattr(SYSTEM, "f0", SYSTEM.fc))
        self.steering = []
        for r in ranges:
            d = propagation_distances(float(r), 0.0, x)
            rel = (d-r)/C
            amp = r/d
            weights = amp/np.linalg.norm(amp)
            self.steering.append((weights[:, None] * np.exp(
                2j*np.pi*(f0+freq[None, :])*rel[:, None]
            )).astype(np.complex64))
        self.steering = np.asarray(self.steering)

    def scores(self, data):
        padded = np.pad(data, ((0, 0),
                               (self.guard, self.nfft-self.nt-self.guard)))
        spectrum = np.fft.fft(padded, axis=1).astype(np.complex64)
        aligned = np.einsum("rmf,mf->rf", self.steering, spectrum,
                            optimize=False)
        trace = np.fft.ifft(aligned, axis=1)
        return (abs(trace[:, self.guard:self.guard+self.nt])**2).astype(np.float32)


def add_noise(clean, rng, variance):
    return clean + np.sqrt(variance/2) * (
        rng.standard_normal(clean.shape) + 1j*rng.standard_normal(clean.shape)
    )


def profile_peaks(score, noise_height, prominence_fraction):
    """Blind in source count and arrival time, conditional on theta=0 slice.

    First maximize over the *entire* declared range domain at each time.
    scipy.signal.find_peaks finds time maxima; range is assigned at each peak.
    The truth enters only the evaluation metrics after the estimates are set.
    """
    profile = score.max(axis=0)
    indices, props = find_peaks(
        profile, distance=MIN_PEAK_DISTANCE_SAMPLES,
        height=noise_height,
        prominence=prominence_fraction*float(profile.max()),
    )
    r_indices = score[:, indices].argmax(axis=0)
    return [(float(RANGES_M[ir]), int(it), float(p))
            for ir, it, p in zip(r_indices, indices, props["peak_heights"])], profile


def maximum_secondary_prominence_fraction(profile):
    indices, props = find_peaks(profile, distance=MIN_PEAK_DISTANCE_SAMPLES,
                                prominence=0)
    if len(indices) < 2 or profile.max() <= 0:
        return 0.0
    return float(np.sort(props["prominences"])[-2]/profile.max())


def calibrate_detector(hrt, x, variance):
    """Independent noise-only and one-source records set fixed thresholds."""
    rng = np.random.default_rng(SEED + 990)
    tau1 = R1_M/C
    noise_maxima, secondary = [], []
    for k in range(N_CALIBRATION):
        empty = add_noise(np.zeros((len(x), len(hrt.t)), complex), rng, variance)
        noise_maxima.append(float(hrt.scores(empty).max()))
        r = R1_M if k % 2 == 0 else R2_M
        phase = rng.uniform(-np.pi, np.pi)
        one = add_noise(gaussian_source(hrt.t, x, r, tau1, 8.0, phase),
                        rng, variance)
        profile = hrt.scores(one).max(axis=0)
        secondary.append(maximum_secondary_prominence_fraction(profile))
    noise_height = float(np.quantile(noise_maxima, NOISE_MAX_QUANTILE))
    prominence = float(np.quantile(secondary,
                                   CALIBRATED_FALSE_SPLIT_QUANTILE))
    # An additional small floor guards against zero-prominence plateaus.
    prominence = max(prominence, 0.02)
    return noise_height, prominence, secondary


def pair_is_correct(peaks, t, true_rs, true_taus):
    if len(peaks) != 2:
        return False
    rs = np.array([p[0] for p in peaks])
    taus = np.array([t[p[1]] for p in peaks])
    true_rs = np.asarray(true_rs)
    true_taus = np.asarray(true_taus)
    dr = abs(rs[:, None]-true_rs[None, :])
    dt = abs(taus[:, None]-true_taus[None, :])
    cost = dr/RANGE_TOL_M + dt/(TAU_TOL_SAMPLES*SYSTEM.dt)
    i, j = linear_sum_assignment(cost)
    return bool(np.all(dr[i, j] <= RANGE_TOL_M) and
                np.all(dt[i, j] <= TAU_TOL_SAMPLES*SYSTEM.dt))


def plot_maps(maps, path):
    fig, axes = plt.subplots(2, 3, figsize=(15.6, 8.4), layout="constrained")
    for row, waveform in enumerate(("Gaussian", "RRC-QAM")):
        for col, B in enumerate(BANDWIDTHS_GHZ):
            score, t, info = maps[(waveform, B)]
            ax = axes[row, col]
            tau_ps = (t-R1_M/C)*1e12
            db = 10*np.log10(np.maximum(score/score.max(), 1e-15))
            im = ax.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                           extent=(RANGES_M[0], RANGES_M[-1],
                                   tau_ps[0], tau_ps[-1]), vmin=-15, vmax=0)
            ax.plot([R1_M, R2_M], [0, ILLUSTRATION_GAP_PS], "x",
                    color="#23d7b0", ms=8, mew=2)
            ax.set_title(f"{waveform}: {B:g} GHz symbol rate" if row else
                         f"Gaussian: B={B:g} GHz", fontsize=14)
            ax.set_xlabel("Candidate range (m)")
            if col == 0:
                ax.set_ylabel("Arrival relative to source 1 (ps)")
            if row == 0:
                ax.set_ylim(-450, 600)
            else:
                ax.set_ylim(-2200, 2400)
    fig.colorbar(im, ax=axes, shrink=0.8,
                 label="HRT power (dB below each panel maximum)")
    fig.suptitle("Raw HRT, same received energy per source; recorded time samples only")
    fig.savefig(path, dpi=170, facecolor="white")
    plt.close(fig)

    # Separate slide-sized figures avoid squeezing six maps onto one slide.
    for row, (waveform, filename) in enumerate((
        ("Gaussian", "gaussian_bandwidth_hrt.png"),
        ("RRC-QAM", "rrc_qam_bandwidth_hrt.png"),
    )):
        figure, panels = plt.subplots(1, 3, figsize=(15.3, 4.5),
                                      sharex=True, sharey=True,
                                      layout="constrained")
        for panel, B in zip(panels, BANDWIDTHS_GHZ):
            score, t, _ = maps[(waveform, B)]
            tau_ps = (t-R1_M/C)*1e12
            db = 10*np.log10(np.maximum(score/score.max(), 1e-15))
            im = panel.imshow(db.T, origin="lower", aspect="auto", cmap="magma",
                              extent=(RANGES_M[0], RANGES_M[-1],
                                      tau_ps[0], tau_ps[-1]), vmin=-15, vmax=0)
            panel.plot([R1_M, R2_M], [0, ILLUSTRATION_GAP_PS], "x",
                       color="#23d7b0", ms=9, mew=2)
            panel.set_title(f"{B:g} GHz", fontsize=20)
            panel.set_xlabel("Candidate range (m)", fontsize=15)
            panel.set_ylim((-450, 600) if row == 0 else (-2200, 2400))
            panel.tick_params(labelsize=12)
        panels[0].set_ylabel("Arrival relative to source 1 (ps)", fontsize=14)
        figure.colorbar(im, ax=panels, shrink=0.85,
                        label="HRT power (dB below panel maximum)")
        if row == 0:
            title = "Gaussian pulses: Equal energy, duration varies with bandwidth"
        else:
            title = ("RRC-QAM: Equal energy, fixed 2 ns burst; "
                     "symbol rate changes")
            figure.supxlabel("Crosses mark burst centers; QAM peaks depend on symbols",
                             fontsize=11)
        figure.suptitle(title, fontsize=19)
        figure.savefig(OUTPUT / filename, dpi=190, facecolor="white")
        plt.close(figure)


def plot_detection(summary_rows, out):
    gap = [row["gap_ps"] for row in summary_rows]
    fig, ax = plt.subplots(figsize=(8, 4.8), layout="constrained")
    ax.plot(gap, [row["exact_two_rate"] for row in summary_rows], "o-",
            lw=2, label="Exactly two detected")
    ax.plot(gap, [row["both_correct_rate"] for row in summary_rows], "s-",
            lw=2, label="Both within range/time tolerances")
    ax.set(xlabel="True center-arrival gap (ps)", ylabel="Fraction of trials",
           ylim=(-0.04, 1.04), xlim=(min(gap), max(gap)),
           title="8 GHz Gaussian: Blind Peak Detection, No Time Interpolation")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(out, dpi=190, facecolor="white")
    plt.close(fig)


def broader_study(hrt, x, variance, noise_height, prominence_fraction):
    """Random broadside ranges, gaps, source strength, and carrier phases."""
    rng = np.random.default_rng(SEED + 520)
    trial_rows = []
    tau_center = R1_M / C
    for trial in range(N_GENERAL_MC):
        r1, r2 = rng.uniform(35.0, 70.0, size=2)
        gap_ps = float(rng.uniform(0.0, max(GAPS_PS)))
        tau1 = tau_center + float(rng.uniform(-40e-12, 40e-12))
        tau2 = tau1 + gap_ps*1e-12
        weaker_db = float(rng.uniform(0.0, 6.0))
        a2 = 10**(-weaker_db/20)
        phases = rng.uniform(-np.pi, np.pi, size=2)
        signal = gaussian_source(hrt.t, x, r1, tau1, 8.0, phases[0])
        signal += a2*gaussian_source(hrt.t, x, r2, tau2, 8.0, phases[1])
        observed = add_noise(signal, rng, variance)
        peaks, _ = profile_peaks(hrt.scores(observed), noise_height,
                                prominence_fraction)
        trial_rows.append(dict(
            trial=trial, range1_m=r1, range2_m=r2,
            range_gap_m=float(abs(r1-r2)), arrival_gap_ps=gap_ps,
            second_source_attenuation_db=weaker_db,
            estimated_count=len(peaks), exact_two=int(len(peaks)==2),
            both_correct=int(pair_is_correct(
                peaks, hrt.t, (r1, r2), (tau1, tau2))),
            peak_ranges_m=json.dumps([p[0] for p in peaks]),
            peak_times_s=json.dumps([float(hrt.t[p[1]]) for p in peaks]),
        ))
        if (trial+1) % 50 == 0:
            write_csv(OUTPUT / "random_pair_trials.csv", trial_rows)
            print(f"Random pairs {trial+1}/{N_GENERAL_MC}", flush=True)
    write_csv(OUTPUT / "random_pair_trials.csv", trial_rows)
    bins = [0, 64, 128, 192, 257]
    grouped = []
    for low, high in zip(bins[:-1], bins[1:]):
        group = [r for r in trial_rows if low <= r["arrival_gap_ps"] < high]
        grouped.append(dict(gap_min_ps=low, gap_max_ps=high,
                            n=len(group),
                            exact_two_rate=float(np.mean([
                                r["exact_two"] for r in group])) if group else None,
                            both_correct_rate=float(np.mean([
                                r["both_correct"] for r in group])) if group else None))
    write_csv(OUTPUT / "random_pair_by_gap.csv", grouped)
    return trial_rows, grouped


def independent_null_study(hrt, x, variance, noise_height, prominence_fraction):
    """Evaluate zero- and one-source false alarms on held-out records."""
    rng = np.random.default_rng(SEED + 840)
    rows = []
    for trial in range(N_SINGLE_MC):
        r = float(rng.uniform(35.0, 70.0))
        tau = R1_M/C + float(rng.uniform(-80e-12, 80e-12))
        phase = float(rng.uniform(-np.pi, np.pi))
        single_clean = gaussian_source(hrt.t, x, r, tau, 8.0, phase)
        for count, clean in ((0, np.zeros_like(single_clean)),
                             (1, single_clean)):
            observed = add_noise(clean, rng, variance)
            peaks, _ = profile_peaks(hrt.scores(observed), noise_height,
                                    prominence_fraction)
            good_one = (count == 1 and len(peaks) == 1 and
                        abs(peaks[0][0]-r) <= RANGE_TOL_M and
                        abs(hrt.t[peaks[0][1]]-tau) <=
                        TAU_TOL_SAMPLES*SYSTEM.dt)
            rows.append(dict(trial=trial, true_count=count,
                             true_range_m=r if count else None,
                             estimated_count=len(peaks),
                             correct_single=int(good_one) if count else None,
                             peak_ranges_m=json.dumps([p[0] for p in peaks])))
        if (trial+1) % 50 == 0:
            write_csv(OUTPUT / "null_trials.csv", rows)
            print(f"Held-out 0/1-source nulls {trial+1}/{N_SINGLE_MC}",
                  flush=True)
    write_csv(OUTPUT / "null_trials.csv", rows)
    zero = [r for r in rows if r["true_count"] == 0]
    one = [r for r in rows if r["true_count"] == 1]
    return dict(noise_only_false_alarm_rate=float(np.mean([
                    z["estimated_count"] > 0 for z in zero])),
                one_source_false_split_rate=float(np.mean([
                    z["estimated_count"] >= 2 for z in one])),
                one_source_correct_count_rate=float(np.mean([
                    z["estimated_count"] == 1 for z in one])),
                one_source_correct_location_rate=float(np.mean([
                    z["correct_single"] for z in one])))


def ranges_match(peaks, true_rs=(R1_M, R2_M)):
    if len(peaks) != 2:
        return False
    dr = abs(np.array([p[0] for p in peaks])[:, None]
             - np.asarray(true_rs)[None, :])
    i, j = linear_sum_assignment(dr)
    return bool(np.all(dr[i, j] <= RANGE_TOL_M))


def qam_fixed_gap_study(hrt, x, variance):
    """Independent ordinary QAM sequences; calibrate each symbol rate's null."""
    rng = np.random.default_rng(SEED + 1230)
    rows, stats = [], []
    for B in BANDWIDTHS_GHZ:
        noise_max, secondary = [], []
        for k in range(N_QAM_CALIBRATION):
            empty = add_noise(np.zeros((len(x), len(hrt.t)), complex),
                              rng, variance)
            noise_max.append(float(hrt.scores(empty).max()))
            one, _ = qam_pair(
                hrt.t, x, B, ILLUSTRATION_GAP_PS,
                symbol_seed=SEED+2100+2*k, active_sources=(k % 2,),
                phases=rng.uniform(-np.pi, np.pi, size=2),
            )
            one_score = hrt.scores(add_noise(one, rng, variance))
            secondary.append(maximum_secondary_prominence_fraction(
                one_score.max(axis=0)))
        height = float(np.quantile(noise_max, NOISE_MAX_QUANTILE))
        prominence = max(0.02, float(np.quantile(
            secondary, CALIBRATED_FALSE_SPLIT_QUANTILE)))
        group = []
        for k in range(N_QAM_MC):
            symbol_seed = SEED+5100+k
            # Paired phases and noise across B; QAM symbol counts change with
            # the symbol rate, but each rate starts from the same RNG seed.
            rng_trial = np.random.default_rng(SEED+39000+k)
            phases = rng_trial.uniform(-np.pi, np.pi, size=2)
            common_noise = add_noise(np.zeros((len(x), len(hrt.t)), complex),
                                     rng_trial, variance)
            both, _ = qam_pair(hrt.t, x, B, ILLUSTRATION_GAP_PS,
                               symbol_seed=symbol_seed, phases=phases)
            one, _ = qam_pair(hrt.t, x, B, ILLUSTRATION_GAP_PS,
                              symbol_seed=symbol_seed, phases=phases,
                              active_sources=(k % 2,))
            for kind, clean in (("single", one), ("pair", both)):
                peaks, _ = profile_peaks(hrt.scores(clean+common_noise),
                                        height, prominence)
                row = dict(bandwidth_ghz=B, trial=k, kind=kind,
                           estimated_count=len(peaks),
                           exact_two=int(len(peaks)==2) if kind=="pair" else None,
                           both_ranges_correct=int(ranges_match(peaks))
                           if kind=="pair" else None,
                           peak_ranges_m=json.dumps([p[0] for p in peaks]))
                rows.append(row)
                group.append(row)
            if (k+1) % 50 == 0:
                write_csv(OUTPUT / "qam_fixed_gap_trials.csv", rows)
                print(f"QAM {B:g} GHz: {k+1}/{N_QAM_MC}", flush=True)
        single = [z for z in group if z["kind"] == "single"]
        pair = [z for z in group if z["kind"] == "pair"]
        stat = dict(symbol_rate_ghz=B, n_pair=len(pair),
                    exact_two_rate=float(np.mean([z["exact_two"] for z in pair])),
                    both_ranges_correct_rate=float(np.mean([
                        z["both_ranges_correct"] for z in pair])),
                    independent_single_source_false_split_rate=float(np.mean([
                        z["estimated_count"] >= 2 for z in single])),
                    calibrated_prominence_fraction=prominence,
                    calibrated_noise_height=height)
        stats.append(stat)
        print("QAM result:", stat, flush=True)
    write_csv(OUTPUT / "qam_fixed_gap_trials.csv", rows)
    write_csv(OUTPUT / "qam_fixed_gap_summary.csv", stats)
    return stats


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    started = time.perf_counter()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    x = np.asarray(ARRAY.positions(SYSTEM), float)
    dt = float(SYSTEM.dt)
    tau1 = R1_M/C
    variance, ref_r, ref_snr = reference_noise(x)
    rng_demo = np.random.default_rng(SEED + 1)
    maps, meta = {}, []
    print(f"Config={ROOT / 'config.py'}, Nx={len(x)}, dt={dt*1e12:.3f} ps, "
          f"reference peak SNR={ref_snr:g} dB at {ref_r:g} m; "
          f"fixed noise variance={variance:.5g}", flush=True)

    # Fixed observation windows within each waveform family; identical noise
    # realization across its three bandwidth cases. No model receives tau1.
    g_margin = 8*bw_sigma(min(BANDWIDTHS_GHZ)) + 0.4e-9
    g_time = axis(tau1, g_margin, g_margin+ILLUSTRATION_GAP_PS*1e-12, dt)
    q_margin = 1.5*QAM_DURATION_S + 0.4e-9
    q_time = axis(tau1, q_margin, q_margin+ILLUSTRATION_GAP_PS*1e-12, dt)
    for kind, t in (("Gaussian", g_time), ("RRC-QAM", q_time)):
        hrt = BroadsideHRT(t, x, RANGES_M)
        noise = add_noise(np.zeros((len(x), len(t)), complex),
                          rng_demo, variance)
        for B in BANDWIDTHS_GHZ:
            tic = time.perf_counter()
            if kind == "Gaussian":
                clean = (gaussian_source(t, x, R1_M, tau1, B)
                         + gaussian_source(t, x, R2_M,
                                           tau1+ILLUSTRATION_GAP_PS*1e-12, B))
                info = [dict(source=1, sigma_ps=bw_sigma(B)*1e12),
                        dict(source=2, sigma_ps=bw_sigma(B)*1e12)]
            else:
                clean, info = qam_pair(t, x, B, ILLUSTRATION_GAP_PS)
            score = hrt.scores(clean+noise)
            maps[(kind, B)] = (score, t, info)
            meta.append(dict(waveform=kind, nominal_bandwidth_ghz=B,
                             waveform_info=info, elapsed_s=time.perf_counter()-tic))
            print(f"Bandwidth map {kind} {B:g} GHz: "
                  f"{time.perf_counter()-tic:.2f}s; {info}", flush=True)
        del hrt
    plot_maps(maps, OUTPUT / "bandwidth_hrt_maps.png")
    del maps

    # 8 GHz sweep: one common time axis for every gap, even zero gap; same
    # source positions, energy, noise realization, and unknown TX phases for
    # a given trial across every gap (paired Monte Carlo).
    sweep_margin = 8*bw_sigma(8.0) + 0.22e-9
    sweep_time = axis(tau1, sweep_margin,
                      sweep_margin+max(GAPS_PS)*1e-12, dt)
    hrt = BroadsideHRT(sweep_time, x, RANGES_M)
    noise_height, prominence_fraction, null_secondary = calibrate_detector(
        hrt, x, variance)
    print(f"Fixed detector: noise height={noise_height:.5g}; prominence "
          f"fraction={prominence_fraction:.3f} "
          f"(single-source 95th percentile), min time separation "
          f"{MIN_PEAK_DISTANCE_SAMPLES} samples", flush=True)
    rng_phase = np.random.default_rng(SEED + 18)
    rng_noise = np.random.default_rng(SEED + 19)
    trials = []
    per_gap = {gap: [] for gap in GAPS_PS}
    for trial in range(N_MC):
        phases = rng_phase.uniform(-np.pi, np.pi, size=2)
        common_noise = add_noise(np.zeros((len(x), len(sweep_time)), complex),
                                 rng_noise, variance)
        first = gaussian_source(sweep_time, x, R1_M, tau1, 8.0, phases[0])
        for gap in GAPS_PS:
            tau2 = tau1+gap*1e-12
            second = gaussian_source(sweep_time, x, R2_M, tau2, 8.0,
                                     phases[1])
            score = hrt.scores(first+second+common_noise)
            peaks, _ = profile_peaks(score, noise_height, prominence_fraction)
            row = dict(trial=trial, gap_ps=gap,
                       estimated_count=len(peaks),
                       exact_two=int(len(peaks)==2),
                       both_correct=int(pair_is_correct(
                           peaks, sweep_time, (R1_M, R2_M), (tau1, tau2))),
                       peak_ranges_m=json.dumps([p[0] for p in peaks]),
                       peak_arrivals_ps=json.dumps([
                           round((sweep_time[p[1]]-tau1)*1e12, 3)
                           for p in peaks]),
                       peak_powers=json.dumps([p[2] for p in peaks]))
            trials.append(row)
            per_gap[gap].append(row)
        if (trial+1) % 10 == 0:
            write_csv(OUTPUT / "sweep_trials.csv", trials)
            print(f"{trial+1}/{N_MC} trials; {time.perf_counter()-started:.1f}s",
                  flush=True)
    summary_rows = []
    for gap in GAPS_PS:
        r = per_gap[gap]
        entry = dict(gap_ps=gap, n=len(r),
                     exact_two_rate=float(np.mean([z["exact_two"] for z in r])),
                     both_correct_rate=float(np.mean([z["both_correct"] for z in r])),
                     mean_detected_count=float(np.mean([
                         z["estimated_count"] for z in r])))
        summary_rows.append(entry)
        print(f"{gap:3g} ps: exactly two {entry['exact_two_rate']:.1%}; "
              f"both correct {entry['both_correct_rate']:.1%}; "
              f"mean count {entry['mean_detected_count']:.2f}", flush=True)
    write_csv(OUTPUT / "sweep_trials.csv", trials)
    write_csv(OUTPUT / "sweep_summary.csv", summary_rows)
    plot_detection(summary_rows, OUTPUT / "separation_detection.png")
    print("Running independent random-pair and null studies", flush=True)
    general_rows, general_by_gap = broader_study(
        hrt, x, variance, noise_height, prominence_fraction)
    null_metrics = independent_null_study(
        hrt, x, variance, noise_height, prominence_fraction)
    print("Random pairs:", general_by_gap, flush=True)
    print("Held-out null controls:", null_metrics, flush=True)
    print("Running ordinary QAM fixed-gap MC across symbol rates", flush=True)
    qam_hrt = BroadsideHRT(q_time, x, RANGES_M)
    qam_stats = qam_fixed_gap_study(qam_hrt, x, variance)
    summary = dict(config_path=str(ROOT / "config.py"), seed=SEED,
                   antennas=len(x), sample_period_ps=dt*1e12,
                   time_oversampling=TIME_OVERSAMPLE, n_mc_per_gap=N_MC,
                   n_random_pairs=N_GENERAL_MC, n_null_trials_per_count=N_SINGLE_MC,
                   n_qam_per_bandwidth=N_QAM_MC,
                   n_calibration=N_CALIBRATION, search="full R=30..75m, "
                   "theta=0 slice, all recorded times, no truth hints to detector",
                   waveform_template_used=False, matched_filter=False,
                   reference_range_m=ref_r, reference_peak_snr_db=ref_snr,
                   complex_noise_variance=variance,
                   bandwidth_maps=meta,
                   detector=dict(method="scipy.signal.find_peaks on max-over-range "
                                 "HRT time profile; range at each time peak",
                                 noise_height=noise_height,
                                 prominence_fraction=prominence_fraction,
                                 minimum_peak_distance_samples=MIN_PEAK_DISTANCE_SAMPLES,
                                 calibrated_single_source_secondary_prominences=
                                 null_secondary),
                   match_tolerance=dict(range_m=RANGE_TOL_M,
                                        time_samples=TAU_TOL_SAMPLES),
                   random_pairs=dict(exact_two_rate=float(np.mean([
                       r["exact_two"] for r in general_rows])),
                       both_correct_rate=float(np.mean([
                           r["both_correct"] for r in general_rows])),
                       by_arrival_gap=general_by_gap),
                   null_controls=null_metrics,
                   qam_fixed_gap=qam_stats,
                   elapsed_s=time.perf_counter()-started)
    (OUTPUT / "summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(f"Saved results in {OUTPUT}; elapsed {summary['elapsed_s']:.1f}s",
          flush=True)


if __name__ == "__main__":
    main()
