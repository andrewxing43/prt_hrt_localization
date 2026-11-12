"""Independent RRC-shaped QAM sources using the Gaussian model's array and noise API.

This module generates a *clean* complex array. Pass that array to
``signal_model.noise.add_awgn`` to use the same fixed receiver noise as the
Gaussian experiments.
"""

from __future__ import annotations

import numpy as np

from config import C, SYSTEM, ARRAY, SIM
from signal_model.propagation import propagation_distances, propagation_delays


def root_raised_cosine(t_norm: np.ndarray, beta: float = 0.35) -> np.ndarray:
    """
    Unit-symbol-period RRC pulse including removable singularities
    """
    u = np.asarray(t_norm, dtype=float)
    if not 0 <= beta <= 1:
        raise ValueError("RRC rolloff beta must be between 0 and 1")
    if beta == 0:
        return np.sinc(u)

    result = np.empty_like(u)
    # t=0 special case
    at_zero = np.isclose(u, 0.0, rtol=0.0, atol=1e-12)
    # indexes w/asymptotes i.e. denom of rrc goes to 0
    at_pole = np.isclose(np.abs(u), 1.0 / (4.0 * beta), rtol=0.0, atol=1e-12)
    # all other indices
    regular = ~(at_zero | at_pole)
    result[at_zero] = 1.0 - beta + 4.0 * beta / np.pi
    result[at_pole] = beta / np.sqrt(2.0) * (
        (1.0 + 2.0 / np.pi) * np.sin(np.pi / (4.0 * beta))
        + (1.0 - 2.0 / np.pi) * np.cos(np.pi / (4.0 * beta))
    )
    v = u[regular]
    result[regular] = (
        np.sin(np.pi * v * (1.0 - beta))
        + 4.0 * beta * v * np.cos(np.pi * v * (1.0 + beta))
    ) / (np.pi * v * (1.0 - (4.0 * beta * v) ** 2))
    return result


class MultiSourceGen:
    """Generate multiple sources with independent QAM symbols and exact delays.

    sources - list of dictionaries with range_m, theta_deg
    optional: clock_offset, tx_power, symbols, wavefront='plane'
    carrier freq SYSTEM.fc, use the system ARRAY
    tx_power, if not provided, uses signal_model.propagation

    The returned metadata includes each source's transmitted complex waveform,
    its time axis, symbols, and antenna delays.
    """
    def __init__(self, pw_sec: float = 2e-9, M: int = 16,
                 beta: float = 0.35,
                 antenna_x: np.ndarray | None = None,
                 f0: float | None = None, seed: int | None = None):
        # bandwidth of signal
        self.bandwidth = float(SYSTEM.bandwidth)
        # sampling freq/period
        self.dt = float(SYSTEM.dt)
        # pulse width
        self.pw_sec = float(pw_sec)
        # n symbols
        self.n_symbols = int(round(self.pw_sec * self.bandwidth))
        # M-QAM symbols
        self.M = int(M)
        self.beta = float(beta)
        # signal carrier freq for phase shifts in RX BB
        self.f0 = float(f0 if f0 is not None else getattr(SYSTEM, "f0", SYSTEM.fc))
        # antenna array setup; spacing is dictated by SYSTEM.fc (array design center freq)
        self.antenna_x = np.asarray(
            ARRAY.positions(SYSTEM) if antenna_x is None else antenna_x, dtype=float
        )
        self.rng = np.random.default_rng(
            getattr(SIM, "rng_seed", 0) if seed is None else seed
        )
        self.half_samples = int(np.ceil(1.5 * self.pw_sec / self.dt))
        self.waveform_time = np.arange(-self.half_samples, self.half_samples + 1) * self.dt

        if self.n_symbols < 1:
            raise ValueError("pw_sec * SYSTEM.bandwidth must give at least one symbol")
        if self.M < 4 or int(np.sqrt(self.M)) ** 2 != self.M:
            raise ValueError("M must be a square QAM constellation of size at least 4")
        if self.antenna_x.ndim != 1 or not len(self.antenna_x):
            raise ValueError("antenna_x must be a nonempty one-dimensional array")

    def _qam_symbols(self, count: int, rng: np.random.Generator) -> np.ndarray:
        side = int(np.sqrt(self.M))
        # normalize s.t. Es = 1 i.e. A = sqrt(3/(2 * (M-1))) -> From Homework 2, Question 7
        scale = np.sqrt(3.0 / (2.0 * (self.M - 1)))
        # generate n random symbols
        in_phase = 2 * rng.integers(side, size=count) + 1 - side
        quadrature = 2 * rng.integers(side, size=count) + 1 - side
        return scale * (in_phase + 1j * quadrature)

    def _waveform(self, symbols: np.ndarray) -> np.ndarray:
        t = self.waveform_time
        symbol_times = np.arange(self.n_symbols)
        # shift so normalized time=0 aligns w/when symbol 'k' to shape each symbol individually w/ RRC appropriately
        u = t[None, :] * self.bandwidth + (self.n_symbols - 1) / 2 - symbol_times[:, None]
        # add the RRC pulse shaping at TX
        waveform = symbols @ root_raised_cosine(u, self.beta)
        # keep just the pulse w/the duration we want
        waveform[np.abs(t) > 1.5 * self.pw_sec] = 0
        return np.asarray(waveform, dtype=np.complex128)

    def _geometry(self, source: dict) -> tuple[np.ndarray, np.ndarray]:
        r = float(source["range_m"])
        theta = float(source["theta_deg"])
        offset = float(source.get("clock_offset", 0.0))
        if source.get("wavefront", "spherical") == "plane":
            # force linear arrivals for a plane wave sim
            distances = np.full(len(self.antenna_x), r)
            delays = r / C - np.sin(np.deg2rad(theta)) * self.antenna_x / C + offset
        else:
            # with true spherical propagation
            distances = propagation_distances(r, theta, antenna_x=self.antenna_x)
            delays = propagation_delays(r, theta, offset, antenna_x=self.antenna_x)
        return distances, delays

    def time_axis(self, sources: list[dict]) -> np.ndarray:
        """
        Cover every source arrival, every antenna, and the QAM pulse 
        """
        all_delays = [self._geometry(source)[1] for source in sources]
        # add padding so pulse crosses full array + doesn't get clipped + extra samples for various roundings relative to global n samples
        margin = 1.5 * self.pw_sec + 4 * self.dt
        first = int(np.floor((min(d.min() for d in all_delays) - margin) / self.dt))
        last = int(np.ceil((max(d.max() for d in all_delays) + margin) / self.dt))
        return np.arange(first, last + 1) * self.dt

    def received_signal(self, time_axis: np.ndarray, sources: list[dict],
                        rng: np.random.Generator | None = None,
                        include_path_loss: bool = True) -> tuple[np.ndarray, list[dict]]:
        """
        Return antenna vs time samples
        Every source uses one pulse waveform shared across its antennas. 
        The sources have independent QAM sequences.
        Apply fraction delay to envelope and carrier phase to shift for arrival to antennas.
        """
        t = np.asarray(time_axis, dtype=float)
        if rng is None:
            rng = self.rng

        # full correctly dimensioned blank matrix to deposit rx bb signal into
        rx = np.zeros((len(self.antenna_x), len(t)), dtype=np.complex128)
        metadata: list[dict] = []
        for source in sources:
            # generate M-QAM symbols, or use the supplied symbols for repeatable trials
            if "symbols" in source:
                symbols = np.asarray(source["symbols"], dtype=np.complex128)
                if symbols.shape != (self.n_symbols,):
                    raise ValueError("Each supplied symbol vector must have n_symbols entries")
            else:
                symbols = self._qam_symbols(self.n_symbols, rng)
            reference = self._waveform(symbols)

            distances, delays = self._geometry(source)
            center_arrival = float(source["range_m"]) / C + float(source.get("clock_offset", 0.0))
            # find local window for this source such that full PW of source crosses array
            shift_guard = int(np.ceil(np.max(np.abs(delays - center_arrival)) / self.dt)) + 4
            half = self.half_samples + shift_guard
            local_len = 2 * half + 1
            base = np.zeros(local_len, dtype=np.complex128)
            base[half-self.half_samples:half+self.half_samples+1] = reference

            # find where the source falls on the global map
            center_index = int(np.rint((center_arrival - t[0]) / self.dt))
            # get fractional error from previous rounding
            fractional_offset = center_arrival - (t[0] + center_index * self.dt)
            relative_delay = delays - center_arrival + fractional_offset
            freqs = np.fft.fftfreq(local_len, d=self.dt)
            # time delay in time domain = mult by exp in freq domain
            shifted = np.fft.ifft(
                np.fft.fft(base)[None, :]
                * np.exp(-2j * np.pi * relative_delay[:, None] * freqs[None, :]),
                axis=1,
            )

            power = source.get("tx_power", getattr(SIM, "tx_power", 1.0))
            if float(power) < 0:
                raise ValueError("tx_power must be nonnegative watts")
            gain = np.sqrt(float(power))
            if include_path_loss:
                gain = gain * SYSTEM.wavelength / (4 * np.pi * distances)
            # apply carrier phase shift and path loss and store for this element
            phase = np.exp(-2j * np.pi * self.f0 * delays)
            shifted *= (gain * phase)[:, None]

            # attach the local signal to the global rx bb matrix
            start = center_index - half
            # Boundary logic to prevent out-of-bounds array additions
            first = max(start, 0)
            last = min(start + local_len, len(t))
            if first < last:
                rx[:, first:last] += shifted[:, first-start:last-start]

            metadata.append(dict(
                range_m=float(source["range_m"]),
                theta_deg=float(source["theta_deg"]),
                clock_offset=float(source.get("clock_offset", 0.0)),
                delays=delays,
                symbols=symbols.copy(),
                waveform_time=self.waveform_time.copy(),
                waveform_reference=reference,
            ))
        return rx, metadata

    def generate(self, sources: list[dict],
                 rng: np.random.Generator | None = None,
                 include_path_loss: bool = True
                 ) -> tuple[np.ndarray, np.ndarray, list[dict]]:
        """
        Generate sources, time samples, and the metadata
        """
        t = self.time_axis(sources)
        rx, metadata = self.received_signal(t, sources, rng, include_path_loss)
        return rx, t, metadata
