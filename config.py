from dataclasses import dataclass
import numpy as np

C = 299_792_458.0


@dataclass(frozen=True)
class SystemConfig:
    fc: float = 24e9
    bandwidth: float = 8e9
    oversamp: int = 8

    @property
    def fs(self) -> float:
        return self.oversamp * self.bandwidth

    @property
    def dt(self) -> float:
        return 1.0 / self.fs

    @property
    def wavelength(self) -> float:
        return C / self.fc

    @property
    def gaussian_sigma(self) -> float:
        return 1.0 / (np.sqrt(2.0) * np.pi * self.bandwidth)


@dataclass(frozen=True)
class ArrayConfig:
    num_antennas: int = 513

    def positions(self, system: SystemConfig) -> np.ndarray:
        spacing = system.wavelength / 2.0
        idx = np.arange(self.num_antennas) - (self.num_antennas - 1) / 2.0
        return idx * spacing


@dataclass(frozen=True)
class PRTConfig:
    """Shared PRT-axis discretization used by the main ROI workflow."""
    theta_step_deg: float = 0.1
    dq: float = 1e-13

    @property
    def tau_step(self) -> float:
        # The PRT tau axis is sampled on the signal time grid.
        return SYSTEM.dt


@dataclass(frozen=True)
class SimulationConfig:
    snr_db: float = 10.0
    pulse_span_sigma: float = 6.0
    rng_seed: int = 0


SYSTEM = SystemConfig()
ARRAY = ArrayConfig()
SIM = SimulationConfig()
PRT = PRTConfig()