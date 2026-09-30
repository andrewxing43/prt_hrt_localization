import numpy as np
import matplotlib.pyplot as plt


def plot_q_slice(q_grid: np.ndarray, prt: np.ndarray, tau_idx: int, p_idx: int,
                 q_true: float | None = None, q_hat: float | None = None) -> None:
    """Plot |PRT| versus q for a fixed tau and p index."""
    magnitude = np.abs(prt[tau_idx, p_idx, :])

    plt.figure()
    plt.plot(q_grid, magnitude)

    if q_true is not None:
        plt.axvline(q_true, linestyle="--", label="True q")

    if q_hat is not None:
        plt.axvline(q_hat, linestyle=":", label="Estimated q")

    plt.xlabel("q (s/m²)")
    plt.ylabel("|PRT|")
    plt.title("PRT q-slice")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_range_error(range_true: np.ndarray, range_error: np.ndarray) -> None:
    """Plot range estimation error versus true range."""
    plt.figure()
    plt.plot(range_true, range_error, marker="o")
    plt.xlabel("True range (m)")
    plt.ylabel("Range error (m)")
    plt.title("PRT Range Estimation Error")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_range_error_map(ranges: np.ndarray, angles: np.ndarray,
                         error_map: np.ndarray) -> None:
    """Plot range error over a range-angle grid."""
    plt.figure()
    image = plt.imshow(
        error_map,
        origin="lower",
        aspect="auto",
        extent=[ranges[0], ranges[-1], angles[0], angles[-1]]
    )

    plt.colorbar(image, label="Range error (m)")
    plt.xlabel("True range (m)")
    plt.ylabel("True angle (deg)")
    plt.title("PRT Range Error Map")
    plt.tight_layout()
    plt.show()