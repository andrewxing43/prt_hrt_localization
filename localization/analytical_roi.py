import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.constants import speed_of_light as c
from localization.roi import ROI
from localization.parameter_mapping import p_to_theta_deg, pq_to_range

def sinc_derivatives(u):
    """
    get sinc(u), sinc'(u), sinc''(u)
    
    use Taylor expansion near u = 0 to prevent div by 0 later
    """
    if abs(u) < 1e-4:

        # first two terms of sinc taylor expansion around 0
        s = 1.0 - (np.pi**2 / 6.0) * u**2
        sp = -(np.pi**2 / 3.0) * u
        spp = -(np.pi**2 / 3.0)
    else:
        pi_u = np.pi * u
        s = np.sin(pi_u) / pi_u
        sp = (np.cos(pi_u) - s) / u
        spp = -(np.pi**2) * s - 2.0 * sp / u
    return s, sp, spp

def gaussian_derivatives(u):
    """
    Get first second and third derivatives of the Gaussian pulse

    Use normalized gaussian autocorrelation (post MF) as the "pulse" to differentiate
    """
    s = np.exp(-0.5 * np.pi**2 * u**2)
    sp = -np.pi**2 * u * s
    spp = (np.pi**4 * u**2 - np.pi**2) * s
    return s, sp, spp

def finite_bw_bifurcation_equation(gamma, rho):
    """
    find A, A', and A'' to evaluate G' wrt beta at w = 0, beta = 0
    """
    def integrand_A(y):
        s, _, _ = gaussian_derivatives(rho * gamma * y)
        phase = 2 * np.pi * gamma * y
        return s * np.exp(1j * phase)
        
    def integrand_A_b(y):
        s, sp, _ = gaussian_derivatives(rho * gamma * y)
        return (y**2) * (rho * sp + 1j * 2 * np.pi * s) * np.exp(1j * 2 * np.pi * gamma * y)
        
    def integrand_A_bb(y):
        s, sp, spp = gaussian_derivatives(rho * gamma * y)
        return (y**4) * (rho**2 * spp + 1j * 4 * np.pi * rho * sp - 4 * np.pi**2 * s) * np.exp(1j * 2 * np.pi * gamma * y)

    def complex_quad(func, a, b):
        """
        Use quad to integrate the complex valued functions on a interval
        """
        real_part, _ = quad(lambda y: np.real(func(y)), a, b)
        imag_part, _ = quad(lambda y: np.imag(func(y)), a, b)
        return real_part + 1j * imag_part

    # find the integrals
    A = complex_quad(integrand_A, -0.5, 0.5)
    A_b = complex_quad(integrand_A_b, -0.5, 0.5)
    A_bb = complex_quad(integrand_A_bb, -0.5, 0.5)
    
    # G_bb = 2 * ( |A_b|^2 + Re{A_bb * A^*} )
    G_bb = 2 * (np.abs(A_b)**2 + np.real(A_bb * np.conj(A)))
    
    return G_bb

def get_np(L, f0, B):
    """
    Get npx for the PRT slowness grid
    This can be done offline 
    """
    lam = c / f0 
    rho = B / f0

    # same thing but for the finite BW 
    # find where second derivative of G = 0
    sol_fb = root_scalar(finite_bw_bifurcation_equation, args=(rho,), bracket=[0.1, 1.0], method='brentq')
    gamma_c_fb = sol_fb.root

    # solve for np -> Np > 1 + (L / lambda) / gamma_c
    Np_exact_fb = 1 + (L / lam) / gamma_c_fb
    Np_fb = int(np.ceil(Np_exact_fb))

    return Np_fb

def eval_G_derivatives(w, gamma, beta, rho):
    """
    Find A, A', and A'' to evaluate G' wrt beta at beta = 0
    """
    def integrand_A(y):
        s, _, _ = gaussian_derivatives(w + rho * (gamma * y + beta * y**2))
        phase = 2 * np.pi * (gamma * y + beta * y**2)
        return s * np.exp(1j * phase)
        
    def integrand_A_b(y):
        s, sp, _ = gaussian_derivatives(w + rho * (gamma * y + beta * y**2))
        phase = 2 * np.pi * (gamma * y + beta * y**2)
        term = rho * sp + 1j * 2 * np.pi * s
        return (y**2) * term * np.exp(1j * phase)
        
    def integrand_A_bb(y):
        s, sp, spp = gaussian_derivatives(w + rho * (gamma * y + beta * y**2))
        phase = 2 * np.pi * (gamma * y + beta * y**2)
        term = rho**2 * spp + 1j * 4 * np.pi * rho * sp - 4 * np.pi**2 * s
        return (y**4) * term * np.exp(1j * phase)

    def complex_quad(func, a, b):
        """
        Use quad to integrate the complex valued functions on a interval
        """
        real_part, _ = quad(lambda y: np.real(func(y)), a, b)
        imag_part, _ = quad(lambda y: np.imag(func(y)), a, b)
        return real_part + 1j * imag_part

    # find the integrals
    A = complex_quad(integrand_A, -0.5, 0.5)
    A_b = complex_quad(integrand_A_b, -0.5, 0.5)
    A_bb = complex_quad(integrand_A_bb, -0.5, 0.5)

    G_b = 2 * np.real(A_b * np.conj(A))
    # G_bb = 2 * ( |A_b|^2 + Re{A_bb * A^*} )
    G_bb = 2 * (np.abs(A_b)**2 + np.real(A_bb * np.conj(A)))
    
    return G_b, G_bb

def solve_beta_star(w_val, g_val, rho, guess=0.0, initial_step=0.05, max_radius=1.5):
    """
    Looking for the new q peak location beta*
    """
    def f(beta_val):
        G_b, _ = eval_G_derivatives(w_val, g_val, beta_val, rho)
        return G_b

    radius = initial_step

    while radius <= max_radius:
        left = guess - radius
        right = guess + radius
        f_left = f(left)
        f_right = f(right)

        if np.isfinite(f_left) and np.isfinite(f_right):
            # guarantee a local maximum (increasing on the left and decreasing on the right of beta*)
            if f_left > 0 and f_right < 0:
                res = root_scalar(f, bracket=(left, right), method='brentq', maxiter=50)

                if res.converged:
                    beta_opt = res.root

                    # G'' WRT beta_opt
                    _, gbb_opt = eval_G_derivatives(w_val, g_val, beta_opt, rho)
                    return beta_opt, gbb_opt, True

        radius *= 1.5

    return guess, np.nan, False

def get_error_margins(L, f0, B, Np, oversampling=2.0, n_grid=21):
    """
    Get the numerical error bound on E_q before running any RTs
    Since this is source independent, it can be done offline based solely off dp/dt
    The dq/2 allowance assumes selection of a nearest q bin.
    """
    # system parameters
    rho = B / f0

    # time quantization + error based on oversampling rate
    dt = 1.0 / (oversampling * B)
    w_max = B * dt / 2.0

    # slowness grid
    dp = (2.0 / c) / (Np - 1)
    gamma_max = f0 * L * dp / 2.0

    # get the worst case E_q
    w_values = np.linspace(-w_max, w_max, n_grid)
    gamma_values = np.linspace(-gamma_max, gamma_max, n_grid)
    max_beta_shift = 0.0

    # solve over different combinations of of omega/gamma
    for w_val in w_values:
        for g_val in gamma_values:

            # start w/guess of no error (beta = 0)
            b_opt, gbb_opt, success = solve_beta_star(w_val, g_val, rho, guess=0.0)
            if not success or not np.isfinite(gbb_opt) or gbb_opt >= 0:
                raise RuntimeError(f"No resolved local maximum at w={w_val}, gamma={g_val}; refine Np or inspect the beta search")
            
            # largest beta (peak) displacement observed in our grid
            max_beta_shift = max(max_beta_shift, abs(b_opt))

    # largest q displacement observed in the grid
    # NOT YET accounting for dq grid
    E_q_cont = max_beta_shift / (f0 * L**2) 
    return E_q_cont, dp, dt

def get_p_cell_bounds(p_peak, p_grid, theta_bounds=None):
    """
    Get the nearest-node cell of p_peak on the p grid
    """
    p_sorted = np.sort(p_grid)
    peak_idx = int(np.argmin(np.abs(p_sorted - p_peak)))

    p_min_bound = p_sorted[0] if peak_idx == 0 else 0.5 * (p_sorted[peak_idx - 1] + p_sorted[peak_idx])
    p_max_bound = p_sorted[-1] if peak_idx == len(p_sorted) - 1 else 0.5 * (p_sorted[peak_idx] + p_sorted[peak_idx + 1])

    if theta_bounds is not None:
        theta_p = -np.sin(np.deg2rad(np.asarray(theta_bounds, dtype=float))) / c
        p_min_bound = max(p_min_bound, float(np.min(theta_p)))
        p_max_bound = min(p_max_bound, float(np.max(theta_p)))


    return float(p_min_bound), float(p_max_bound)


def get_roi(p_peak, q_peak, tau_peak, p_grid, E_q, dt, range_bounds=None, theta_bounds=None):
    """
    Given the maximum p and tau errors to narrow HRT, we can find maximum q displacement
    Return as R/theta/tau deta, i.e. an ROI object centered around the detected PRT argmax
    """
    p_min_bound, p_max_bound = get_p_cell_bounds(p_peak, p_grid, theta_bounds=theta_bounds)

    # convert p bounds to theta bounds for HRT
    theta_1 = p_to_theta_deg(p_min_bound)
    theta_2 = p_to_theta_deg(p_max_bound)

    # order correctly and clip to theta_bounds
    theta_min_calc = min(theta_1, theta_2)
    theta_max_calc = max(theta_1, theta_2)

    if theta_bounds is not None:
        theta_min = max(theta_min_calc, theta_bounds[0])
        theta_max = min(theta_max_calc, theta_bounds[1])
    else:
        theta_min = theta_min_calc
        theta_max = theta_max_calc

    # get q bounds
    q_min_bound = q_peak - E_q
    q_max_bound = q_peak + E_q
    
    # we want to check all p values that can produce the max or min range over that interval
    # if the interval crosses 0, p = 0 should be included
    p_candidates = np.array([p_min_bound, p_max_bound])
    if p_min_bound <= 0.0 <= p_max_bound:
        p_candidates = np.append(p_candidates, 0.0)

    # handle far-field/negative q edge cases
    if q_min_bound <= 1e-12:
        R_max_calc = float('inf')
    else:
        R_max_calc = float(np.max(pq_to_range(p_candidates, q_min_bound)))
        
    if q_max_bound <= 1e-12:
        R_min_calc = 0.0
    else:
        R_min_calc = float(np.min(pq_to_range(p_candidates, q_max_bound)))
        
    if R_min_calc > R_max_calc:
        R_min_calc, R_max_calc = R_max_calc, R_min_calc
    
    # clip to range_bounds
    if range_bounds is not None:
        R_min = max(R_min_calc, range_bounds[0])
        R_max = min(R_max_calc, range_bounds[1])
    else:
        R_min = R_min_calc
        R_max = R_max_calc

    # tau bounds based off time bin
    tau_min = tau_peak - (dt/2.0)
    tau_max = tau_peak + (dt/2.0)

    return ROI(
        R_min=float(R_min),
        R_max=float(R_max),
        theta_min=float(theta_min),
        theta_max=float(theta_max),
        tau_min=float(tau_min),
        tau_max=float(tau_max),
        W_R=float(R_max - R_min),
        W_theta=float(theta_max - theta_min),
        W_tau=float(tau_max - tau_min)
    )