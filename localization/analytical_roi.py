import numpy as np
from scipy.integrate import quad
from scipy.optimize import root_scalar
from scipy.constants import speed_of_light as c
from localization.roi import ROI
from localization.parameter_mapping import p_to_theta, pq_to_range

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

def finite_bw_bifurcation_equation(gamma, rho):
    """
    find A, A', and A'' to evaluate G' wrt beta at w = 0, beta = 0
    """
    def integrand_A(y):
        s, _, _ = sinc_derivatives(rho * gamma * y)
        phase = 2 * np.pi * gamma * y
        return s * np.exp(1j * phase)
        
    def integrand_A_b(y):
        s, sp, _ = sinc_derivatives(rho * gamma * y)
        return (y**2) * (rho * sp + 1j * 2 * np.pi * s) * np.exp(1j * 2 * np.pi * gamma * y)
        
    def integrand_A_bb(y):
        s, sp, spp = sinc_derivatives(rho * gamma * y)
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
        s, _, _ = sinc_derivatives(w + rho * (gamma * y + beta * y**2))
        phase = 2 * np.pi * (gamma * y + beta * y**2)
        return s * np.exp(1j * phase)
        
    def integrand_A_b(y):
        s, sp, _ = sinc_derivatives(w + rho * (gamma * y + beta * y**2))
        phase = 2 * np.pi * (gamma * y + beta * y**2)
        term = rho * sp + 1j * 2 * np.pi * s
        return (y**2) * term * np.exp(1j * phase)
        
    def integrand_A_bb(y):
        s, sp, spp = sinc_derivatives(w + rho * (gamma * y + beta * y**2))
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


def get_roi(L, f0, B, Np, p_peak, q_peak, tau_peak, oversampling=2.0):
    """
    Establish maximum q displacement given the maximum p and tau errors to narrow HRT
    Return as an ROI object centered around the detected PRT argmax
    """
    # system parameters
    rho = B / f0

    # time quantization + error based on oversampling rate
    w_max = 1/(2*oversampling)
    dt = 1 / (oversampling * 2 * B)

    # slowness grid (Np from w=0 case is 306)
    dp = (2.0 / c) / (Np - 1)
    gamma_max = f0 * L * dp / 2.0

    # search only the worst case tau/p errors
    corners = [
        (w_max, gamma_max),
        (w_max, -gamma_max),
        (-w_max, gamma_max),
        (-w_max, -gamma_max)
    ]
    
    max_beta_shift = 0.0

    # solve over different combinations of of omega/gamma
    for w_val, g_val in corners:
        # start near the previous guess B_star 
        # the peak hopefully shouldnt move much between diff values of omega >= 0
        b_opt, gbb_opt, success = solve_beta_star(w_val, g_val, rho, guess=0.0)
        
        if success and gbb_opt < 0:
            max_beta_shift = max(max_beta_shift, abs(b_opt))

    # largest beta (peak) displacement observed in our grid
    E_beta = max_beta_shift

    # largest q displacement observed in the grid
    E_q = E_beta / (f0 * L**2)

    p_min_bound = p_peak - (dp/2.0)
    p_max_bound = p_peak + (dp/2.0)

    # convert p bounds to theta bounds for HRT
    theta_1 = p_to_theta(p_min_bound)
    theta_2 = p_to_theta(p_max_bound)
    
    theta_min = min(theta_1, theta_2)
    theta_max = max(theta_1, theta_2)

    # get q bounds
    q_min_bound = q_peak - E_q
    q_max_bound = q_peak + E_q
    
    # handle far-field/negative q edge cases
    if q_min_bound <= 0:
        R_max = float('inf')
    else:
        R_max = pq_to_range(p_peak, q_min_bound)
        
    if q_max_bound <= 0:
        R_min = 0.0
    else:
        R_min = pq_to_range(p_peak, q_max_bound)
        
    if R_min > R_max:
        R_min, R_max = R_max, R_min

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