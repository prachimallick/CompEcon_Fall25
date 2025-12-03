# %%
from __future__ import annotations


# %%
"""
PS8 - Fully fixed and cleaned script for replication & SMM

Usage:
    python ps8_fixed.py --test   # quick (fast) smoke test
    python ps8_fixed.py --full   # full two-step SMM (slow)
"""

import numpy as np
import pandas as pd
from scipy import optimize
import time
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
from scipy.stats import norm


# Names used in outputs

PARAM_NAMES = ["alpha", "gamma", "rho", "sigma", "phi0"]
MOMENT_NAMES = ["a1 (Q coeff)", "a2 (CF coeff)", "std(I/K)",
                "std(π/K)", "AR1(I/K)", "mean Q", "ext_frac"]


# Target moments (Table 3)

TARGET_MOMENTS = np.array([
    0.011,    # a1: Q coefficient
    0.145,    # a2: Cash flow coefficient
    0.081,    # std(I/K)
    0.164,    # std(π/K)
    0.322,    # AR1(I/K)
    1.268,    # mean Q
    0.390     # ext_frac
])

# Parameters dataclass
@dataclass
class Params:
    # Parameters to ESTIMATE 
    alpha: float = 0.6956
    gamma: float = 0.1331   
    rho: float = 0.0976
    sigma: float = 0.8932
    phi0: float = 0.0       

    # Calibrated parameters
    delta: float = 0.15
    beta: float = 0.95
    phi1: float = 0.0

    
    nK: int = 300
    Kmin: float = 0.01
    Kmax: float = 40.0
    nA: int = 7

    Nfirms: int = 200
    T: int = 50
    burn: int = 10
    random_seed: int = 12345

    # DP solver settings
    vfi_tol: float = 1e-6
    vfi_maxiter: int = 2000

    # output dir
    output_dir: str = "ps8_results"


# 3. TAUCHEN DISCRETIZATION
def norm_cdf(x):
    """Standard normal CDF (works for scalars or numpy arrays)."""
    return norm.cdf(x)

def tauchen(rho: float, sigma: float, m: float = 3.0, n: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    """
    Tauchen discretization for AR(1) process
    """
    std_x = sigma / np.sqrt(1 - rho ** 2)
    x_max = m * std_x
    x_min = -x_max
    x_grid = np.linspace(x_min, x_max, n)
    step = (x_max - x_min) / (n - 1)
    
    P = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if j == 0:
                P[i, j] = norm_cdf((x_grid[0] - rho * x_grid[i] + step / 2.0) / sigma)
            elif j == n - 1:
                P[i, j] = 1.0 - norm_cdf((x_grid[-1] - rho * x_grid[i] - step / 2.0) / sigma)
            else:
                upper = (x_grid[j] - rho * x_grid[i] + step / 2.0) / sigma
                lower = (x_grid[j] - rho * x_grid[i] - step / 2.0) / sigma
                P[i, j] = norm_cdf(upper) - norm_cdf(lower)
    
    # Ensure rows sum to 1
    P = P / P.sum(axis=1, keepdims=True)
    return x_grid, P


# %%
def solve_dp(params: Params, Kgrid: np.ndarray, 
             grid_i: np.ndarray, Agrid: np.ndarray, 
             P_A: np.ndarray) -> Dict[str, Any]:
    """
    Vectorized DP solver over choices of K' chosen from the same Kgrid.
    Returns V (nK x nA) and policy indices into Kgrid (nK x nA).
    """
    nK = len(Kgrid)
    nA = len(Agrid)
    
    # Ensure P_A shape is (nA, nA)
    P_A = np.atleast_2d(P_A)
    if P_A.shape != (nA, nA):
        raise ValueError(f"P_A has shape {P_A.shape} but expected ({nA},{nA})")
    
    # Precompute constants
    beta = params.beta
    delta = params.delta
    K_mean = float(np.mean(Kgrid))
    
    # initialize value and policy
    V = np.zeros((nK, nA))
    policy_idx = np.zeros((nK, nA), dtype=int)
    
    # Create arrays for vectorized K' choices:
    K_mat = Kgrid[:, None]            # shape (nK, 1)
    Kprime_mat = Kgrid[None, :]       # shape (1, nK)  (K' choices)
    
    # create A grid broadcast shapes: (1, nA)
    A_row = Agrid[None, :]            # (1, nA)
    
    for it in range(params.vfi_maxiter):
        
        EV = V.dot(P_A.T)   # (nK, nA)
        
        invest_mat = Kprime_mat - (1.0 - delta) * K_mat   
        pi_mat = (A_row * (K_mat ** params.alpha))      # (nK, nA)
        
        # Now expand to include K' choices:
        invest_expanded = invest_mat[:, None, :]        
        invest_expanded = np.repeat(invest_expanded, nA, axis=1)  
        pi_expanded = pi_mat[:, :, None]
        
        # Adjustment cost: C = 0.5 * gamma * ((I/K)^2) * K  (paper eqn (4))
        safe_K = np.maximum(K_mat, 1e-12)   
        safe_K_exp = safe_K[:, :, None]     
    
        adj_cost = 0.5 * params.gamma * ((invest_expanded / safe_K_exp) ** 2) * safe_K_exp

        
        # External finance: ext_funds 
        ext_funds = np.maximum(0.0, invest_expanded - pi_expanded)   
        # Fixed cost 
        finance_cost = params.phi0 * K_mat[:, :, None] * (ext_funds > 0).astype(float)

        
        # Flow profit (net)
        net_flow = pi_expanded - invest_expanded - adj_cost - finance_cost
        
    
        EV_for_Kprime = EV[None, :, :]   
        EV_for_Kprime = np.transpose(EV_for_Kprime, (1, 2, 0))  
        EV_next = np.transpose(EV_for_Kprime, (2,1,0))         
        EV_next = np.repeat(EV_next, nK, axis=0)               
        
        
        # RHS of Bellman: net_flow + beta * EV_next
        RHS = net_flow + beta * EV_next   # (nK,nA,nKprime)
        
        # Maximize over K' choice axis=2
        V_new = np.max(RHS, axis=2)            # (nK,nA)
        policy_idx_new = np.argmax(RHS, axis=2) 
        
        # Convergence
        diff = np.max(np.abs(V_new - V))
        V = V_new
        policy_idx = policy_idx_new
        
        if diff < params.vfi_tol:
            print(f"VFI converged in {it+1} iterations, diff={diff:.6g}")
            break
    else:
        print("VFI did not converge within maxiter")
    
    # return V, policy indices and the chosen capital as values
    policy_capital = Kgrid[policy_idx]
    return {"V": V, "policy_idx": policy_idx, "policy_capital": policy_capital}


# %%
def simulate_panel(params: Params, sol: Dict[str, Any],
                   Kgrid: np.ndarray, Agrid: np.ndarray,
                   P_A: np.ndarray, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Vectorized simulation function that uses capital grid indices directly.
    Computes marginal q via finite differences on V (central where possible,
    one-sided at boundaries).
    """
    if seed is None:
        seed = params.random_seed
    rng = np.random.default_rng(seed)

    N = params.Nfirms
    T_total = params.T + params.burn
    nK = len(Kgrid)
    nA = len(Agrid)

    # Extract policy and value function
    policy_idx = sol["policy_idx"]  # shape: (nK, nA) - indices in Kgrid
    V = sol["V"]                    # shape: (nK, nA)

    # Initialize arrays
    K_path = np.zeros((T_total, N))
    A_idx_path = np.zeros((T_total, N), dtype=int)
    I_over_K = np.zeros((T_total, N))
    pi_over_K = np.zeros((T_total, N))
    ext_amount = np.zeros((T_total, N))
    Q = np.zeros((T_total, N))

    # Initial states (start at median)
    K_idx = np.full(N, nK // 2, dtype=int)
    A_idx = np.full(N, nA // 2, dtype=int)

    K_path[0, :] = Kgrid[K_idx]
    A_idx_path[0, :] = A_idx

    # Pre-compute cumulative probabilities for transitions
    cumP = np.cumsum(P_A, axis=1)  # shape (nA, nA)

    K_forward_diff = np.empty(nK)
    K_backward_diff = np.empty(nK)
    K_forward_diff[:-1] = Kgrid[1:] - Kgrid[:-1]
    K_forward_diff[-1] = K_forward_diff[-2]  
    K_backward_diff[1:] = Kgrid[1:] - Kgrid[:-1]
    K_backward_diff[0] = K_backward_diff[1]

    for t in range(T_total):
        
        K_curr = Kgrid[K_idx]              # (N,)
        A_curr = Agrid[A_idx]              # (N,)

        # Next period capital index from policy (vectorized)
        K_next_idx = policy_idx[K_idx, A_idx]  # (N,)

        # Compute investment and other variables
        K_next = Kgrid[K_next_idx]
        invest = K_next - (1 - params.delta) * K_curr

        # Profit
        pi = A_curr * (K_curr ** params.alpha)

        # External finance
        ext = np.maximum(0.0, invest - pi)

        # Store results
        safe_K = np.maximum(K_curr, 1e-12)
        I_over_K[t, :] = invest / safe_K
        pi_over_K[t, :] = pi / safe_K
        ext_amount[t, :] = ext

        k_plus = np.minimum(K_idx + 1, nK - 1)
        k_minus = np.maximum(K_idx - 1, 0)

        # central denom
        dk_central = Kgrid[k_plus] - Kgrid[k_minus]

        # Values used in numerator
        V_plus = V[k_plus, A_idx]
        V_minus = V[k_minus, A_idx]

        # Prepare Q array
        q_vals = np.empty(N, dtype=float)

        # interior points (where k_plus > k_minus)
        interior_mask = (k_plus > k_minus)
        q_vals[interior_mask] = (V_plus[interior_mask] - V_minus[interior_mask]) / dk_central[interior_mask]

        left_mask = (K_idx == 0)
        if np.any(left_mask):
            kp = K_idx[left_mask] + 1
            km = K_idx[left_mask]
            num = V[kp, A_idx[left_mask]] - V[km, A_idx[left_mask]]
            den = Kgrid[kp] - Kgrid[km]
            q_vals[left_mask] = num / np.where(den == 0, 1e-12, den)

        right_mask = (K_idx == nK - 1)
        if np.any(right_mask):
            kp = K_idx[right_mask]
            km = K_idx[right_mask] - 1
            num = V[kp, A_idx[right_mask]] - V[km, A_idx[right_mask]]
            den = Kgrid[kp] - Kgrid[km]
            q_vals[right_mask] = num / np.where(den == 0, 1e-12, den)

        Q[t, :] = q_vals

        # Update for next period
        if t < T_total - 1:
            K_idx = K_next_idx
            K_path[t + 1, :] = Kgrid[K_idx]

            # Draw next productivity states vectorized:
            u = rng.random(N)  # uniform draws
            rows = cumP[A_idx, :]             # (N, nA)
            A_idx = np.sum(u[:, None] > rows, axis=1)
            A_idx_path[t + 1, :] = A_idx

    # Discard burn-in
    def discard(arr):
        return arr[params.burn:, :]

    return {
        "I_over_K": discard(I_over_K),
        "pi_over_K": discard(pi_over_K),
        "K_path": discard(K_path),
        "A_path": Agrid[discard(A_idx_path)],  # Convert indices to actual values
        "ext_amount": discard(ext_amount),
        "Q": discard(Q)
    }


# %%
# -------------------------
# Moments
# -------------------------
def compute_moments(sim_data: Dict[str, np.ndarray]) -> np.ndarray:
    I_over_K = sim_data["I_over_K"]     # shape (T, N)
    pi_over_K = sim_data["pi_over_K"]
    Q = sim_data["Q"]
    ext_amount = sim_data["ext_amount"]
    K_path = sim_data["K_path"]

    
    y = I_over_K[:-1].ravel()         # I_t
    q_lead = Q[1:].ravel()            # q_{t+1} proxy for E_t[q_{t+1}]
    pi_t = pi_over_K[:-1].ravel()     # pi_t/K_t
    X = np.column_stack([q_lead, pi_t, np.ones_like(y)])
    coeff, *_ = np.linalg.lstsq(X, y, rcond=None)
    a1, a2 = float(coeff[0]), float(coeff[1])

    # standard deviations (pooled)
    std_IK = float(np.std(I_over_K))
    std_piK = float(np.std(pi_over_K))

    
    T_sim, N = I_over_K.shape
    ar1_list = []
    for i in range(N):
        series = I_over_K[:, i]
        if np.all(np.isfinite(series)) and len(series) > 1:
            x1 = series[:-1]
            x2 = series[1:]
            if np.std(x1) > 0 and np.std(x2) > 0:
                ar1_list.append(np.corrcoef(x1, x2)[0, 1])
    ar1_IK = float(np.nanmean(ar1_list)) if len(ar1_list) > 0 else 0.0

    mean_q = float(np.nanmean(Q))

    total_invest = np.sum(I_over_K * K_path)
    total_ext = np.sum(ext_amount)
    ext_frac = float(total_ext / total_invest) if total_invest > 0 else 0.0

    return np.array([a1, a2, std_IK, std_piK, ar1_IK, mean_q, ext_frac])



# %%
# -------------------------
# Estimation helpers
# -------------------------
def estimate_single_run(theta: np.ndarray, params_base: Params, W: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float]:
    """
    Run model once with parameters theta.
    Returns (moments, distance) where distance = (moments - TARGET).T @ W @ (moments - TARGET).
    If W is None, use identity.
    """
    params = Params(
        alpha=float(theta[0]),
        gamma=float(theta[1]),
        rho=float(theta[2]),
        sigma=float(theta[3]),
        phi0=float(theta[4]),
        nK=params_base.nK,
        Kmin=params_base.Kmin,
        Kmax=params_base.Kmax,
        nA=params_base.nA,
        Nfirms=params_base.Nfirms,
        T=params_base.T,
        burn=params_base.burn,
        random_seed=params_base.random_seed
    )

    Alog, P_A = tauchen(params.rho, params.sigma, n=params.nA)
    Agrid = np.exp(Alog)
    Kgrid = np.linspace(params.Kmin, params.Kmax, params.nK)
    grid_i = np.linspace(0.0, 8.0, 80)

    sol = solve_dp(params, Kgrid, grid_i, Agrid, P_A)
    sim_data = simulate_panel(params, sol, Kgrid, Agrid, P_A)
    moments = compute_moments(sim_data)

    diff = moments - TARGET_MOMENTS
    if W is None:
        W = np.eye(len(TARGET_MOMENTS))
    distance = float(diff @ W @ diff)
    return moments, distance

def compute_optimal_weighting_matrix(theta_hat: np.ndarray, params_base: Params, n_simulations: int = 50) -> np.ndarray:
    all_moments = []
    for i in range(n_simulations):
        params = Params(
            alpha=theta_hat[0],
            gamma=theta_hat[1],
            rho=theta_hat[2],
            sigma=theta_hat[3],
            phi0=theta_hat[4],
            delta=params_base.delta,
            beta=params_base.beta,
            phi1=params_base.phi1,
            nK=params_base.nK,
            Kmin=params_base.Kmin,
            Kmax=params_base.Kmax,
            nA=params_base.nA,
            Nfirms=params_base.Nfirms,
            T=params_base.T,
            burn=params_base.burn,
            random_seed=params_base.random_seed + i
        )
        Alog, P_A = tauchen(params.rho, params.sigma, n=params.nA)
        Agrid = np.exp(Alog)
        Kgrid = np.linspace(params.Kmin, params.Kmax, params.nK)

        sol = solve_dp(params, Kgrid, np.linspace(0.0, 8.0, 80), Agrid, P_A)
        sim_data = simulate_panel(params, sol, Kgrid, Agrid, P_A)
        moments = compute_moments(sim_data)
        all_moments.append(moments)

    all_moments = np.array(all_moments)
    Omega = np.cov(all_moments, rowvar=False)
    Omega += 1e-8 * np.eye(Omega.shape[0])
    return np.linalg.pinv(Omega)

def compute_standard_errors(theta: np.ndarray, params_base: Params,
                           W_optimal: np.ndarray, n_simulations: int = 50,
                           base_eps: float = 1e-5) -> np.ndarray:
    """
    Compute standard errors.
    n_simulations should match the number of panels used to compute W_optimal.
    base_eps is a base step; actual step is scaled by parameter magnitude.
    """
    n_params = len(theta)
    n_moments = len(TARGET_MOMENTS)
    G = np.zeros((n_moments, n_params))

    for j in range(n_params):
        # scale eps relative to parameter magnitude
        h = base_eps * max(1e-6, abs(theta[j]))
        theta_plus = theta.copy()
        theta_minus = theta.copy()
        theta_plus[j] += h
        theta_minus[j] -= h
        moments_plus, _ = estimate_single_run(theta_plus, params_base)
        moments_minus, _ = estimate_single_run(theta_minus, params_base)
        G[:, j] = (moments_plus - moments_minus) / (2 * h)

    S = int(n_simulations)   # number of simulated panels used to compute W
    GtWG = G.T @ W_optimal @ G
    # invert with pinv for numerical stability
    inv = np.linalg.pinv(GtWG)
    Q = (1 + 1.0 / S) * inv
    se = np.sqrt(np.maximum(0.0, np.diag(Q)))
    return se






# %%
# -------------------------
# Two-step SMM
# -------------------------
def run_two_step_smm(params_base: Params):
    print("=" * 80)
    print("TWO-STEP SMM ESTIMATION")
    print("=" * 80)

    output_dir = Path(params_base.output_dir)
    output_dir.mkdir(exist_ok=True)

    theta0 = np.array([0.6956, 0.1331, 0.0976, 0.8932, 0.0])
    bounds = [
        (0.1, 1.0),
        (0.01, 2.0),
        (0.01, 0.99),
        (0.1, 2.0),
        (0.0, 0.02)
    ]

    # STEP 1: identity-weighted SMM (simple squared distance)
    print("\nSTEP 1: First-stage estimation (identity weighting)...")
    start_time = time.time()

    def objective_identity(theta):
        _, dist = estimate_single_run(theta, params_base, W=np.eye(len(TARGET_MOMENTS)))
        return dist

    res1 = optimize.minimize(
        objective_identity,
        theta0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20, 'disp': True, 'ftol': 1e-3}
    )
    theta_identity = res1.x
    moments_identity, _ = estimate_single_run(theta_identity, params_base, W=np.eye(len(TARGET_MOMENTS)))
    print(f"First-stage completed in {time.time() - start_time:.1f} sec, obj={res1.fun:.6f}")

    # STEP 2: compute optimal weighting matrix
    print("\nSTEP 2: Computing optimal weighting matrix...")
    start_time = time.time()
    n_sim_W = 20
    W_optimal = compute_optimal_weighting_matrix(theta_identity, params_base, n_simulations=n_sim_W)
    print(f"Computed W (shape={W_optimal.shape}) in {time.time() - start_time:.1f} sec")


    # STEP 3: efficient SMM
    print("\nSTEP 3: Second-stage estimation (optimal weighting)...")
    start_time = time.time()

    def objective_optimal(theta):
        _, dist = estimate_single_run(theta, params_base, W=W_optimal)
        return dist

    res2 = optimize.minimize(
        objective_optimal,
        theta_identity,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 20, 'disp': True, 'ftol': 1e-3}
    )
    theta_efficient = res2.x
    moments_efficient, _ = estimate_single_run(theta_efficient, params_base, W=W_optimal)
    print(f"Second-stage completed in {time.time() - start_time:.1f} sec, obj={res2.fun:.6f}")

    # STEP 4: compute standard errors
    print("\nSTEP 4: Computing standard errors...")
    start_time = time.time()
    se = compute_standard_errors(theta_efficient, params_base, W_optimal, n_simulations=n_sim_W)
    print(f"Std errors computed in {time.time() - start_time:.1f} sec")

    # STEP 5: create LaTeX tables & CSVs
    print("\nSTEP 5: Creating LaTeX & CSV outputs...")
    param_tex_path, moments_tex_path = create_latex_tables(theta_identity, theta_efficient, se,
                                                          moments_identity, moments_efficient, output_dir)

    print("\nESTIMATION SUMMARY")
    print("-" * 60)
    for i, name in enumerate(PARAM_NAMES):
        ident = theta_identity[i]
        eff = theta_efficient[i]
        se_val = se[i]
        t_stat = eff / se_val if se_val > 0 else 0.0
        print(f"{name:<10} {ident:<12.6f} {eff:<12.6f} {se_val:<12.6f} {t_stat:<8.3f}")

    print("\nMoment comparison:")
    for i, name in enumerate(MOMENT_NAMES):
        target = TARGET_MOMENTS[i]
        ident = moments_identity[i]
        eff = moments_efficient[i]
        diff = eff - target
        print(f"{name:<20} {target:<8.4f} {ident:<8.4f} {eff:<8.4f} {diff:<8.4f}")

    print(f"\nAll results saved to: {output_dir}")
    return {
        'theta_identity': theta_identity,
        'theta_efficient': theta_efficient,
        'standard_errors': se,
        'moments_identity': moments_identity,
        'moments_efficient': moments_efficient,
        'W_optimal': W_optimal,
        'output_dir': output_dir
    }
def create_latex_tables(theta_identity: np.ndarray, 
                        theta_efficient: np.ndarray,
                        se: np.ndarray,
                        moments_identity: np.ndarray,
                        moments_efficient: np.ndarray,
                        output_dir: Path):
    """Create LaTeX tables and CSV summaries for results.

    Writes files using UTF-8 and converts Unicode pi to LaTeX math ($\\pi$).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    def latex_safe(name: str) -> str:
        
        return name.replace('π', r'$\pi$')

    # --- Parameter estimates table (LaTeX) ---
    param_tex_path = output_dir / "parameter_estimates.tex"
    with open(param_tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lccccc}\n")
        f.write("\\toprule\n")
        f.write("Parameter & Identity-W & Efficient-W & Std. Error & t-Stat \\\\\n")
        f.write("\\midrule\n")

        n_params = len(PARAM_NAMES)
        for i in range(n_params):
            name = PARAM_NAMES[i]
            ident = float(theta_identity[i]) if i < len(theta_identity) else np.nan
            eff = float(theta_efficient[i]) if i < len(theta_efficient) else np.nan
            se_val = float(se[i]) if i < len(se) else np.nan
            t_stat = eff / se_val if (se_val is not None and se_val != 0 and not np.isnan(se_val)) else 0.0
            f.write(f"{name} & {ident:.6f} & {eff:.6f} & {se_val:.6f} & {t_stat:.3f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # --- Moments comparison table (LaTeX) ---
    moments_tex_path = output_dir / "moments_comparison.tex"
    with open(moments_tex_path, "w", encoding="utf-8") as f:
        f.write("\\begin{tabular}{lcccc}\n")
        f.write("\\toprule\n")
        f.write("Moment & Target & Identity-W & Efficient-W & Diff \\\\\n")
        f.write("\\midrule\n")

        n_mom = len(MOMENT_NAMES)
        for i in range(n_mom):
            raw_name = MOMENT_NAMES[i]
            name = latex_safe(raw_name)
            target = float(TARGET_MOMENTS[i]) if i < len(TARGET_MOMENTS) else np.nan
            ident = float(moments_identity[i]) if i < len(moments_identity) else np.nan
            eff = float(moments_efficient[i]) if i < len(moments_efficient) else np.nan
            diff = eff - target if (not np.isnan(eff) and not np.isnan(target)) else np.nan
            f.write(f"{name} & {target:.4f} & {ident:.4f} & {eff:.4f} & {diff:.4f} \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

    # --- CSV summaries 
    try:
        param_df = pd.DataFrame({
            'Parameter': PARAM_NAMES,
            'Identity_W': np.asarray(theta_identity, dtype=float),
            'Efficient_W': np.asarray(theta_efficient, dtype=float),
            'Std_Error': np.asarray(se, dtype=float),
            't_Stat': np.asarray(theta_efficient, dtype=float) / (np.asarray(se, dtype=float) + 1e-12)
        })
        param_df.to_csv(output_dir / "parameter_summary.csv", index=False, encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write parameter_summary.csv ({e})")

    try:
        moments_df = pd.DataFrame({
            'Moment': MOMENT_NAMES,
            'Target': np.asarray(TARGET_MOMENTS, dtype=float),
            'Identity_W': np.asarray(moments_identity, dtype=float),
            'Efficient_W': np.asarray(moments_efficient, dtype=float),
            'Difference': np.asarray(moments_efficient, dtype=float) - np.asarray(TARGET_MOMENTS, dtype=float)
        })
        moments_df.to_csv(output_dir / "moments_summary.csv", index=False, encoding="utf-8")
    except Exception as e:
        print(f"Warning: could not write moments_summary.csv ({e})")

    return param_tex_path, moments_tex_path


# %%
# -------------------------
# Quick test 
# -------------------------
def test_quick_run():
    print("Running quick test...")
    params = Params()
    params.nK = 100
    params.Nfirms = 100
    params.T = 20
    params.burn = 5
    params.output_dir = "ps8_test_results"

    theta_test = np.array([0.6956, 0.1331, 0.0976, 0.8932, 0.0])
    params.alpha = theta_test[0]
    params.gamma = theta_test[1]
    params.rho = theta_test[2]
    params.sigma = theta_test[3]
    params.phi0 = theta_test[4]

    Alog, P_A = tauchen(params.rho, params.sigma, n=params.nA)
    Agrid = np.exp(Alog)
    Kgrid = np.linspace(params.Kmin, params.Kmax, params.nK)
    grid_i = np.linspace(0.0, 8.0, 80)

    print("\nSolving DP...")
    st = time.time()
    sol = solve_dp(params, Kgrid, grid_i, Agrid, P_A)
    print(f"DP solved in {time.time() - st:.2f}s")

    print("Simulating panel...")
    st = time.time()
    sim_data = simulate_panel(params, sol, Kgrid, Agrid, P_A)
    print(f"Simulation done in {time.time() - st:.2f}s")

    moments = compute_moments(sim_data)
    print("\nTest moments:")
    for name, val in zip(MOMENT_NAMES, moments):
        print(f"  {name:<20}: {val:.6f}")

    print("\nComparison with target:")
    print(f"{'Moment':<20} {'Model':<12} {'Target':<12} {'Diff':<12}")
    print("-"*60)
    for i, name in enumerate(MOMENT_NAMES):
        print(f"{name:<20} {moments[i]:<12.4f} {TARGET_MOMENTS[i]:<12.4f} {moments[i]-TARGET_MOMENTS[i]:<12.4f}")

    return moments, sol, sim_data

# %%
# -------------------------
# Main
# -------------------------
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--test":
        test_quick_run()
    elif len(sys.argv) > 1 and sys.argv[1] == "--full":
        params_base = Params()
        results = run_two_step_smm(params_base)

        # Save a small LaTeX wrapper
        tex_path = Path(results['output_dir']) / "ps8_report.tex"
        with open(tex_path, "w") as f:
            f.write("\\documentclass[12pt]{article}\n")
            f.write("\\usepackage{amsmath, amssymb, booktabs}\n")
            f.write("\\usepackage[margin=1in]{geometry}\n")
            f.write("\\begin{document}\n")
            f.write("\\section{Parameter Estimates}\n\\input{parameter_estimates.tex}\n")
            f.write("\\section{Moment Comparison}\n\\input{moments_comparison.tex}\n")
            f.write("\\end{document}\n")
        print(f"LaTeX report template created: {tex_path}")
    else:
        print("Usage: python ps8_fixed.py [--test|--full]")
        print("  --test : quick smoke test (fast)")
        print("  --full : full two-step SMM (slow)")

# %%



