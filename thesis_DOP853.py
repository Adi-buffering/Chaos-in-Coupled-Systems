"""
FORCED NONLINEAR CART–PENDULUM WITH TRANSLATIONAL + ROTATIONAL DAMPING
 CHAOS ANALYSIS

Uses explicitly provided acceleration equations.


Numerical solver:
    solve_ivp(method="DOP853")
    rtol=1e-9
    atol=1e-12
    dense_output=True
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from numpy.fft import fft, fftfreq

# ============================================================
# 1) PARAMETERS
# ============================================================

params = {
    "M1": 1.0,
    "M2": 0.3,
    "l": 1.0,
    "g": 9.81,
    "K": 20.0,
    "C1": 0.5,        # cart damping
    "C2": 0.05,       # rotational damping
    "A": 0.6,         # forcing amplitude
    "omega": 2.2      # forcing frequency
}

T_total = 900
transient_fraction = 0.99

# ============================================================
# 2) SYSTEM EQUATIONS (USING EXACT PROVIDED EXPRESSIONS)
# ============================================================

def system_equations(t, y, p):
    x2, x2_dot, theta, theta_dot = y
    
    M1 = p["M1"]
    M2 = p["M2"]
    l  = p["l"]
    g  = p["g"]
    K  = p["K"]
    C1 = p["C1"]
    C2 = p["C2"]
    A  = p["A"]
    omega = p["omega"]
    
    # Base motion
    x1 = A*np.sin(omega*t)
    x1_dot = A*omega*np.cos(omega*t)
    
    s = np.sin(theta)
    c = np.cos(theta)
    
    # ---- EXACT x2_ddot (provided expression)
    numerator = (
        M2*g*s*c
        + M2*l*s*(theta_dot**2)
        - K*(x2 - x1)
        - C1*(x2_dot - x1_dot)
        - (C2/l)*theta_dot*c
    )
    
    denominator = (M1 + M2*s**2)
    
    x2_dd = numerator / denominator
    
    # ---- EXACT theta_ddot (provided expression)
    theta_dd = -(1/l)*(
        x2_dd*c
        + g*s
        + (C2/(M2*l))*theta_dot
    )
    
    return [x2_dot, x2_dd, theta_dot, theta_dd]

# ============================================================
# 3) HIGH-PRECISION SIMULATION
# ============================================================

def run_simulation(y0, p):
    sol = solve_ivp(
        lambda t, y: system_equations(t, y, p),
        [0, T_total],
        y0,
        method="DOP853",
        rtol=1e-9,
        atol=1e-12,
        dense_output=True
    )
    return sol

# ============================================================
# 4) REMOVE TRANSIENT
# ============================================================

def remove_transient(sol):
    t = sol.t
    cutoff = int(len(t)*transient_fraction)
    return t[cutoff:], sol.y[:, cutoff:]

# ============================================================
# 5A) LARGEST LYAPUNOV EXPONENT
# ============================================================

def compute_lyapunov(y0, p, delta0=1e-8):
    y_ref = np.array(y0)
    y_pert = np.array(y0)
    y_pert[2] += delta0
    
    dt = 0.5
    t0 = 0
    sum_log = 0
    steps = 0
    
    while t0 < T_total:
        sol_ref = solve_ivp(
            lambda t,y: system_equations(t,y,p),
            [t0, t0+dt],
            y_ref,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )
        
        sol_pert = solve_ivp(
            lambda t,y: system_equations(t,y,p),
            [t0, t0+dt],
            y_pert,
            method="DOP853",
            rtol=1e-9,
            atol=1e-12
        )
        
        y_ref = sol_ref.y[:, -1]
        y_pert = sol_pert.y[:, -1]
        
        diff = y_pert - y_ref
        dist = np.linalg.norm(diff)
        
        sum_log += np.log(dist/delta0)
        steps += 1
        
        diff = delta0 * diff / dist
        y_pert = y_ref + diff
        
        t0 += dt
    
    return sum_log/(steps*dt)

# ============================================================
# 5B) PHASE PORTRAIT
# ============================================================

def plot_phase(t, y):
    plt.figure()
    plt.plot(y[2], y[3], linewidth=0.4)
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.title("Phase Portrait")
    plt.show()

# ============================================================
# 5C) POINCARÉ SECTION
# ============================================================

def poincare_section(sol, p):
    omega = p["omega"]
    T_drive = 2*np.pi/omega
    n_max = int(T_total/T_drive)
    
    t_samples = np.array([n*T_drive for n in range(n_max)])
    y_samples = sol.sol(t_samples)
    
    cutoff = int(len(t_samples)*transient_fraction)
    
    plt.figure()
    plt.scatter(
        y_samples[2][cutoff:],
        y_samples[3][cutoff:],
        s=4
    )
    plt.xlabel("theta")
    plt.ylabel("theta_dot")
    plt.title("Stroboscopic Poincare Section")
    plt.show()

# ============================================================
# 5D) FFT
# ============================================================

def plot_fft(t, y):
    theta = y[2]
    dt = t[1] - t[0]
    
    fft_vals = fft(theta)
    freq = fftfreq(len(theta), dt)
    
    plt.figure()
    plt.plot(freq[:len(freq)//2],
             np.abs(fft_vals[:len(freq)//2]))
    plt.xlabel("Frequency")
    plt.ylabel("Amplitude")
    plt.title("FFT Spectrum")
    plt.show()

# ============================================================
# 5E) BIFURCATION DIAGRAM
# ============================================================

def bifurcation_diagram(omega_values, p):
    theta_vals = []
    omega_plot = []
    
    for w in omega_values:
        p["omega"] = w
        sol = run_simulation([0,0,0.2,0], p)
        
        T_drive = 2*np.pi/w
        n_max = int(T_total/T_drive)
        t_samples = np.array([n*T_drive for n in range(n_max)])
        y_samples = sol.sol(t_samples)
        
        cutoff = int(len(t_samples)*transient_fraction)
        
        theta_vals.extend(y_samples[2][cutoff:])
        omega_plot.extend([w]*len(y_samples[2][cutoff:]))
    
    plt.figure(figsize=(8,6))
    plt.scatter(omega_plot, theta_vals, s=1)
    plt.xlabel("omega")
    plt.ylabel("theta")
    plt.title("Bifurcation Diagram")
    plt.show()

# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":
    
    y0 = [0, 0, 0.2, 0]
    
    sol = run_simulation(y0, params)
    t_clean, y_clean = remove_transient(sol)
    
    lambda_max = compute_lyapunov(y0, params)
    print("Largest Lyapunov Exponent =", lambda_max)
    
    plot_phase(t_clean, y_clean)
    poincare_section(sol, params)
    plot_fft(t_clean, y_clean)
    
    omega_range = np.linspace(1.5, 3.0, 50)
    bifurcation_diagram(omega_range, params)
