
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
N_TOTAL  = 30
n        = 9
alpha    = 0.314423
lambda1 = 43128.095004
a       = 1.2e-5
lambda2 = 15700
b       = 6.4e-5
D        = 60000
P        = 784
s   = int(alpha * n)
if s == 0: s = 1
k   = n - s
r   = 1.0 / ((n / (s + 1))**2 + (n / (s + 1)))
print("r:",r)
mu1 = lambda1 / (r * D)     # computation rate
mu2 = lambda2 / P           # communication rate
c   = a * r * D + b * P     # total shift

print("=" * 60)
print("  PARAMETERS & DERIVED VALUES")
print("=" * 60)
print(f"  n={n}, s={s}, k={k}, r={r:.6f}")
print(f"  mu1 = λ1/(r·D) = {mu1:.6f}")
print(f"  mu2 = λ2/p     = {mu2:.6f}")
print(f"  c   = a·r·D + b·p = {c:.6f} s")
def f_W(t_arr):
    """Hypoexponential PDF of one worker time W = T_comp + T_comm"""
    result = np.zeros_like(t_arr, dtype=float)
    mask   = t_arr > c
    u      = t_arr[mask] - c
    result[mask] = (mu1 * mu2 / (mu1 - mu2)) * (
        np.exp(-mu2 * u) - np.exp(-mu1 * u)
    )
    return result
def F_W(t_arr):
    """Hypoexponential CDF of one worker time"""
    result = np.zeros_like(t_arr, dtype=float)
    mask   = t_arr > c
    u      = t_arr[mask] - c
    result[mask] = 1.0 - (
        mu1 * np.exp(-mu2 * u) - mu2 * np.exp(-mu1 * u)
    ) / (mu1 - mu2)
    return result

def f_Z(t_arr):
    """PDF of Z = k-th order statistic of n i.i.d. workers"""
    coeff = factorial(n) / (factorial(k - 1) * factorial(n - k))
    fw    = f_W(t_arr)
    Fw    = F_W(t_arr)
    return coeff * (Fw ** (k - 1)) * ((1 - Fw) ** (n - k)) * fw
# STEP 1 — Evaluate f_Z on fine grid
N_POINTS = 5_000_000
t_arr    = np.linspace(c + 1e-6, c + 30.0, N_POINTS)
dt       = t_arr[1] - t_arr[0]
fZ_arr   = f_Z(t_arr)
integral = np.sum(fZ_arr) * dt
print(f"\n  Integral of f_Z = {integral:.8f}  (should be 1.0)")

# STEP 2 — Compute mean and variance from f_Z
mean_Z = np.sum(t_arr * fZ_arr) * dt
var_Z  = np.sum((t_arr - mean_Z)**2 * fZ_arr) * dt
std_Z  = np.sqrt(var_Z)
mode_Z = t_arr[np.argmax(fZ_arr)]
print(f"\n  Mean(Z) = {mean_Z:.6f} s")
print(f"  Var(Z)  = {var_Z:.6f} s²")
print(f"  Std(Z)  = {std_Z:.6f} s")
print(f"  Mode(Z) = {mode_Z:.6f} s")

# STEP 3 — Method of Moments fit
mu_fit = 1.0 / std_Z
c_fit  = mean_Z - std_Z
check_mean = c_fit + 1.0 / mu_fit
check_var  = 1.0 / mu_fit**2
print(f"\n  ── Method of Moments Fit ──────────────────")
print(f"  mu_fit = 1/std(Z)        = {mu_fit:.6f}")
print(f"  c_fit  = mean(Z)-std(Z)  = {c_fit:.6f}")
print(f"\n  Verification:")
print(f"  Fitted mean = c + 1/mu   = {check_mean:.6f}  (actual = {mean_Z:.6f})")
print(f"  Fitted var  = 1/mu²      = {check_var:.6f}  (actual = {var_Z:.6f})")
fZ_fit = np.zeros_like(t_arr, dtype=float)
mask_fit = t_arr > c_fit
fZ_fit[mask_fit] = mu_fit * np.exp(-mu_fit * (t_arr[mask_fit] - c_fit))
mse  = np.sum((fZ_arr - fZ_fit) ** 2) * dt          
mae  = np.sum(np.abs(fZ_arr - fZ_fit)) * dt          
eps  = 1e-300
mask_pos = fZ_arr > eps
kl   = np.sum(fZ_arr[mask_pos] * np.log(fZ_arr[mask_pos] / (fZ_fit[mask_pos] + eps))) * dt

print(f"\n  ── Fit Quality Metrics ────────────────────")
print(f"  MSE= {mse:.6e}")
print(f"  MAE= {mae:.6f}  (total variation distance)")
