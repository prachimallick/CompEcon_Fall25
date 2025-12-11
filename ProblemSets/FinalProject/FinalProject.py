# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
import statsmodels.api as sm

sns.set(style="whitegrid")

# Load dataset
DATA_PATH = "student_depression_dataset.csv"
df = pd.read_csv(DATA_PATH)
print("Loaded dataset: rows =", len(df), "columns =", len(df.columns))
print(df.columns.tolist())




# %%
#clean data
import re
import numpy as np
import pandas as pd
import os

OUTDIR = "./analysis_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# helper: remove outer quotes and normalize
def _normalize_raw(s):
    if pd.isna(s):
        return s
    # convert to str, strip whitespace, remove surrounding quotes, lower
    t = str(s).strip()
    t = re.sub(r'^[\'"]+|[\'"]+$', '', t)   # strip surrounding quotes
    return t

# Clean Sleep
def clean_sleep(s):
    if pd.isna(s):
        return np.nan
    raw = _normalize_raw(s).lower()

   
    lt_match = re.search(r'(less than|<|under|below)\s*(\d+(\.\d+)?)', raw)
    if lt_match:
        try:
            n = float(lt_match.group(2))
            return max(0.0, n - 0.5)   
        except:
            return np.nan

    # ranges: "5-6", "5 – 6", "5 to 6"
    range_match = re.search(r'(\d+(?:\.\d+)?)[\s]*(?:-|–|to)[\s]*(\d+(?:\.\d+)?)', raw)
    if range_match:
        try:
            a = float(range_match.group(1))
            b = float(range_match.group(2))
            return (a + b) / 2.0
        except:
            pass

    # hh:mm format
    time_match = re.search(r'(\d+):(\d{1,2})', raw)
    if time_match:
        try:
            h = float(time_match.group(1))
            m = float(time_match.group(2))
            return h + m / 60.0
        except:
            pass

    # direct numerics
    num_match = re.search(r'(\d+(?:\.\d+)?)', raw)
    if num_match:
        try:
            return float(num_match.group(1))
        except:
            return np.nan

    # small mapping for spelled-out
    word_to_num = {
        'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
        'eight':8,'nine':9,'ten':10
    }
    for w,v in word_to_num.items():
        if re.search(r'\b' + w + r'\b', raw):
            return float(v)

    return np.nan

# --- Clean Work/Study Hours robustly ---
def clean_hours(x):
    if pd.isna(x):
        return np.nan
    s = _normalize_raw(x).lower()

    # hh:mm
    m = re.search(r'(\d+):(\d{1,2})', s)
    if m:
        try:
            return float(m.group(1)) + float(m.group(2)) / 60.0
        except:
            pass

    # range (5 - 7 or 5 to 7)
    rm = re.search(r'(\d+(?:\.\d+)?)[\s]*(?:-|–|to)[\s]*(\d+(?:\.\d+)?)', s)
    if rm:
        try:
            return (float(rm.group(1)) + float(rm.group(2))) / 2.0
        except:
            pass

    # simple number
    nm = re.search(r'(\d+(?:\.\d+)?)', s)
    if nm:
        try:
            return float(nm.group(1))
        except:
            return np.nan

    return np.nan

# Clean Financial Stress 
def clean_finstress(x):
    if pd.isna(x):
        return np.nan
    s = _normalize_raw(x)
    # common non-numeric tokens -> NaN
    if re.fullmatch(r'[-\?]|n/?a|none|unknown|others?', s.strip().lower()):
        return np.nan
    # otherwise try numeric (may include decimals or commas)
    s2 = s.replace(',', '.')
    m = re.search(r'(\d+(?:\.\d+)?)', s2)
    if m:
        try:
            return float(m.group(1))
        except:
            return np.nan
    return np.nan


# 1) Sleep
if 'Sleep Duration' in df.columns:
    df['Sleep_clean'] = df['Sleep Duration'].apply(clean_sleep)
else:
    print("Warning: 'Sleep Duration' not found.")

# 2) Hours 
if 'Work/Study Hours' in df.columns:
    df['Hours_clean'] = df['Work/Study Hours'].apply(clean_hours)
    df['Hours'] = df['Hours_clean']
else:
    print("Warning: 'Work/Study Hours' not found.")

# 3) Financial Stress numeric 
if 'Financial Stress' in df.columns:
    df['FinancialStress_num'] = df['Financial Stress'].apply(clean_finstress)
else:
    print("Warning: 'Financial Stress' not found.")

mask_sleep_bad = pd.Series(False, index=df.index)
mask_fs_bad = pd.Series(False, index=df.index)

if 'Sleep Duration' in df.columns and 'Sleep_clean' in df.columns:
    mask_sleep_bad = df['Sleep_clean'].isna() & df['Sleep Duration'].notna()
    print("Rows with missing Sleep_clean (showing index and a few cols):")

if 'Financial Stress' in df.columns and 'FinancialStress_num' in df.columns:
    mask_fs_bad = df['FinancialStress_num'].isna() & df['Financial Stress'].notna()
    print("Rows with missing FinancialStress_num (showing index and a few cols):")

parts = []
if mask_sleep_bad.any():
    parts.append(df.loc[mask_sleep_bad].copy().assign(problem='sleep_parse'))
if mask_fs_bad.any():
    parts.append(df.loc[mask_fs_bad].copy().assign(problem='finstress_parse'))

if parts:
    problem_rows = pd.concat(parts, axis=0)
    problem_rows_path = os.path.join(OUTDIR, "problem_rows_for_manual_review.csv")
    problem_rows.to_csv(problem_rows_path, index=False)
    print("Saved problematic rows for manual review to:", problem_rows_path)
else:
    print("No problematic rows to save.")

print("=== QA: sample conversions ===")
if 'Sleep Duration' in df.columns:
    sample_sleep_bad = df.loc[df['Sleep_clean'].isna() & df['Sleep Duration'].notna(), 'Sleep Duration'].unique()[:10]
    print("Sleep entries that failed to convert (up to 10):", list(sample_sleep_bad))

if 'Work/Study Hours' in df.columns:
    sample_hours_bad = df.loc[df['Hours'].isna() & df['Work/Study Hours'].notna(), 'Work/Study Hours'].unique()[:10]
    print("Work/Study Hours entries that failed to convert (up to 10):", list(sample_hours_bad))
    print("Hours_clean non-missing:", int(df['Hours'].notna().sum()), "/", len(df))

if 'Financial Stress' in df.columns:
    print("Financial Stress dtype before cleaning:", df['Financial Stress'].dtype)
    orig_bad = df.loc[df['FinancialStress_num'].isna() & df['Financial Stress'].notna(), 'Financial Stress'].unique()[:15]
    print("Examples of non-numeric Financial Stress entries (up to 15):", list(orig_bad))
    print("FinancialStress_num non-missing:", int(df['FinancialStress_num'].notna().sum()), "/", len(df))

clean_path = os.path.join(OUTDIR, "cleaned_student_data_snapshot.csv")
df.to_csv(clean_path, index=False)
print("Saved cleaned snapshot to:", clean_path)

def _assert_close(a, b, tol=1e-6):
    assert abs(a - b) <= tol, f"{a} not close to {b}"

_tests_passed = True
try:
    assert clean_sleep("7") == 7.0
    assert clean_sleep("7 hrs") == 7.0
    assert abs(clean_sleep("4:30") - 4.5) < 1e-9
    assert abs(clean_sleep("5-6") - 5.5) < 1e-9
    assert clean_hours("2:30") == 2.5
    assert clean_hours("5 to 7") == 6.0
    assert clean_finstress("3") == 3.0
    assert clean_sleep("seven hours") == 7.0
except AssertionError as e:
    _tests_passed = False
    print("Unit test failed:", e)

print("Unit tests passed?" , _tests_passed)


# %%
group_means = df.groupby('Depression').agg(
    n=('CGPA','count'),
    mean_CGPA=('CGPA','mean'),
    mean_hours=('Hours','mean'),
    mean_sleep=('Sleep_clean','mean'),
    mean_finstress=('FinancialStress_num','mean')
).reset_index()
group_means
# mean financial stress safely
if df['FinancialStress_num'].notna().any():
    fin = df.groupby('Depression')['FinancialStress_num'].mean().reset_index().rename(columns={'FinancialStress_num':'mean_finstress'})
    group_means = group_means.merge(fin, on='Depression', how='left')
else:
    group_means['mean_finstress'] = np.nan

print("\nGroup means by Depression:")
print(group_means)


# T-TESTS for CGPA, Hours, Sleep_clean

def welch_ttest_and_ci(x1, x2, alpha=0.05):
    x1 = x1.dropna().astype(float)
    x2 = x2.dropna().astype(float)
    if len(x1) < 2 or len(x2) < 2:
        return None
    t_stat, p_val = stats.ttest_ind(x1, x2, equal_var=False)
    diff = x1.mean() - x2.mean()
    se = np.sqrt(x1.var(ddof=1)/len(x1) + x2.var(ddof=1)/len(x2))
    z = stats.norm.ppf(1-alpha/2)
    ci_lower = diff - z*se
    ci_upper = diff + z*se
    return {'t': t_stat, 'p': p_val, 'diff': diff, 'se': se, 'ci': (ci_lower, ci_upper), 'n1': len(x1), 'n2': len(x2)}

dep = df[df['Depression']==1]
nondep = df[df['Depression']==0]

print("\nT-tests (Depressed - NonDepressed):")
for col in ['CGPA','Hours','Sleep_clean']:
    if col in df.columns:
        res = welch_ttest_and_ci(dep[col], nondep[col])
        if res is None:
            print(f"{col}: insufficient data for t-test")
        else:
            print(f"{col}: diff={res['diff']:.4f}, t={res['t']:.4f}, p={res['p']:.4g}, 95%CI={res['ci']}")


# %%
plt.figure(figsize=(7,5))

for dep, label, color in [(0, 'No Depression', 'blue'),
                          (1, 'Depressed', 'orange')]:
    sns.kdeplot(
        df[df['Depression'] == dep]['CGPA'].dropna(),
        fill=True, alpha=0.3, linewidth=2,
        label=label, color=color
    )

plt.xlabel('CGPA')
plt.ylabel('Density')
plt.title('CGPA Distribution by Depression Status (Density Plot)')
plt.legend()

# ---- Save the plot ----
plt.savefig(os.path.join(OUTDIR, "cgpa_density_by_depression.png"), dpi=300, bbox_inches='tight')



# %%
# REGRESSION: Hours ~ Depression + Age + C(Gender) + C(Degree)
# Basic protection: if Degree not found treat as Others
if 'Degree' not in df.columns:
    df['Degree'] = 'Others'

print("\nFitting OLS for Hours (HC1 robust SE)...")
m_hours = smf.ols("Hours ~ Depression + Age + C(Gender) + C(Degree)", data=df).fit(cov_type='HC1')
print(m_hours.summary())

# REGRESSION: CGPA ~ Hours (+ Hours^2) + Depression + interactions
print("\nFitting interaction OLS: CGPA ~ Hours + I(Hours**2) + Depression + Depression:Hours + Depression:I(Hours**2)")
model = smf.ols("""CGPA ~ Hours + I(Hours**2) + Depression + Depression:Hours + Depression:I(Hours**2) + Age + C(Gender) + C(Degree)""", data=df).fit(cov_type='HC1')
print(model.summary())

# MARGINAL PRODUCTS (from interaction model)
def safe_param(model, *names):
    """Try multiple alternative parameter names, return first found value else 0."""
    for name in names:
        if name in model.params.index:
            return float(model.params[name])
    return 0.0

beta1 = safe_param(model, 'Hours', 'Hours')
beta2 = safe_param(model, 'I(Hours ** 2)', 'I(Hours**2)', 'I(Hours ** 2)')
gamma2 = safe_param(model, 'Depression:Hours', 'Depression:Hours')
gamma3 = safe_param(model, 'Depression:I(Hours ** 2)', 'Depression:I(Hours**2)', 'Depression:I(Hours ** 2)')

def MP_non(H):
    return beta1 + 2.0 * beta2 * H

def MP_dep(H):
    return (beta1 + gamma2) + 2.0 * (beta2 + gamma3) * H

test_hours = [2,4,6,8]
print("\nMarginal products at selected hours:")
for h in test_hours:
    mpn = MP_non(h)
    mpd = MP_dep(h)
    ratio = mpd / mpn if mpn != 0 else np.nan
    print(f"Hours={h}: MP_non={mpn:.6f},  MP_dep={mpd:.6f},  ratio={ratio}")



# %%

H_range = np.linspace(df['Hours'].min(), df['Hours'].max(), 200)

MP_non_vals = [MP_non(h) for h in H_range]
MP_dep_vals = [MP_dep(h) for h in H_range]

plt.figure(figsize=(7,5))
plt.plot(H_range, MP_non_vals, label='MP Non-Depressed', color='blue')
plt.plot(H_range, MP_dep_vals, label='MP Depressed', color='orange')

plt.axhline(0, color='black', linestyle='--', linewidth=1)  # zero line

plt.xlabel("Work/Study Hours")
plt.ylabel("Marginal Effect on CGPA")
plt.title("Marginal Product of Hours on CGPA by Depression Status")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(OUTDIR, "marginal_product_plot.png"), dpi=300, bbox_inches='tight')


# %%
# DYNAMIC PROGRAMMING SETUP
b1 = beta1
b2 = beta2
g2 = gamma2
g3 = gamma3

mean_H_non = float(df.loc[df['Depression']==0, 'Hours'].mean())
mean_H_dep = float(df.loc[df['Depression']==1, 'Hours'].mean())
mean_GPA = float(df['CGPA'].mean())

os.makedirs(OUTDIR, exist_ok=True)

MP_non_at_mean = MP_non(mean_H_non)
MP_dep_at_mean = MP_dep(mean_H_dep)
e_dep_empirical = MP_dep_at_mean / MP_non_at_mean if MP_non_at_mean != 0 else 1.0

print("\nProduction coefficients used:")
print(" b1 (Hours) =", b1)
print(" b2 (Hours^2) =", b2)
print(" g2 (Dep:Hours) =", g2)
print(" g3 (Dep:Hours^2) =", g3)
print("\nEmpirical anchors:")
print(" mean_H_non =", round(mean_H_non,3), " MP_non =", round(MP_non_at_mean,6))
print(" mean_H_dep =", round(mean_H_dep,3), " MP_dep =", round(MP_dep_at_mean,6))
print(" e_dep_empirical =", round(e_dep_empirical,3))

# Save cleaned snapshot
clean_path = os.path.join(OUTDIR, "cleaned_student_data_snapshot.csv")
df.to_csv(clean_path, index=False)

# DP grids
beta = 0.95
G_min, G_max, nG = 0.0, 10.0, 201
G_grid = np.linspace(G_min, G_max, nG)
l_min, l_max, nL = 0.0, 12.0, 49
l_grid = np.linspace(l_min, l_max, nL)

# utility curvature setting 
# sigma = 1.0 -> log utility; sigma != 1 -> CRRA with parameter sigma
# choose sigma according to professor suggestion (log is a good default)
sigma = 1.0
_eps = 1e-8  # numerical floor for G_next to avoid log(0) or negative powers

def solve_dp(prod_increment_func, chi, tol=1e-6, max_iter=1000):
    V = np.zeros(nG); policy = np.zeros(nG)
    for it in range(max_iter):
        V_new = np.empty_like(V)
        for iG, G in enumerate(G_grid):
            incs = prod_increment_func(l_grid)
            G_next = np.clip(G + incs, G_min, G_max)
            idx = (G_next - G_min) / (G_max - G_min) * (nG - 1)
            i_lo = np.clip(np.floor(idx).astype(int), 0, nG-1)
            i_hi = np.clip(np.ceil(idx).astype(int), 0, nG-1)
            w = idx - i_lo
            V_interp = (1-w)*V[i_lo] + w*V[i_hi]

            # CONCAVE UTILITY OVER GPA (log / CRRA) 
            Gpos = np.maximum(G_next, _eps)
            if abs(sigma - 1.0) < 1e-12:
                u_g = np.log(Gpos)
            else:
                u_g = (np.power(Gpos, 1.0 - sigma) - 1.0) / (1.0 - sigma)
            u_vals = u_g - chi * l_grid
            # ------------------------------------------------------------

            total = u_vals + beta * V_interp
            best = np.argmax(total)
            V_new[iG] = total[best]
            policy[iG] = l_grid[best]
        if np.max(np.abs(V_new - V)) < tol:
            V = V_new; break
        V = V_new
    return V, policy

chi_non = MP_non_at_mean if (np.isfinite(MP_non_at_mean) and MP_non_at_mean>0) else 0.003
chi_dep = MP_dep_at_mean if (np.isfinite(MP_dep_at_mean) and MP_dep_at_mean>0) else chi_non

# Variant 1: linearized MP (MP_at_mean * l)
V_lin_nd, pol_lin_nd = solve_dp(lambda l: MP_non_at_mean * l, chi_non)
V_lin_dep, pol_lin_dep = solve_dp(lambda l: MP_dep_at_mean * l, chi_dep)

# Variant 2: full quadratic (raw)
V_qf_nd, pol_qf_nd = solve_dp(lambda l: b1*l + b2*(l**2), chi_non)
V_qf_dep, pol_qf_dep = solve_dp(lambda l: (b1+g2)*l + (b2+g3)*(l**2), chi_dep)

# Variant 3 :centered polynomial 
df['Hours_c'] = df['Hours'] - df['Hours'].mean()
model_c = smf.ols("""CGPA ~ Hours_c + I(Hours_c**2) + Depression + Depression:Hours_c + Depression:I(Hours_c**2) + Age + C(Gender) + C(Degree)""", data=df).fit(cov_type='HC1')
b1c = model_c.params.get('Hours_c', 0.0)
b2c = model_c.params.get('I(Hours_c ** 2)', model_c.params.get('I(Hours_c**2)', 0.0))
g2c = model_c.params.get('Depression:Hours_c', 0.0)
g3c = model_c.params.get('Depression:I(Hours_c ** 2)', model_c.params.get('Depression:I(Hours_c**2)', 0.0))

mean_H = df['Hours'].mean()
V_cc_nd, pol_cc_nd = solve_dp(lambda l: b1c*(l - mean_H) + b2c*((l - mean_H)**2), chi_non)
V_cc_dep, pol_cc_dep = solve_dp(lambda l: (b1c+g2c)*(l - mean_H) + (b2c+g3c)*((l - mean_H)**2), chi_dep)

# Save policies and summary
pd.DataFrame({'GPA': G_grid, 'pol_lin_nd': pol_lin_nd, 'pol_lin_dep': pol_lin_dep}).to_csv(os.path.join(OUTDIR,'policy_linear.csv'), index=False)
pd.DataFrame({'GPA': G_grid, 'pol_qf_nd': pol_qf_nd, 'pol_qf_dep': pol_qf_dep}).to_csv(os.path.join(OUTDIR,'policy_quad_full.csv'), index=False)
pd.DataFrame({'GPA': G_grid, 'pol_cc_nd': pol_cc_nd, 'pol_cc_dep': pol_cc_dep}).to_csv(os.path.join(OUTDIR,'policy_center.csv'), index=False)

print("Saved policies to", OUTDIR)
print("Mean GPA:", mean_GPA)
i_mean = np.argmin(np.abs(G_grid - mean_GPA))
print("Optimal hours @ mean GPA (non-dep,dep):")
print(" linear:    ", pol_lin_nd[i_mean], pol_lin_dep[i_mean])
print(" quad_full: ", pol_qf_nd[i_mean], pol_qf_dep[i_mean])
print(" centered:  ", pol_cc_nd[i_mean], pol_cc_dep[i_mean])


# %%

import os
import re
import numpy as np
import matplotlib.pyplot as plt

os.makedirs(OUTDIR, exist_ok=True)

# Boxplot Hours by Depression

d0 = df[df['Depression'] == 0]['Hours'].dropna()
d1 = df[df['Depression'] == 1]['Hours'].dropna()
groups = []
labels = []
if len(d0) > 0:
    groups.append(d0)
    labels.append('No (0)')
if len(d1) > 0:
    groups.append(d1)
    labels.append('Yes (1)')

if len(groups) > 0:
    plt.figure(figsize=(6,5))
    
    try:
        plt.boxplot(groups, tick_labels=labels)
    except Exception:
        plt.boxplot(groups, labels=labels)
    plt.xlabel('Depression'); plt.ylabel('Work/Study Hours')
    plt.title('Work/Study Hours by Depression Status')
    p1 = os.path.join(OUTDIR, "hours_by_depression_boxplot.png")
    plt.savefig(p1, bbox_inches='tight'); plt.close()
else:
    p1 = None
    print("Skipping boxplot: no Hours data available for either group.")

# Scatter Hours vs CGPA with group fits 

plt.figure(figsize=(7,6))
scatter_df = df.dropna(subset=['Hours', 'CGPA'])
if len(scatter_df) > 0:
    plt.scatter(scatter_df['Hours'], scatter_df['CGPA'], alpha=0.05, s=8)

    colors = {0: 'blue', 1: 'orange'}
    for val in [0, 1]:
        sub = df[df['Depression'] == val].dropna(subset=['Hours', 'CGPA'])
        if len(sub) < 2:
            continue
        x = sub['Hours'].to_numpy()
        y = sub['CGPA'].to_numpy()
        try:
            m = np.polyfit(x, y, 1)
            xs = np.array([np.nanmin(x), np.nanmax(x)])
            ys = m[0] * xs + m[1]
            plt.plot(xs, ys, label=f'fit dep={val}', color=colors.get(val, None))
        except np.linalg.LinAlgError:
            pass

    plt.xlabel('Work/Study Hours'); plt.ylabel('CGPA'); plt.title('Hours vs CGPA with group fits')
    plt.legend()
    p3 = os.path.join(OUTDIR, "hours_vs_cgpa_scatter.png")
    plt.savefig(p3, bbox_inches='tight'); plt.close()
else:
    p3 = None
    print("Skipping scatter: need both Hours and CGPA present.")


# DP policy plot

G_grid_arr = np.asarray(G_grid) if 'G_grid' in globals() else np.array([])

policy_pairs = [
    ('pol_lin_nd', 'pol_lin_dep', 'linear'),
    ('pol_qf_nd',  'pol_qf_dep',  'quad_full'),
    ('pol_cc_nd',  'pol_cc_dep',  'centered'),
]

found_any = False
plt.figure(figsize=(10,5))

for lhs, rhs, label_prefix in policy_pairs:
    lhs_exists = lhs in globals()
    rhs_exists = rhs in globals()

    # convert to numpy arrays if present
    lhs_arr = np.asarray(globals()[lhs]) if lhs_exists else None
    rhs_arr = np.asarray(globals()[rhs]) if rhs_exists else None

    if G_grid_arr.size > 0:
        if lhs_arr is not None and lhs_arr.size == G_grid_arr.size:
            plt.plot(G_grid_arr, lhs_arr, label=f'{label_prefix} non-dep')
            found_any = True
        if rhs_arr is not None and rhs_arr.size == G_grid_arr.size:
            plt.plot(G_grid_arr, rhs_arr, label=f'{label_prefix} dep')
            found_any = True
    else:
        if lhs_arr is not None and lhs_arr.size > 0:
            plt.plot(np.arange(lhs_arr.size), lhs_arr, label=f'{label_prefix} non-dep')
            found_any = True
        if rhs_arr is not None and rhs_arr.size > 0:
            plt.plot(np.arange(rhs_arr.size), rhs_arr, label=f'{label_prefix} dep')
            found_any = True

if not found_any:
    for name, val in list(globals().items()):
        if re.search(r'^(pol|policy|opt)_', name, re.I):
            try:
                arr = np.asarray(val)
                if arr.ndim == 1 and arr.size > 0:
                    if G_grid_arr.size > 0 and arr.size == G_grid_arr.size:
                        plt.plot(G_grid_arr, arr, label=name)
                        found_any = True
                    elif G_grid_arr.size == 0:
                        plt.plot(np.arange(arr.size), arr, label=name)
                        found_any = True
            except Exception:
                continue


if found_any:
    if 'mean_GPA' in globals() and mean_GPA is not None and G_grid_arr.size > 0:
        plt.axvline(mean_GPA, color='k', ls='--', label='mean observed CGPA')
    plt.xlabel('GPA' if G_grid_arr.size > 0 else 'index')
    plt.ylabel('Optimal hours')
    plt.title('DP Policy Comparison')
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True)
    p_dp = os.path.join(OUTDIR, "dp_opt_policy.png")
    plt.savefig(p_dp, bbox_inches='tight'); plt.close()
else:
    p_dp = None
    print("Skipping DP policy plot: no policy arrays found with expected names.")

# create a simple combined policy dataframe for convenience
policy_df = pd.DataFrame({
    'GPA': G_grid,
    'pol_lin_nd': pol_lin_nd, 'pol_lin_dep': pol_lin_dep,
    'pol_qf_nd' : pol_qf_nd,  'pol_qf_dep' : pol_qf_dep,
    'pol_cc_nd' : pol_cc_nd,  'pol_cc_dep' : pol_cc_dep
})

def safe_get(name):
    if name in globals():
        try:
            arr = np.asarray(globals()[name])
            return arr
        except Exception:
            pass
    if name in policy_df.columns:
        col = policy_df[name]
        if pd.api.types.is_numeric_dtype(col):
            return col.to_numpy()
        try:
            return np.array(col.tolist(), dtype=float)
        except Exception:
            pass
    return None

G_plot = safe_get('G_grid')
if G_plot is None or len(np.atleast_1d(G_plot)) == 0:
    # try using policy_df GPA column
    if 'GPA' in policy_df.columns and policy_df['GPA'].notna().any():
        G_plot = policy_df['GPA'].to_numpy()
    else:
        G_plot = np.linspace(0,10,201)

# policy arrays 
pol_lin_nd = safe_get('pol_lin_nd')
pol_lin_dep = safe_get('pol_lin_dep')
pol_qf_nd  = safe_get('pol_qf_nd')
pol_qf_dep = safe_get('pol_qf_dep')
pol_cc_nd  = safe_get('pol_cc_nd')
pol_cc_dep = safe_get('pol_cc_dep')

# 1) DP policy comparison (linear / quad / centered)
plt.figure(figsize=(10,6))
plotted_any = False

def plot_policy_pair(xarr, a_non, a_dep, label, color):
    global plotted_any
    if a_non is None or a_dep is None:
        return
    a_non = np.asarray(a_non); a_dep = np.asarray(a_dep)
    if a_non.size == xarr.size and a_dep.size == xarr.size:
        plt.plot(xarr, a_non, linestyle='--', label=f'{label} non-dep', color=color, linewidth=1.6)
        plt.plot(xarr, a_dep,  linestyle='-',  label=f'{label} dep',     color=color, linewidth=2.0)
        plotted_any = True

plot_policy_pair(G_plot, pol_lin_nd, pol_lin_dep, 'Linearized', '#1f77b4')
plot_policy_pair(G_plot, pol_qf_nd,  pol_qf_dep,  'Quad full',  '#ff7f0e')
plot_policy_pair(G_plot, pol_cc_nd,  pol_cc_dep,  'Centered (rec.)', '#2ca02c')

if plotted_any:
    plt.xlabel('GPA')
    plt.ylabel('Optimal Hours (policy)')
    plt.title('DP: Optimal Hours (policy) vs GPA — comparison (log utility)')
    if 'mean_GPA' in globals() and mean_GPA is not None:
        plt.axvline(mean_GPA, color='k', linestyle=':', linewidth=1, label=f'Mean observed GPA = {mean_GPA:.2f}')
    plt.legend(ncol=2, fontsize='small')
    plt.grid(True, linestyle=':', alpha=0.6)
    p_policy_cmp = os.path.join(OUTDIR, "dp_policy_comparison_logutil.png")
    plt.savefig(p_policy_cmp, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved DP policy comparison to:", p_policy_cmp)
else:
    plt.close()
    print("DP policy comparison: required policy arrays not found (pol_*).")

# 2) DP value function plot (centered spec) 
V_cc_nd = safe_get('V_cc_nd') if safe_get('V_cc_nd') is not None else globals().get('V_cc_nd', None)
V_cc_dep = safe_get('V_cc_dep') if safe_get('V_cc_dep') is not None else globals().get('V_cc_dep', None)

if V_cc_nd is not None and V_cc_dep is not None:
    V_cc_nd = np.asarray(V_cc_nd); V_cc_dep = np.asarray(V_cc_dep)
    if V_cc_nd.size == G_plot.size and V_cc_dep.size == G_plot.size:
        plt.figure(figsize=(9,5))
        plt.plot(G_plot, V_cc_nd, label='Value (non-dep)', linestyle='--', linewidth=1.6)
        plt.plot(G_plot, V_cc_dep, label='Value (dep)', linestyle='-', linewidth=1.8)
        plt.xlabel('GPA')
        plt.ylabel('Value (V)')
        plt.title('DP: Value Functions (centered) — log utility')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        p_value_center = os.path.join(OUTDIR, "dp_value_centered_logutil.png")
        plt.savefig(p_value_center, dpi=300, bbox_inches='tight')
        plt.close()
        print("Saved centered DP value functions to:", p_value_center)
    else:
        print("DP value arrays found but length mismatch vs G_grid; skipping value plot.")
else:
    print("DP value plot: V_cc_nd / V_cc_dep not available. (If you want this, ensure V_cc_nd and V_cc_dep are in memory.)")


# -------------------------
# Save summary CSVs
# -------------------------
try:
    group_means.to_csv(os.path.join(OUTDIR, "group_means_by_depression.csv"), index=False)
except Exception as e:
    print("Warning: could not save group_means:", e)
try:
    policy_df.to_csv(os.path.join(OUTDIR, "dp_opt_policy_simple_util.csv"), index=False)
except Exception as e:
    print("Warning: could not save policy_df:", e)

# Report saved files
saved_files = []
if p1: saved_files.append(p1)
if p3: saved_files.append(p3)
if p_dp: saved_files.append(p_dp)
# add new DP outputs if they exist
for fname in ["dp_policy_comparison_logutil.png", "dp_value_centered_logutil.png"]:
    path = os.path.join(OUTDIR, fname)
    if os.path.exists(path):
        saved_files.append(path)

print("\nSaved plots to:", ", ".join(saved_files) if saved_files else "No plots saved.")
print("\nSaved summary tables (if no warnings).")
print("\nDONE.")





# %%



