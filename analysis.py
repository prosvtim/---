# analysis.py
# Статистический анализ: z-test, bootstrap (оптимизированный), CUPED, выводы

import pandas as pd
import numpy as np
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.proportion import proportions_ztest
from scipy import stats

DATA_PATH = "ab_test_data.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

df = pd.read_csv(DATA_PATH)

# Precompute buckets
bucket_a = df[df['bucket']=='A']
bucket_b = df[df['bucket']=='B']

# Aggregated counts for proportion tests
success_a = bucket_a['purchases'].sum()
nobs_a = bucket_a['clicks'].sum()
success_b = bucket_b['purchases'].sum()
nobs_b = bucket_b['clicks'].sum()

# Z-test (two-sided or one-sided). Здесь ожидаем uplift B > A -> alternative='larger'
count = np.array([success_b, success_a])
nobs = np.array([nobs_b, nobs_a])
z_stat, p_val = proportions_ztest(count, nobs, alternative='larger')
print(f"proportions_ztest (B > A): z={z_stat:.3f}, p={p_val:.6f}")

# --- Оптимизированный bootstrap для разницы долей (CR_B - CR_A)
def bootstrap_cr_fast(a_clicks, a_successes, b_clicks, b_successes, n_boot=5000, seed=42):
    rng = np.random.default_rng(seed)
    pA = a_successes / a_clicks
    pB = b_successes / b_clicks
    diffs = rng.binomial(a_clicks, pA, size=n_boot) / a_clicks - rng.binomial(b_clicks, pB, size=n_boot) / b_clicks
    # diffs currently = boot_A/a_clicks - boot_B/b_clicks, need B-A
    diffs = -diffs  # now diffs = B - A
    lower, med, upper = np.percentile(diffs, [2.5, 50, 97.5])
    return diffs, (lower, med, upper)

diffs, (ci_low, ci_med, ci_high) = bootstrap_cr_fast(nobs_a, success_a, nobs_b, success_b, n_boot=5000)
print(f"Bootstrap CR diff (B - A) 95% CI: [{ci_low:.6f}, {ci_high:.6f}], median={ci_med:.6f}")
pd.Series(diffs).to_csv(os.path.join(PLOTS_DIR, "bootstrap_cr_diffs.csv"), index=False)

# --- Revenue: bootstrap mean-diff (robust to heavy-tail)
def bootstrap_mean_diff(seriesA, seriesB, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    a = seriesA.values
    b = seriesB.values
    nA = len(a); nB = len(b)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        sampA = rng.choice(a, size=nA, replace=True)
        sampB = rng.choice(b, size=nB, replace=True)
        diffs[i] = sampB.mean() - sampA.mean()
    lower, med, upper = np.percentile(diffs, [2.5, 50, 97.5])
    return diffs, (lower, med, upper)

rev_diffs, (rev_low, rev_med, rev_high) = bootstrap_mean_diff(bucket_a['revenue'], bucket_b['revenue'], n_boot=1000)
print(f"Bootstrap mean revenue diff (B - A) 95% CI: [{rev_low:.4f}, {rev_high:.4f}], median={rev_med:.4f}")
pd.Series(rev_diffs).to_csv(os.path.join(PLOTS_DIR, "bootstrap_rev_diffs.csv"), index=False)

# --- t-test for revenue (Welch)
t_stat, t_p = stats.ttest_ind(bucket_b['revenue'], bucket_a['revenue'], equal_var=False)
print(f"t-test revenue (B vs A): t={t_stat:.3f}, p={t_p:.6f}")

# --- CUPED: уменьшаем дисперсию revenue с помощью historical_spend
X = df['historical_spend'].values
Y = df['revenue'].values
# theta = cov(Y,X) / var(X)
theta = np.cov(Y, X, ddof=0)[0,1] / np.var(X)
df['revenue_cuped'] = df['revenue'] - theta * (df['historical_spend'] - np.mean(df['historical_spend']))

# Сравнение средних revenue_cuped по бакетам (bootstrap + t-test)
bucket_a_c = df[df['bucket']=='A']
bucket_b_c = df[df['bucket']=='B']

# t-test on cuped
t_c, p_c = stats.ttest_ind(bucket_b_c['revenue_cuped'], bucket_a_c['revenue_cuped'], equal_var=False)
print(f"CUPED t-test: t={t_c:.3f}, p={p_c:.6f}")

# bootstrap on cuped means
cuped_diffs, (cuped_low, cuped_med, cuped_high) = bootstrap_mean_diff(bucket_a_c['revenue_cuped'], bucket_b_c['revenue_cuped'], n_boot=1000)
print(f"CUPED mean diff 95% CI (B - A): [{cuped_low:.4f}, {cuped_high:.4f}]")
pd.Series(cuped_diffs).to_csv(os.path.join(PLOTS_DIR, "bootstrap_cuped_rev_diffs.csv"), index=False)

# --- Сохраняем ключевые результаты
results = {
    "z_stat": float(z_stat),
    "z_pvalue": float(p_val),
    "cr_bootstrap_ci": [float(ci_low), float(ci_med), float(ci_high)],
    "rev_bootstrap_ci": [float(rev_low), float(rev_med), float(rev_high)],
    "t_stat_revenue": float(t_stat),
    "t_p_revenue": float(t_p),
    "cuped_theta": float(theta),
    "cuped_t": float(t_c),
    "cuped_p": float(p_c),
    "cuped_bootstrap_ci": [float(cuped_low), float(cuped_med), float(cuped_high)]
}
with open(os.path.join(PLOTS_DIR, "stat_results.json"), "w") as f:
    json.dump(results, f, indent=2)

# --- Визуализация bootstrap распределений
plt.figure()
sns.histplot(diffs, bins=60)
plt.axvline(ci_low, color='red', linestyle='--')
plt.axvline(ci_med, color='blue', linestyle='--')
plt.axvline(ci_high, color='red', linestyle='--')
plt.title('Bootstrap distribution of CR diffs (B - A)')
plt.savefig(os.path.join(PLOTS_DIR, "bootstrap_cr_diff_hist.png"))
plt.close()

plt.figure()
sns.histplot(rev_diffs, bins=60)
plt.axvline(rev_low, color='red', linestyle='--')
plt.axvline(rev_med, color='blue', linestyle='--')
plt.axvline(rev_high, color='red', linestyle='--')
plt.title('Bootstrap distribution of revenue mean diffs (B - A)')
plt.savefig(os.path.join(PLOTS_DIR, "bootstrap_rev_diff_hist.png"))
plt.close()

print("All results saved to:", PLOTS_DIR)

