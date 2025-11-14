# eda.py
# Разведочный анализ, метрики и визуализации (работает с ab_test_data.csv)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
import os
from statsmodels.stats.proportion import proportions_ztest

sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10,6)

DATA_PATH = "ab_test_data.csv"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# ---------------------------
# 1. Загрузка
# ---------------------------
df = pd.read_csv(DATA_PATH)
print("Loaded:", len(df), "rows")
print(df.info())
print(df.describe())
print("NaNs:\n", df.isna().sum())

# Добавляем пользовательский CR (purchases / clicks). clicks==0 -> NaN
df['CR'] = df['purchases'] / df['clicks'].replace(0, np.nan)

# ---------------------------
# 2. Баланс бакетов и источников
# ---------------------------
print("Bucket counts:\n", df['bucket'].value_counts(normalize=False))
ct_source = pd.crosstab(df['source'], df['bucket'])
print("Source x Bucket:\n", ct_source)

# ---------------------------
# 3. Агрегированные метрики по бакетам
# ---------------------------
agg = df.groupby('bucket').agg(
    users=('user_id','nunique'),
    impressions=('impressions','sum'),
    clicks=('clicks','sum'),
    purchases=('purchases','sum'),
    revenue=('revenue','sum')
).reset_index()

agg['CTR'] = agg['clicks'] / agg['impressions']
agg['CR'] = agg['purchases'] / agg['clicks'].replace(0, np.nan)
agg['ARPU'] = agg['revenue'] / agg['users']
agg['ARPPU'] = agg['revenue'] / agg['purchases'].replace(0, np.nan)
print("Aggregates by bucket:\n", agg)
agg.to_csv(os.path.join(PLOTS_DIR, "agg_by_bucket.csv"), index=False)

# ---------------------------
# 4. SRM: chi2 for source x bucket
# ---------------------------
chi2, p, dof, exp = chi2_contingency(ct_source)
print(f"SRM chi2 p-value: {p:.6f}")

# ---------------------------
# 5. Быстрые визуализации (с учётом больших данных)
# ---------------------------
# Revenue distribution (truncated at 95th percentile)
df_pos = df[df['revenue'] > 0]
if len(df_pos) > 0:
    x_max = df_pos['revenue'].quantile(0.95)
    plt.figure()
    sns.histplot(df_pos['revenue'], bins=80)
    plt.xlim(0, x_max)
    plt.title('Revenue distribution (truncated at 95th percentile)')
    plt.xlabel('Revenue per user')
    plt.ylabel('Count')
    plt.savefig(os.path.join(PLOTS_DIR, "revenue_distribution_truncated.png"))
    plt.close()

# CR: KDE and histogram (dropna)
plt.figure()
sns.histplot(df.loc[df['bucket']=='A','CR'].dropna(), bins=50, label='A', stat='density', alpha=0.6)
sns.histplot(df.loc[df['bucket']=='B','CR'].dropna(), bins=50, label='B', stat='density', alpha=0.6)
plt.legend()
plt.title('CR histogram by bucket (density)')
plt.savefig(os.path.join(PLOTS_DIR, "cr_histogram.png"))
plt.close()

plt.figure()
sns.kdeplot(df.loc[df['bucket']=='A','CR'].dropna(), label='A', fill=True)
sns.kdeplot(df.loc[df['bucket']=='B','CR'].dropna(), label='B', fill=True)
plt.title('CR KDE by bucket')
plt.savefig(os.path.join(PLOTS_DIR, "cr_kde.png"))
plt.close()

# Boxplot CR (no points)
plt.figure()
sns.boxplot(x='bucket', y='CR', data=df)
plt.title('CR boxplot by bucket')
plt.savefig(os.path.join(PLOTS_DIR, "cr_boxplot.png"))
plt.close()

# Optional: sample points for visual check (1% sample)
sampled = df.sample(frac=0.01, random_state=42)
plt.figure()
sns.stripplot(x='bucket', y='CR', data=sampled, color='black', size=2, jitter=True)
plt.title('CR sample points (1% of users)')
plt.savefig(os.path.join(PLOTS_DIR, "cr_points_sample.png"))
plt.close()

# ARPU by bucket (mean revenue per user)
df_bucket = df.groupby('bucket').agg(ARPU=('revenue','mean')).reset_index()
plt.figure()
sns.barplot(x='bucket', y='ARPU', data=df_bucket)
plt.title('ARPU by bucket (mean revenue per user)')
plt.savefig(os.path.join(PLOTS_DIR, "arpu_by_bucket.png"))
plt.close()

print("Plots saved to:", PLOTS_DIR)
