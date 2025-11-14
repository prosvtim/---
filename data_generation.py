# data_generation.py
# Генерация синтетических данных для A/B-теста (1_000_000 пользователей)

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1_000_000  # количество пользователей

# user_id
user_id = np.arange(1, N + 1)

# Источники и устройства
sources = ["SEO", "PPC", "Direct", "Social", "Referral"]
devices = ["mobile", "desktop"]

source = np.random.choice(sources, size=N, p=[0.25,0.25,0.2,0.2,0.1])
device = np.random.choice(devices, size=N, p=[0.6,0.4])

# Исторические траты (ковариата для CUPED) — экспоненциальное распределение (long-tail)
historical_spend = np.round(np.random.exponential(scale=5, size=N), 2)

# Разбиение на бакеты (рандомно по user_id)
bucket = np.where(np.mod(user_id, 2) == 0, "A", "B")

# Импрессии (Poisson) и base CTR по source
impressions = np.random.poisson(lam=3, size=N)
ctr_map = {"SEO":0.08, "PPC":0.12, "Direct":0.10, "Social":0.05, "Referral":0.07}
base_ctr = np.array([ctr_map[s] for s in source])

# Клики (биномиально от impressions)
def safe_binomial(n, p):
    # n и p — numpy arrays
    return np.array([np.random.binomial(int(n_i), float(p_i)) if int(n_i)>0 else 0
                     for n_i, p_i in zip(n, p)])

clicks = safe_binomial(impressions, base_ctr)

# Base CR по source, добавляем uplift в группе B (+3% относительный)
cr_map = {"SEO":0.04, "PPC":0.05, "Direct":0.045, "Social":0.02, "Referral":0.03}
base_cr = np.array([cr_map[s] for s in source])
uplift = 0.03
cr_adjusted = np.where(bucket == "B", base_cr * (1 + uplift), base_cr)

# Purchases (биномиально от clicks, с учётом cr_adjusted)
purchases = safe_binomial(clicks, cr_adjusted)

# Revenue: длинный хвост, экспонента на покупку
revenue = purchases * np.round(np.random.exponential(scale=20, size=N), 2)

# Собираем DataFrame
df = pd.DataFrame({
    "user_id": user_id,
    "source": source,
    "device": device,
    "bucket": bucket,
    "historical_spend": historical_spend,
    "impressions": impressions,
    "clicks": clicks,
    "purchases": purchases,
    "revenue": revenue
})

# Сохраняем
df.to_csv("ab_test_data.csv", index=False)
print("Saved ab_test_data.csv with", len(df), "rows")

