# train full rockyou
import math, re, pickle
from collections import Counter
import numpy as np
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb   # pip install lightgbm
from pwd import AIPasswordAnalyzer  # or copy extract_features here

ROCKYOU = "rockyou.txt"
MAX_UNIQUE = 2_000_000          # or None for all unique
TARGET_SAMPLES = 1_000_000      # total training examples to sample (weighted)
CHUNK = 100_000

print(f'Open {ROCKYOU}')
freq = Counter()
with open(ROCKYOU, "r", encoding="latin-1", errors="ignore") as f:
    for line in f:
        pwd = line.strip()
        if 4 <= len(pwd) <= 30:
            freq[pwd] += 1
print(f'Sort')
unique_pwds = list(freq.keys())
counts = np.array([freq[p] for p in unique_pwds], dtype=float)

print(f'Compute rank map')
order = counts.argsort()[::-1]
percentiles = np.linspace(0.0, 1.0, num=len(unique_pwds), endpoint=False)
rank_map = { unique_pwds[idx]: percentiles[pos] for pos, idx in enumerate(order) }

print(f'Create set')
if MAX_UNIQUE and MAX_UNIQUE < len(unique_pwds):
    top_k = set([unique_pwds[i] for i in order[:MAX_UNIQUE//2]])
    tail = [unique_pwds[i] for i in order[MAX_UNIQUE//2:]]
    rng = np.random.default_rng(42)
    tail_sample = list(rng.choice(tail, size=MAX_UNIQUE - len(top_k), replace=False))
    pool = list(top_k) + tail_sample
else:
    pool = unique_pwds

print(f'weigthed sample')
weights = np.array([freq[p] for p in pool], dtype=float)
weights = weights / weights.sum()
rng = np.random.default_rng(42)
sample_indices = rng.choice(len(pool), size=TARGET_SAMPLES, p=weights)
sampled_pwds = [pool[i] for i in sample_indices]

print(f'Password analyzer')
analyzer = AIPasswordAnalyzer(use_pretrained=False)  
X_chunks, y_chunks = [], []
for i in range(0, len(sampled_pwds), CHUNK):
    chunk = sampled_pwds[i:i+CHUNK]
    Xc = np.vstack([analyzer.extract_features(p) for p in chunk])
    ranks = np.array([rank_map[p] for p in chunk])
    log_seconds = np.zeros(len(ranks))
    log_seconds[ranks < 0.01] = np.random.uniform(-2, 0, np.sum(ranks < 0.01))
    log_seconds[(ranks >= 0.01) & (ranks < 0.1)] = np.random.uniform(0, 2, np.sum((ranks >= 0.01) & (ranks < 0.1)))
    log_seconds[(ranks >= 0.1) & (ranks < 0.5)] = np.random.uniform(2, 6, np.sum((ranks >= 0.1) & (ranks < 0.5)))
    log_seconds[ranks >= 0.5] = np.random.uniform(6, 15, np.sum(ranks >= 0.5))
    log_seconds = log_seconds + (Xc[:,2] / 50) * 2 - Xc[:,3] * 5
    X_chunks.append(Xc); y_chunks.append(log_seconds)

print(f'Some np stuff')
X = np.vstack(X_chunks)
y = np.concatenate(y_chunks)

print(f'train LightGBM')
scaler = StandardScaler().fit(X)
Xs = scaler.transform(X)
dtrain = lgb.Dataset(Xs, label=y)
params = {"objective":"regression", "metric":"l2", "num_leaves":255, "learning_rate":0.05}
bst = lgb.train(
    params,
    dtrain,
    num_boost_round=1000,
    valid_sets=[dtrain],
    callbacks=[
        lgb.early_stopping(50),
        lgb.log_evaluation(50)
    ]
)


print(f'Saving')
pickle.dump(bst, open("lgb_crack_regressor.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
print("Done")
