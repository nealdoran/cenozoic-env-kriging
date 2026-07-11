"""
generate_sample_data.py
Generates a geographically plausible synthetic observation dataset
for pipeline demonstration. NOT real PBDB or Macrostrat data.

Produces data/sample_observations.csv (~400 rows).
Run once: python generate_sample_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path

rng = np.random.default_rng(42)

EPOCH_AGES = {
    "Paleogene":  (23.03, 66.0),
    "Neogene":    (2.58,  23.03),
    "Quaternary": (0.0,   2.58),
}

# Geographic clusters: (center_lat, center_lng, n_points, env, label)
# Designed to be geographically plausible but entirely synthetic
CLUSTERS = [
    # North America — terrestrial interior
    (42.0, -105.0,  40, "terrestrial", "macrostrat", 5.0, "Paleogene"),
    (45.0,  -98.0,  30, "terrestrial", "fossil",     0.5, "Paleogene"),
    (38.0, -100.0,  25, "terrestrial", "fossil",     0.5, "Neogene"),
    # Gulf Coast / Atlantic margin — marine
    (30.0,  -88.0,  20, "marine",      "macrostrat", 5.0, "Paleogene"),
    (32.0,  -82.0,  15, "marine",      "fossil",     0.5, "Paleogene"),
    (28.0,  -95.0,  10, "marine",      "fossil",     0.5, "Neogene"),
    # Europe — mixed
    (48.0,   12.0,  25, "marine",      "fossil",     0.5, "Neogene"),
    (45.0,   15.0,  20, "marine",      "macrostrat", 5.0, "Paleogene"),
    (52.0,    8.0,  15, "terrestrial", "fossil",     0.5, "Quaternary"),
    # Mediterranean / Tethys — marine
    (36.0,   22.0,  30, "marine",      "fossil",     0.5, "Paleogene"),
    (38.0,   28.0,  20, "marine",      "macrostrat", 5.0, "Paleogene"),
    (35.0,   10.0,  15, "marine",      "fossil",     0.5, "Neogene"),
    # North Africa — Tethyan marine
    (28.0,   25.0,  12, "marine",      "fossil",     0.5, "Paleogene"),
    (24.0,   16.0,   8, "marine",      "fossil",     0.5, "Paleogene"),
    # South America — terrestrial interior
    (-15.0,  -60.0, 20, "terrestrial", "fossil",     0.5, "Paleogene"),
    (-25.0,  -65.0, 15, "terrestrial", "fossil",     0.5, "Neogene"),
    (-10.0,  -55.0, 10, "terrestrial", "macrostrat", 5.0, "Paleogene"),
    # South America — coastal marine
    (-35.0,  -58.0, 12, "marine",      "fossil",     0.5, "Paleogene"),
    (-40.0,  -62.0,  8, "marine",      "fossil",     0.5, "Neogene"),
    # Asia — mixed
    (35.0,   80.0,  15, "terrestrial", "fossil",     0.5, "Paleogene"),
    (40.0,   70.0,  10, "terrestrial", "fossil",     0.5, "Neogene"),
    (30.0,   90.0,  10, "marine",      "fossil",     0.5, "Paleogene"),
    # Australia
    (-25.0,  133.0,  12, "terrestrial","fossil",     0.5, "Neogene"),
    (-32.0,  138.0,   8, "marine",     "fossil",     0.5, "Paleogene"),
    # Quaternary coastal spots
    (51.0,    1.0,   8, "marine",      "fossil",     0.5, "Quaternary"),
    (38.0,  -76.0,   8, "marine",      "fossil",     0.5, "Quaternary"),
    (-33.0,  18.0,   6, "marine",      "fossil",     0.5, "Quaternary"),
    # Marginal environments
    (29.0,  -90.0,   8, "marginal",    "fossil",     0.5, "Paleogene"),
    (45.0,   13.0,   6, "marginal",    "fossil",     0.5, "Neogene"),
]

rows = []
for (clat, clng, n, env, source, weight, epoch) in CLUSTERS:
    age_lo, age_hi = EPOCH_AGES[epoch]
    lats = rng.normal(clat, 2.5, n)
    lngs = rng.normal(clng, 3.0, n)
    ages = rng.uniform(age_lo, age_hi, n)
    # Clip to valid range
    lats = np.clip(lats, -85, 85)
    lngs = np.clip(lngs, -180, 180)
    for lat, lng, age in zip(lats, lngs, ages):
        rows.append({
            "lat":         round(float(lat), 4),
            "lng":         round(float(lng), 4),
            "environment": env,
            "age_ma":      round(float(age), 3),
            "source":      source,
            "weight":      weight,
        })

df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle

Path("data").mkdir(exist_ok=True)
df.to_csv("data/sample_observations.csv", index=False)
print(f"Generated {len(df):,} synthetic observations → data/sample_observations.csv")
print(df.groupby(["source","environment"]).size().unstack(fill_value=0).to_string())
print("\nNOTE: This is synthetic data for pipeline demonstration only.")
print("      It is NOT real PBDB or Macrostrat data.")
