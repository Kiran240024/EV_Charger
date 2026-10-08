import pandas as pd
import numpy as np
from pathlib import Path

INPUT = Path("data/raw/uncontrolled_bat_0.5_office_10.csv")
OUTPUT = Path("data/processed/ev_charging_processed.csv")

df = pd.read_csv(INPUT)
df["timestamps"] = pd.to_datetime(df["timestamps"], errors="coerce")
df = df.dropna(subset=["timestamps"]).sort_values("timestamps").reset_index(drop=True)

connection_cols = [f"EV_{i}.connected" for i in range(10)]
soc_cols = [f"soc.EV_{i}" for i in range(10)]
power_cols = [f"power_applied.EV_{i}" for i in range(10)]

for col in connection_cols:
    df[col] = df[col].fillna(False).astype(int)

df["hour"] = df["timestamps"].dt.hour
df["minute"] = df["timestamps"].dt.minute
df["minutes_since_midnight"] = df["hour"] * 60 + df["minute"]
df["hour_sin"] = np.sin(2 * np.pi * df["minutes_since_midnight"] / 1440)
df["hour_cos"] = np.cos(2 * np.pi * df["minutes_since_midnight"] / 1440)

for soc_col, conn_col in zip(soc_cols, connection_cols):
    df.loc[df[conn_col] == 0, soc_col] = 0.0

charging_power_cols = []
for i, col in enumerate(power_cols):
    new_col = f"charging_power.EV_{i}"
    df[new_col] = -df[col]
    charging_power_cols.append(new_col)

keep_cols = (
    ["timestamps"] + soc_cols + connection_cols +
    ["prices",
     "fixed_power_contribution_per_component.load",
     "fixed_power_contribution_per_component.pv"] +
    charging_power_cols + ["grid_builder_usage"] +
    ["hour", "minute", "minutes_since_midnight", "hour_sin", "hour_cos"]
)

processed = df[keep_cols].copy()
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
processed.to_csv(OUTPUT, index=False)

print(f"Saved {len(processed)} rows and {len(processed.columns)} columns to {OUTPUT}")