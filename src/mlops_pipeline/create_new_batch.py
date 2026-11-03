"""Create the SIMULATED incoming batch (Jan 1 - Mar 31, 2013, 90 days) used by Lab 13.

The public Bike Sharing dataset ends on 2012-12-31, so no real 2013 data exists.
This script derives a plausible batch from the same calendar days of 2012
(seeded, reproducible): weather is perturbed, demand grows ~6% with noise, and
calendar fields (weekday/holiday/workingday/season) are recomputed for 2013.

    python src/mlops_pipeline/create_new_batch.py [output_path]
"""
import sys

import numpy as np
import pandas as pd

from common import load_config

HOLIDAYS_2013 = {"2013-01-01", "2013-01-21", "2013-02-18"}


def main():
    config = load_config()
    out = sys.argv[1] if len(sys.argv) > 1 else "data/raw/new_bike.csv.staged"
    rng = np.random.default_rng(config["monitoring"]["random_seed"])
    df = pd.read_csv(config["data"]["raw_path"])
    last = df.sort_values("instant").iloc[-1]
    dates = pd.date_range("2013-01-01", "2013-03-31", freq="D")
    base = df[(df.yr == 1) & (df.mnth <= 3)].sort_values("dteday").reset_index(drop=True)
    assert len(base) >= len(dates) - 1

    rows = []
    for i, d in enumerate(dates):
        src = base.iloc[min(i, len(base) - 1)]
        temp = float(np.clip(src.temp + rng.normal(0, 0.03), 0.05, 0.95))
        atemp = float(np.clip(src.atemp + (temp - src.temp) + rng.normal(0, 0.01), 0.05, 0.95))
        hum = float(np.clip(src.hum + rng.normal(0, 0.05), 0.15, 0.98))
        wind = float(np.clip(src.windspeed + rng.normal(0, 0.03), 0.02, 0.5))
        weather = int(src.weathersit)
        if rng.random() < 0.05:
            weather = int(min(3, weather + 1))
        cnt = int(max(50, src.cnt * 1.06 * rng.lognormal(0, 0.08)))
        share = float(src.casual) / float(src.cnt)
        casual = int(round(cnt * np.clip(share + rng.normal(0, 0.02), 0.02, 0.5)))
        weekday = (d.dayofweek + 1) % 7  # dataset convention: 0 = Sunday
        holiday = int(d.strftime("%Y-%m-%d") in HOLIDAYS_2013)
        rows.append({
            "instant": int(last.instant) + 1 + i, "dteday": d.strftime("%Y-%m-%d"),
            "season": 1 if d < pd.Timestamp("2013-03-21") else 2, "yr": 2, "mnth": d.month,
            "holiday": holiday, "weekday": weekday,
            "workingday": int(weekday in range(1, 6) and not holiday), "weathersit": weather,
            "temp": round(temp, 6), "atemp": round(atemp, 6), "hum": round(hum, 6), "windspeed": round(wind, 6),
            "casual": casual, "registered": cnt - casual, "cnt": cnt,
        })
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[SUCCESS] Simulated batch written to {out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
