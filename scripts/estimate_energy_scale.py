import pandas as pd
from pathlib import Path

run_dir = Path("runs/run_TEST_STAGE_G_REALRUN_run_ab350e14a589999a")

particle = pd.read_parquet(run_dir / "inputs" / "particle.parquet")
events = pd.read_parquet(run_dir / "intermediate" / "event_catalog.parquet")

# crude time matching
particle["t_utc"] = pd.to_datetime(particle["t_utc"])
events["t_peak_utc"] = pd.to_datetime(events["t_peak_utc"])

merged = pd.merge_asof(
    particle.sort_values("t_utc"),
    events.sort_values("t_peak_utc"),
    left_on="t_utc",
    right_on="t_peak_utc",
    direction="nearest",
    tolerance=pd.Timedelta("2s")
)

merged = merged.dropna(subset=["energy_proxy_E", "mass_mg"])

merged["scale"] = merged["mass_mg"] / merged["energy_proxy_E"]

print("\nMatched events:", len(merged))
print("\nEnergy scale estimate (mass_mg / energy):")
print(merged["scale"].describe())