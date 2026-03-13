import pandas as pd
from pathlib import Path

run_dir = Path("runs/run_TEST_STAGE_G_REALRUN_run_d691021a64444360")

particle = pd.read_parquet(run_dir / "inputs/particle.parquet")
events = pd.read_parquet(run_dir / "intermediate/event_catalog.parquet")

particle_times = pd.to_datetime(particle["t_utc"])
event_times = pd.to_datetime(events["t_peak_utc"])

offset = event_times.median() - particle_times.median()

print("\nEstimated time offset:")
print(offset)