import pandas as pd
from pathlib import Path

run_dir = Path("runs/run_TEST_STAGE_G_REALRUN_run_d691021a64444360")

particle = pd.read_parquet(run_dir / "inputs/particle.parquet")
events = pd.read_parquet(run_dir / "intermediate/event_catalog.parquet")

print("Particle timestamps:")
print(particle["t_utc"].head())

print("\nEvent peak timestamps:")
print(events["t_peak_utc"].head())

print("\nParticle positions:")
print(particle[["x_px", "y_px"]].head())

print("\nEvent centroids:")
print(events[["centroid_start_x", "centroid_start_y"]].head())