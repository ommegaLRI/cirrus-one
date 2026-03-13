import pandas as pd
import json
from pathlib import Path
import numpy as np

run_dir = Path("runs/run_TEST_STAGE_G_REALRUN_V2_run_10dbdc582885105e")

particle = pd.read_parquet(run_dir/"inputs/particle.parquet")
alignment = json.load(open(run_dir/"intermediate/alignment.json"))["payload"]

tb = alignment["frame_timebase"]
p2f = alignment["particle_to_frame"]

particle["t_utc"] = pd.to_datetime(particle["t_utc"], utc=True)

print("particle time range:")
print(particle["t_utc"].min(), "→", particle["t_utc"].max())

print()
print("frame_timebase:")
print("t0 =", tb["t0_utc"])
print("dt =", tb["dt_seconds"])
print("source =", tb["source"])

frames = list(p2f.values())

print()
print("particle_to_frame range:")
print(min(frames), "→", max(frames))
print("count =", len(frames))