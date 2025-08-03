from pymongo import MongoClient
import matplotlib.pyplot as plt
import numpy as np

# ------------------ LOAD DATA ------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["breathing_analysis"]
collection = db["records"]

# Load latest record
data = collection.find_one(sort=[("_id", -1)])
rate_data = data["breathing_rate_over_time"]
duration = int(data["duration_seconds"])
filename = data["audio_file"]

# Convert to breaths per minute
breaths_per_min = [entry["breaths_per_sec"] * 60 for entry in rate_data]

# ------------------ SLEEP STAGE CLASSIFICATION ------------------
sleep_stages = []
stage_labels = []

for bpm in breaths_per_min:
    if bpm > 18:
        stage = "Awake/REM"
    elif bpm > 12:
        stage = "Light Sleep"
    elif bpm >= 8:
        stage = "Deep Sleep"
    else:
        stage = "Artifact/Possibly Awake"
    
    sleep_stages.append(stage)
    stage_labels.append(stage[:1])  # A, L, D, etc.

# ------------------ PLOT ------------------
plt.figure(figsize=(14, 5))
plt.plot(breaths_per_min, label="Breathing Rate (BPM)", color='blue')
plt.title(f"Estimated Sleep Stages - {filename}")
plt.xlabel("Time (s)")
plt.ylabel("Breaths per Minute")
plt.grid(True)

# Color-coded stage shading
for i, stage in enumerate(sleep_stages):
    color = {
        "Awake/REM": "red",
        "Light Sleep": "yellow",
        "Deep Sleep": "green",
        "Artifact/Possibly Awake": "gray"
    }[stage]
    plt.axvspan(i - 0.5, i + 0.5, color=color, alpha=0.2)

plt.legend()
plt.tight_layout()
plt.show()

# ------------------ PRINT SUMMARY ------------------
from collections import Counter
summary = Counter(sleep_stages)
print("\n=== Sleep Stage Summary ===")
for stage, count in summary.items():
    percent = 100 * count / len(sleep_stages)
    print(f"{stage}: {count} seconds ({percent:.1f}%)")
