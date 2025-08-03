import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks
from pprint import pprint
from pymongo import MongoClient

# ------------------ LOAD AUDIO ------------------
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "calm.mp3")
y, sr = librosa.load(file_path, sr=None)

# ------------------ RMS ENERGY ------------------
hop_length = 512
frame_length = 1024
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# Normalize RMS
rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

# ------------------ PEAK DETECTION ------------------
peaks, _ = find_peaks(rms_normalized, height=0.3, distance=20)
labels = ['Inhale' if i % 2 == 0 else 'Exhale' for i in range(len(peaks))]
total_cycles = len(peaks) // 2

# ------------------ BREATH INTENSITY ------------------
intensity_window = 5
intensities = []
for peak in peaks:
    start = max(0, peak - intensity_window)
    end = min(len(rms_normalized), peak + intensity_window)
    intensity = np.sum(rms_normalized[start:end])
    intensities.append(intensity)

# ------------------ TABLE 1: BREATH EVENTS ------------------
breath_events = []
for i, peak in enumerate(peaks):
    breath_events.append({
        "index": i + 1,
        "time_sec": round(float(times[peak]), 3),
        "amplitude": round(float(rms_normalized[peak]), 3),
        "label": labels[i],
        "intensity": round(float(intensities[i]), 3)
    })

# ------------------ BREATHING RATE OVER TIME ------------------
peak_seconds = np.floor(times[peaks]).astype(int)
duration = int(np.ceil(times[-1]))
breathing_rate = np.zeros(duration)
for sec in range(duration):
    if sec == 0 or sec == duration - 1:
        breathing_rate[sec] = 0
    else:
        count = np.sum((peak_seconds >= sec - 1) & (peak_seconds <= sec + 1))
        breathing_rate[sec] = count / 3.0

# ------------------ TABLE 2: BREATHING RATE ------------------
breathing_rate_over_time = []
for sec in range(duration):
    breathing_rate_over_time.append({
        "second": sec,
        "breaths_per_sec": round(float(breathing_rate[sec]), 3)
    })

# ------------------ STRUCTURE FOR MONGODB ------------------
breathing_data = {
    "audio_file": "calm.mp3",
    "sample_rate": sr,
    "duration_seconds": round(times[-1], 3),
    "total_breath_cycles": total_cycles,
    "breath_events": breath_events,
    "breathing_rate_over_time": breathing_rate_over_time
}

# ------------------ INSERT INTO MONGODB ------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["breathing_analysis"]
collection = db["records"]
collection.insert_one(breathing_data)

# ------------------ PRINT TABLES ------------------
print("\n=== Table 1: Breath Events ===")
pprint(breath_events)

print("\n=== Table 2: Breathing Rate Over Time ===")
pprint(breathing_rate_over_time)

# ------------------ PLOT 1: Waveform with Labels ------------------
plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.plot(times, rms_normalized * np.max(y), color='orange', label='RMS Energy')

for i, peak in enumerate(peaks):
    plt.axvline(times[peak], color='red', linestyle='--', alpha=0.6)
    plt.text(times[peak], 0.8 * np.max(y), labels[i], color='black', fontsize=10, rotation=90)

plt.title('Breathing Pattern Detection')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()

# ------------------ PLOT 2: Breathing Rate ------------------
plt.figure(figsize=(14, 4))
plt.plot(np.arange(duration), breathing_rate, color='blue', marker='o')
plt.title('Breathing Rate Over Time (Sliding 3s Window)')
plt.xlabel('Time (s)')
plt.ylabel('Breaths per Second')
plt.grid(True)
plt.tight_layout()

plt.show()


