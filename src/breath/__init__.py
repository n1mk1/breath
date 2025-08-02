import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import find_peaks

# Load audio
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "calm.mp3")
y, sr = librosa.load(file_path, sr=None)

# Compute RMS energy
hop_length = 512
frame_length = 1024
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# Normalize RMS
rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

# Detect peaks
peaks, _ = find_peaks(rms_normalized, height=0.3, distance=20)

# Print peaks
print("Detected Peaks (Time in seconds, Normalized RMS amplitude):")
for peak in peaks:
    print(f"Time: {times[peak]:.3f} s, Amplitude: {rms_normalized[peak]:.3f}")

# Label peaks as Inhale / Exhale
labels = ['Inhale' if i % 2 == 0 else 'Exhale' for i in range(len(peaks))]

# Total breath cycles
total_cycles = len(peaks) // 2
print(f"\nTotal Breath Cycles Detected: {total_cycles}")

# Plot 1: Waveform with Inhale/Exhale labels
plt.figure(figsize=(14, 6))
librosa.display.waveshow(y, sr=sr, alpha=0.6)
plt.plot(times, rms_normalized * np.max(y), color='orange', label='RMS Energy')

for i, peak in enumerate(peaks):
    plt.axvline(times[peak], color='red', linestyle='--', alpha=0.6)
    plt.text(times[peak], 0.8*np.max(y), labels[i], color='black', fontsize=10, rotation=90)

plt.title('Breathing Pattern Detection')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.tight_layout()

# --- SECOND GRAPH: Breathing Rate Per Second (sliding window) ---

# Convert peak times to seconds (rounded down to int)
peak_seconds = np.floor(times[peaks]).astype(int)

# Total duration in seconds
duration = int(np.ceil(times[-1]))

# Initialize breathing rate array
breathing_rate = np.zeros(duration)

# Sliding 3-second window
for sec in range(duration):
    if sec == 0 or sec == duration - 1:
        breathing_rate[sec] = 0
    else:
        # Count breaths from second-1 to second+1
        count = np.sum((peak_seconds >= sec - 1) & (peak_seconds <= sec + 1))
        breathing_rate[sec] = count / 3.0  # average per second

# Plot 2: Breathing rate over time
plt.figure(figsize=(14, 4))
plt.plot(np.arange(duration), breathing_rate, color='blue', marker='o')
plt.title('Breathing Rate Over Time (Sliding 3s Window)')
plt.xlabel('Time (s)')
plt.ylabel('Breaths per Second')
plt.grid(True)
plt.tight_layout()

plt.show()
