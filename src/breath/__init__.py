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

# Compute Root Mean Square Energy (RMS)
hop_length = 512
frame_length = 1024
rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)

# Normalize RMS for easier thresholding
rms_normalized = (rms - np.min(rms)) / (np.max(rms) - np.min(rms))

# Detect peaks (likely breaths)
peaks, _ = find_peaks(rms_normalized, height=0.3, distance=20)  # tweak these values

# Print list of peak coordinates: time and amplitude (normalized) WILL LATER BE UPLOADED TO MONGODB
print("Detected Peaks (Time in seconds, Normalized RMS amplitude):")
for peak in peaks:
    print(f"Time: {times[peak]:.3f} s, Amplitude: {rms_normalized[peak]:.3f}")

# Classify peaks alternately as inhale/exhale
labels = ['Inhale' if i % 2 == 0 else 'Exhale' for i in range(len(peaks))]

# Calculate total number of breath cycles (inhale + exhale)
total_cycles = len(peaks) // 2
print(f"\nTotal Breath Cycles Detected: {total_cycles}")

# Plot
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
plt.show()
