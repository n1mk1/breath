import librosa
import numpy as np
from collections import Counter

def freq_to_note(freq):
    """Convert a frequency (Hz) to the nearest musical note name."""
    A4 = 440.0
    if freq <= 0:
        return "Rest"
    note_num = 12 * np.log2(freq / A4) + 69
    note_index = int(round(note_num)) % 12
    notes = ['C', 'C#', 'D', 'D#', 'E', 'F', 
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    return notes[note_index]

def smooth_notes(note_sequence, window_size=5):
    """Smooth the note sequence by majority vote in a sliding window."""
    smoothed = []
    half_win = window_size // 2
    length = len(note_sequence)

    for i in range(length):
        window_notes = note_sequence[max(0, i - half_win): min(length, i + half_win + 1)]
        most_common_note = Counter(window_notes).most_common(1)[0][0]
        smoothed.append(most_common_note)

    return smoothed

def extract_notes(file_path, mag_threshold=0.1):
    """Load audio file, extract pitch over time, filter by magnitude, and smooth notes."""
    y, sr = librosa.load(file_path)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=50.0, fmax=500.0)

    note_sequence = []
    pitch_sequence = []
    for i in range(pitches.shape[1]):
        index = np.argmax(magnitudes[:, i])
        pitch = pitches[index, i]
        mag = magnitudes[index, i]

        if mag < mag_threshold or pitch <= 0:
            note = "Rest"
        else:
            note = freq_to_note(pitch)

        note_sequence.append(note)
        pitch_sequence.append(pitch if mag >= mag_threshold else 0)

    # Smooth the notes to reduce spurious fluctuations
    smoothed_notes = smooth_notes(note_sequence, window_size=7)

    return smoothed_notes, pitch_sequence, sr

if __name__ == "__main__":
    import os
    from pprint import pprint
    import matplotlib.pyplot as plt

    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "calm.mp3")

    notes, pitches, sr = extract_notes(file_path, mag_threshold=0.1)

    print("\n🎵 Smoothed Extracted Note Sequence (first 100):")
    pprint(notes[:100])

    # Optional: visualize pitch over time
    plt.figure(figsize=(14, 5))
    times = np.arange(len(pitches)) * (512 / sr)  # assuming hop_length=512 default in piptrack
    plt.plot(times, pitches, label='Pitch Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title('Pitch Over Time')
    plt.grid(True)
    plt.show()
