import torch
import torch.nn as nn
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
import breathing_to_notes

# ------------------ Config ------------------
SEQ_LENGTH = 20
EMBED_DIM = 16
HIDDEN_DIM = 64
MODEL_PATH = "note_lstm_model.pt"

# ------------------ Device ------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Model ------------------
class NoteLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# ------------------ Load model ------------------
with open("note_label_encoder.pkl", "rb") as f:
    label_encoder: LabelEncoder = pickle.load(f)

vocab_size = len(label_encoder.classes_)
model = NoteLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ------------------ Anomaly Score ------------------
def anomaly_scores(note_sequence):
    encoded = [label_encoder.transform([n])[0] for n in note_sequence if n in label_encoder.classes_]
    scores = []

    for i in range(len(encoded) - SEQ_LENGTH):
        seq = torch.tensor(encoded[i:i+SEQ_LENGTH]).unsqueeze(0).to(device)
        target = encoded[i + SEQ_LENGTH]

        with torch.no_grad():
            output = model(seq)
            probs = torch.softmax(output, dim=1).squeeze()
            score = 1 - probs[target].item()
            scores.append(score)

    return scores

# ------------------ Compare Files ------------------
def compare_files(base_file, test_file):
    try:
        print(f"\n🔍 Comparing: {os.path.basename(base_file)} ↔ {os.path.basename(test_file)}")

        base_notes, _, _ = breathing_to_notes.extract_notes(base_file)
        test_notes, _, _ = breathing_to_notes.extract_notes(test_file)

        # Filter unknown notes
        base_notes = [n for n in base_notes if n in label_encoder.classes_]
        test_notes = [n for n in test_notes if n in label_encoder.classes_]

        # Check length
        if len(base_notes) <= SEQ_LENGTH or len(test_notes) <= SEQ_LENGTH:
            print("Not enough note data to compare.")
            return

        base_scores = anomaly_scores(base_notes)
        test_scores = anomaly_scores(test_notes)

        # ------------------ Plot ------------------
        plt.figure(figsize=(14, 5))
        plt.plot(base_scores, label="Normal (Calm)", color='green')
        plt.plot(test_scores, label="Test (Anxious?)", color='red', alpha=0.7)
        plt.axhline(0.5, color='gray', linestyle='--', label="Anomaly Threshold")
        plt.title("Anomaly Score Comparison")
        plt.xlabel("Note Index")
        plt.ylabel("Anomaly Score")
        plt.legend()
        plt.tight_layout()
        plt.show()

        # ------------------ Stats ------------------
        def stats(scores):
            return {
                "max": round(float(np.max(scores)), 3),
                "mean": round(float(np.mean(scores)), 3),
                "anomalous (%)": round(100 * np.mean(np.array(scores) > 0.5), 2)
            }

        print("Normal Stats:", stats(base_scores))
        print("Test Stats:", stats(test_scores))

    except Exception as e:
        print("Error during comparison:", str(e))

# ------------------ Main ------------------
if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    base_file = os.path.join(current_dir, "calm.mp3")
    test_file = os.path.join(current_dir, "anxious.mp3")

    assert os.path.isfile(base_file), f"Missing file: {base_file}"
    assert os.path.isfile(test_file), f"Missing file: {test_file}"

    compare_files(base_file, test_file)
