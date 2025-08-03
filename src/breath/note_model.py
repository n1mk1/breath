import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import pickle
import os

# ------------------ Config ------------------
SEQ_LENGTH = 20
BATCH_SIZE = 32
EPOCHS = 20
EMBED_DIM = 16
HIDDEN_DIM = 64
MODEL_PATH = "note_lstm_model.pt"

# ------------------ Dataset ------------------
class NoteDataset(Dataset):
    def __init__(self, notes, seq_len, label_encoder):
        self.inputs = []
        self.targets = []
        encoded = label_encoder.transform(notes)

        for i in range(len(encoded) - seq_len):
            self.inputs.append(encoded[i:i + seq_len])
            self.targets.append(encoded[i + seq_len])

        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]

# ------------------ LSTM Model ------------------
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

# ------------------ Helper ------------------
def is_iterable_but_not_string(obj):
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))

# ------------------ Train ------------------
def train_model(notes):
    print("Debug: First 10 elements of notes before processing:")
    print(notes[:10])
    print("Debug: Types of first 10 elements:")
    print([type(n) for n in notes[:10]])

    # Flatten notes safely if needed
    if len(notes) > 0 and is_iterable_but_not_string(notes[0]):
        flat_notes = []
        for item in notes:
            if is_iterable_but_not_string(item):
                flat_notes.extend(item)
            else:
                flat_notes.append(item)
        notes = flat_notes

    # Remove empty or None entries
    notes = [n for n in notes if n]

    print("Debug: First 10 elements of notes after flattening and cleaning:")
    print(notes[:10])

    label_encoder = LabelEncoder()
    label_encoder.fit(notes)
    vocab_size = len(label_encoder.classes_)

    dataset = NoteDataset(notes, SEQ_LENGTH, label_encoder)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = NoteLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCHS):
        total_loss = 0
        model.train()
        for x, y in dataloader:
            optimizer.zero_grad()
            preds = model(x)
            loss = loss_fn(preds, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss:.4f}")

    # Save model and encoder
    torch.save(model.state_dict(), MODEL_PATH)
    with open("note_label_encoder.pkl", "wb") as f:
        pickle.dump(label_encoder, f)

    print("✅ Model training complete and saved.")

# ------------------ Example ------------------
if __name__ == "__main__":
    import breathing_to_notes
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, "calm.mp3")
    notes = breathing_to_notes.extract_notes(file_path)
    train_model(notes)
