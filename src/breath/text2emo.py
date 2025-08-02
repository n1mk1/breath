import os
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from collections import defaultdict

nltk.download('punkt')
nltk.download('punkt_tab')  # 👈 Add this line


# Get current script directory
current_dir = os.path.dirname(__file__)
file_path = os.path.join(current_dir, "NRC-Emotion-Lexicon-Wordlevel-v0.92.txt")

# Load NRC Emotion Lexicon using relative path
nrc = pd.read_csv(file_path, 
                  names=["word", "emotion", "association"], 
                  sep='\t')

# Filter for words that are associated with emotions
nrc = nrc[nrc["association"] == 1]

# Convert to dictionary: {word: [emotions]}
emotion_dict = defaultdict(list)
for _, row in nrc.iterrows():
    emotion_dict[row["word"]].append(row["emotion"])

# Sample input text
text = "iam doing sooo much homework, its a lot but iam completing them it"

# Tokenize and lower
tokens = word_tokenize(text.lower())

# Detect emotions
emotion_results = defaultdict(int)
emotion_words = defaultdict(list)

for token in tokens:
    if token in emotion_dict:
        for emotion in emotion_dict[token]:
            emotion_results[emotion] += 1
            emotion_words[emotion].append(token)

# Output results
print("Emotion Frequency:")
print(dict(emotion_results))

print("\nWords associated with each emotion:")
for emo, words in emotion_words.items():
    print(f"{emo}: {words}")
