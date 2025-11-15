import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
import random, os
from pathlib import Path
from collections import Counter
import re
import spacy

# --- 2. Pfade dynamisch ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
print(PROJECT_ROOT)
DATA_DIR = PROJECT_ROOT / "data"
print(DATA_DIR)
IMG_DIR = DATA_DIR / "Images"
print(IMG_DIR)
CAPTIONS_FILE = DATA_DIR / "flickr8k_captions.csv"
print(CAPTIONS_FILE)


df = pd.read_csv(str(CAPTIONS_FILE), encoding='utf-8', sep=';', engine='python')
# print(df.head())

for i in range(3):
    row = df.sample(1).iloc[0]
    img = Image.open(os.path.join(IMG_DIR, row['filename']))
    plt.imshow(img)
    plt.title(row['caption'])
    plt.axis('off')
    plt.show()


# alle Captions zu einem Text zusammenfassen
text = " ".join(df["caption"].astype(str)).lower()

# nur Wörter extrahieren
words = re.findall(r"\b\w+\b", text)

# Häufigkeiten zählen
freq = Counter(words)


top30 = freq.most_common(30)

plt.bar([w for w,_ in top30], [c for _,c in top30])
plt.xticks(rotation=90)
plt.show()



nlp = spacy.load("en_core_web_sm")

doc = nlp(text)

allowed = {"NOUN", "VERB", "ADJ"}
filtered = [token.lemma_.lower() for token in doc
            if token.pos_ in allowed and token.is_alpha]

freq_pos = Counter(filtered)
top30_pos = freq_pos.most_common(30)

plt.bar([w for w,_ in top30_pos], [c for _,c in top30_pos])
plt.xticks(rotation=90)
plt.show()