import pandas as pd
from transformers import CLIPTokenizer

# Pfad zur CSV
csv_file = "flickr30k_captions.csv"
df = pd.read_csv(csv_file, sep=";")  # falls andere Trennzeichen, anpassen

# CLIP Tokenizer laden
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
max_len = 77

# Funktion zum Kürzen
def truncate_caption(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    return tokenizer.decode(tokens)

# Captions kürzen
df["caption"] = df["caption"].apply(truncate_caption)

# Neue CSV speichern
df.to_csv("flickr30k_captions_truncated.csv", sep=";", index=False)
print("Truncated CSV gespeichert als flickr39k_captions_truncated.csv")