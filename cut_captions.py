# import pandas as pd
# from transformers import CLIPTokenizer
#
# # Pfad zur CSV
# csv_file = "flickr30k_captions.csv"
# df = pd.read_csv(csv_file, sep=";")  # falls andere Trennzeichen, anpassen
#
# # CLIP Tokenizer laden
# tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
# max_len = 70
#
# # Funktion zum Kürzen
# def truncate_caption(text):
#     tokens = tokenizer.encode(text, add_special_tokens=False)
#     if len(tokens) > max_len:
#         tokens = tokens[:max_len]
#     return tokenizer.decode(tokens)
#
# # Captions kürzen
# df["caption"] = df["caption"].apply(truncate_caption)
#
# # Neue CSV speichern
# df.to_csv("flickr30k_captions_truncated.csv", sep=";", index=False)
# print("Truncated CSV gespeichert als flickr30k_captions_truncated.csv")


# import pandas as pd
#
# csv_file = "flickr30k_captions.csv"
# df = pd.read_csv(csv_file, sep=";")  # anpassen, falls anderes Trennzeichen
#
# max_words = 60
#
# def truncate_caption_words(text):
#     words = text.split()
#     if len(words) > max_words:
#         words = words[:max_words]
#     return " ".join(words)
#
# df["caption"] = df["caption"].apply(truncate_caption_words)
#
# df.to_csv("flickr30k_captions_truncated.csv", sep=";", index=False)
# print("Truncated CSV gespeichert als flickr30k_captions_truncated.csv")
#
# import pandas as pd
# csv_file = "flickr30k_captions_truncated.csv"
# df = pd.read_csv(csv_file, sep=";")
#
# # count length of captions in words
# for index, row in df.iterrows():
#     caption = row["caption"]
#     word_count = len(caption.split())
#     if word_count > 70:
#         print(f"Caption {index} has {word_count} words.")


import pandas as pd
from transformers import CLIPTokenizer

# CSV-Datei laden
csv_file = "flickr30k_captions.csv"
df = pd.read_csv(csv_file, sep=";")  # anpassen, falls anderes Trennzeichen

# CLIP Tokenizer laden
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
max_tokens = 70  # max_position_embeddings für CLIP

def truncate_caption_tokens(text):
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        tokens = tokens[:max_tokens]
    return tokenizer.decode(tokens)

# Captions kürzen
df["caption"] = df["caption"].apply(truncate_caption_tokens)

# Neue CSV speichern
df.to_csv("flickr30k_captions_truncated.csv", sep=";", index=False)
print("Truncated CSV gespeichert als flickr30k_captions_truncated.csv")