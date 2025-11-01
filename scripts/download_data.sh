#!/bin/bash
# Erstellt Ordner und lädt Flickr8k herunter
cd ..
mkdir data/raw
#wget https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/Flickr8k_Dataset.zip -O data/raw/flickr8k.zip
curl -L -o ~/ADL_multimodal_retrieval/data/raw/flickr8k.zip\
  https://www.kaggle.com/api/v1/datasets/download/adityajn105/flickr8k
unzip data/raw/flickr8k.zip -d data/raw/

chmod +x scripts/download_data.sh

